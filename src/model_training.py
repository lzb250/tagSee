# src/model_training.py
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    AutoModelForTokenClassification,
    BertForTokenClassification,
    BertConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Any
from transformers import TrainerCallback

from config.model_config import get_model_path
from src.cross_platform_utils import safe_print


class SkillExtractionDataset(Dataset):
    def __init__(self, texts: List[List[str]], labels: List[List[str]],
                 tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        self.labels = []

        # 标签映射
        self.label2id = {'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}
        self.id2label = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}

        for text, label_list in zip(texts, labels):
            if len(text) == 0:
                continue

            # 编码
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                is_split_into_words=True
            )

            # 对齐标签
            word_ids = encoding.word_ids()
            aligned_labels = []

            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100) # 忽略特殊token
                else:
                    if word_idx < len(label_list):
                        aligned_labels.append(self.label2id[label_list[word_idx]])
                    else:
                        aligned_labels.append(-100)

            encoding['labels'] = aligned_labels
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.encodings[idx].items()}

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # 移除忽略的标签
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for p, l in zip(prediction, label):
            if l != -100:
                true_predictions.append(p)
                true_labels.append(l)

    if len(true_labels) == 0:
        return {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class MetricsDisplayCallback(TrainerCallback):
    """自定义回调函数：以表格形式显示评估指标"""

    def __init__(self):
        self.all_eval_results = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self._print_metrics_table(metrics, state.epoch)
            self.all_eval_results.append({
                'epoch': state.epoch,
                'metrics': metrics.copy()
            })

    def on_train_end(self, args, state, control, **kwargs):
        if self.all_eval_results:
            safe_print("\n" + "="*60)
            safe_print("训练完成 - 评估指标汇总")
            safe_print("="*60)
            self._print_summary_table()
            safe_print("="*60)

    def _print_metrics_table(self, metrics: Dict[str, Any], epoch: float = None):
        title = f"\n{'Epoch '+str(epoch):^10}" if epoch is not None else "\n评估结果"
        safe_print(title)
        safe_print("-" * 40)
        safe_print(f"{'指标':^10} | {'值':^20}")
        safe_print("-" * 40)

        metric_mapping = {
            'eval_accuracy': '准确性',
            'eval_f1': 'F1分数',
            'eval_precision': '精确率',
            'eval_recall': '召回率'
        }

        for eval_key, cn_name in metric_mapping.items():
            value = metrics.get(eval_key, 0)
            safe_print(f"{cn_name:^10} | {value:^20.4f}")
        safe_print("-" * 40)

    def _print_summary_table(self):
        if not self.all_eval_results:
            return

        metric_mapping = {
            'eval_accuracy': '准确性',
            'eval_f1': 'F1分数',
            'eval_precision': '精确率',
            'eval_recall': '召回率'
        }

        header = "Epoch "
        for cn_name in metric_mapping.values():
            header += f"{cn_name:>12}"
        safe_print(header)
        safe_print("-" * len(header))

        for result in self.all_eval_results:
            epoch_str = f"{result['epoch']:>10.1f} "
            for eval_key in metric_mapping.keys():
                value = result['metrics'].get(eval_key, 0)
                epoch_str += f"{value:>12.4f}"
            safe_print(epoch_str)

def train_skill_extraction_model(
        train_texts: List[List[str]],
        train_labels: List[List[str]],
        val_texts: List[List[str]],
        val_labels: List[List[str]],
        model_name: str = "bert-base-chinese",
        output_dir: str = "./models/skill_extraction_model",
        num_epochs: int = 15,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True, # 显卡环境下默认为True，开启混合精度训练
        max_grad_norm: float = 1.0
):
    """
    训练技能提取模型（GPU 优化版）
    """
    from .cross_platform_utils import ensure_directory_exists
    ensure_directory_exists(output_dir)

    # 显卡检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    safe_print(f"🚀 使用设备: {device.upper()}")
    if device == "cuda":
        safe_print(f"📸 显卡型号: {torch.cuda.get_device_name(0)}")
        # 显存优化：如果不启用 FP16 但检测到显卡，建议还是开启
        if not fp16:
            safe_print("💡 建议开启 fp16 以提升显卡训练速度并降低显存占用")
    else:
        fp16 = False # CPU 环境下强制关闭 fp16

    # 加载模型配置
    config = BertConfig(
        vocab_size=21128,
        num_labels=3,
        id2label={0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'},
        label2id={'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}
    )

    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
        model = BertForTokenClassification.from_pretrained(
            model_name,
            config=config,
            local_files_only=True
        ).to(device)
        safe_print("✓ 模型加载成功（离线模式）")
    except Exception as e:
        safe_print(f"离线加载失败，尝试从路径加载: {e}")
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForTokenClassification.from_pretrained(model_name, config=config).to(device)

    train_dataset = SkillExtractionDataset(train_texts, train_labels, tokenizer)
    val_dataset = SkillExtractionDataset(val_texts, val_labels, tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 计算自适应步数
    num_training_steps = len(train_dataset) * num_epochs // batch_size // gradient_accumulation_steps
    warmup_steps = int(num_training_steps * warmup_ratio)

    # 训练参数
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        logging_dir=f"{output_dir}/logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=1,        # 减少保存的检查点数量，节省内存和磁盘
        save_safetensors=False,    # 【关键】如果 safetensors 报错，关闭它改用 bin 格式
        logging_steps=50,
        seed=42,
        dataloader_pin_memory=True if device == "cuda" else False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsDisplayCallback()]
    )

    print(f"开始在 {device.upper()} 上训练模型...")
    trainer.train()

    # 保存模型
    final_output = f"{output_dir}_final"
    ensure_directory_exists(final_output)
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    print(f"模型已保存到: {final_output}")
    return trainer, final_output
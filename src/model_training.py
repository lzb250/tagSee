"""
文件路径: src/model_training.py
功能解析: 模型训练模块

主要功能:
1. SkillExtractionDataset 类
   - 继承自 PyTorch Dataset，用于训练数据加载
   - 使用 BERT tokenizer 将文本序列转换为 token IDs
   - 对齐标签序列，处理特殊 token（[CLS], [SEP], [PAD]）

2. 核心方法
   - train_skill_extraction_model(): 训练技能提取模型
     - 支持 BERT-base-chinese 或自定义模型路径
     - 配置训练参数（epochs, batch_size 等）
     - 使用 HuggingFace Trainer 进行训练
     - 自动保存最佳模型（基于 F1 分数）
     - 兼容新旧版本 transformers 库
   
   - compute_metrics(): 计算评估指标
     - 准确率
     - 精确率
     - 召回率
     - F1 分数

3. 训练流程
   - 加载预训练 BERT 模型和分词器
   - 创建训练集和验证集 Dataset
   - 设置训练参数和优化器
   - 自动评估和保存最佳模型

使用场景:
- 离线模型训练
- 模型微调
- 跨平台训练（支持 CUDA、MPS、CPU）
---
"""

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

            # 不使用 return_offset_mapping，直接编码
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
        pred_list = []
        label_list = []
        for p, l in zip(prediction, label):
            if l != -100:
                pred_list.append(p)
                label_list.append(l)
        true_predictions.append(pred_list)
        true_labels.append(label_list)

    # 展平
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    if len(flat_labels) == 0:
        return {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(flat_labels, flat_predictions)

    """返回评估指标结果

    包含的指标：
    - accuracy: 准确率，预测正确的样本占总样本的比例
    - precision: 精确率，预测为正例中真正为正例的比例
    - recall: 召回率，真正为正例中被正确预测为正例的比例
    - f1: F1分数，精确率和召回率的调和平均值
    """
    return {
        'accuracy': accuracy, # 保留英文键名用于HuggingFace内部使用
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class MetricsDisplayCallback(TrainerCallback):
    """自定义回调函数：以表格形式显示评估指标"""

    def __init__(self):
        self.all_eval_results = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """在每次评估时调用"""
        if metrics is not None:
            # 打印评估指标的表格
            self._print_metrics_table(metrics, state.epoch)
            # 保存结果用于最终汇总
            self.all_eval_results.append({
                'epoch': state.epoch,
                'metrics': metrics.copy()
            })

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时打印汇总表格"""
        if self.all_eval_results:
            safe_print("\n" + "="*60)
            safe_print("训练完成 - 评估指标汇总")
            safe_print("="*60)
            self._print_summary_table()
            safe_print("="*60)

    def _print_metrics_table(self, metrics: Dict[str, Any], epoch: float = None):
        """打印单个评估结果的表格"""
        title = f"\n{'Epoch '+str(epoch):^10}" if epoch is not None else "\n评估结果"
        safe_print(title)
        safe_print("-" * 40)
        safe_print(f"{'指标':^10} | {'值':^20}")
        safe_print("-" * 40)

        # HuggingFace Trainer 传递的 metrics 字典使用 eval_ 前缀
        metric_mapping = {
            'eval_accuracy': '准确性',
            'eval_f1': 'F1分数',
            'eval_precision': '精确率',
            'eval_recall': '召回率'
        }

        for eval_key, cn_name in metric_mapping.items():
            # 尝试两种键名：with eval_ prefix 和 without
            value = metrics.get(eval_key, 0)
            if value == 0 and not eval_key.startswith('eval_'):
                # 兼容旧版本，尝试不带前缀的键名
                value = metrics.get(eval_key, 0)
            safe_print(f"{cn_name:^10} | {value:^20.4f}")
        safe_print("-" * 40)

    def _print_summary_table(self):
        """打印所有评估结果的汇总表格"""
        if not self.all_eval_results:
            return

        # HuggingFace Trainer 传递的 metrics 字典使用 eval_ 前缀
        metric_mapping = {
            'eval_accuracy': '准确性',
            'eval_f1': 'F1分数',
            'eval_precision': '精确率',
            'eval_recall': '召回率'
        }

        # 添加 epoch 列
        header = "Epoch "
        for cn_name in metric_mapping.values():
            header += f"{cn_name:>12}"
        safe_print(header)
        safe_print("-" * len(header))

        for result in self.all_eval_results:
            epoch_str = f"{result['epoch']:>10.1f} "
            for eval_key in metric_mapping.keys():
                # 尝试从 metrics 中获取值
                value = result['metrics'].get(eval_key, 0)
                if value == 0:
                    # 兼容旧版本，尝试不带前缀的键名
                    key_without_prefix = eval_key.replace('eval_', '')
                    value = result['metrics'].get(key_without_prefix, 0)
                epoch_str += f"{value:>12.4f}"
            safe_print(epoch_str)

def train_skill_extraction_model(
        train_texts: List[List[str]],
        train_labels: List[List[str]],
        val_texts: List[List[str]],
        val_labels: List[List[str]],
        model_name: str = "bert-base-chinese", # 这里现在直接是本地路径
        output_dir: str = "./models/skill_extraction_model",
        num_epochs: int = 15, # 增加训练轮数
        batch_size: int = 16,
        learning_rate: float = 2e-5, # 优化学习率
        warmup_ratio: float = 0.1, # 使用比例而非固定步数
        weight_decay: float = 0.01, # 增加正则化
        gradient_accumulation_steps: int = 1, # 梯度累积
        fp16: bool = False,
        max_grad_norm: float = 1.0 # 梯度裁剪
):
    """
    训练技能提取模型（完全优化版）

    改进点：
    1. 自适应 warmup（基于数据量）
    2. 学习率优化（2e-5 更适合中文BERT）
    3. 增加 epochs（15轮充分训练）
    4. 添加权重衰减防止过拟合
    5. 梯度裁剪防止梯度爆炸
    6. 混合精度训练支持
    7. 更好的评估指标（F1优先）
    """

    from .cross_platform_utils import ensure_directory_exists

    # 确保输出目录存在
    ensure_directory_exists(output_dir)

    safe_print(f"加载模型: {model_name}")

    # 创建配置，确保使用中文BERT的正确配置
    config = BertConfig(
        vocab_size=21128, # 中文BERT的词表大小
        num_labels=3,
        id2label={0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'},
        label2id={'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}
    )

    # 直接使用传入的路径，强制离线模式
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
        model = BertForTokenClassification.from_pretrained(
            model_name,
            config=config,
            local_files_only=True
        )
        safe_print("✓ 模型加载成功（离线模式）")
    except Exception as e:
        safe_print(f"离线加载失败: {e}")
        safe_print("尝试不使用 local_files_only 参数...")
        # 如果 local_files_only 失败，尝试普通加载
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForTokenClassification.from_pretrained(
            model_name,
            config=config
        )

    # 创建数据集
    train_dataset = SkillExtractionDataset(train_texts, train_labels, tokenizer)
    val_dataset = SkillExtractionDataset(val_texts, val_labels, tokenizer)

    # 数据整理器
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 计算总训练步数和 warmup 步数（自适应）
    num_training_steps = len(train_dataset) * num_epochs // batch_size // gradient_accumulation_steps
    warmup_steps = int(num_training_steps * warmup_ratio)

    safe_print(f"总训练步数: {num_training_steps}")
    safe_print(f"Warmup 步数: {warmup_steps} ({warmup_ratio*100:.0f}%)")

    # 训练参数（完全优化版）
    # 根据不同版本的transformers库使用兼容的参数名
    try:
        # 新版本参数（优化后）
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
            eval_strategy="epoch", # 新版本使用 eval_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3, # 保留更多checkpoint
            logging_steps=50, # 更频繁的日志
            seed=42, # 固定随机种子
            dataloader_pin_memory=False, # 优化内存使用
            report_to="none", # 禁用wandb等日志
        )
    except TypeError:
        # 旧版本参数（兼容）
        safe_print("使用旧版本的evaluation_strategy参数...")
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
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            logging_steps=50,
            seed=42,
            report_to="none",
        )

    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsDisplayCallback()] # 添加自定义回调以表格形式显示指标
    )

    # 开始训练
    print("开始训练模型...")
    trainer.train()

    # 保存最终模型
    final_output = f"{output_dir}_final"
    ensure_directory_exists(final_output)
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    print(f"模型已保存到: {final_output}")
    return trainer, final_output


# 使用示例
if __name__ == "__main__":
    # 这里可以添加训练示例代码
    pass
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
from typing import List

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
                    aligned_labels.append(-100)  # 忽略特殊token
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

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_skill_extraction_model(
        train_texts: List[List[str]],
        train_labels: List[List[str]],
        val_texts: List[List[str]],
        val_labels: List[List[str]],
        model_name: str = "bert-base-chinese",  # 这里现在直接是本地路径
        output_dir: str = "./models/skill_extraction_model",
        num_epochs: int = 3,
        batch_size: int = 16
):
    """训练技能提取模型（支持离线模式）"""

    from .cross_platform_utils import ensure_directory_exists

    # 确保输出目录存在
    ensure_directory_exists(output_dir)

    safe_print(f"加载模型: {model_name}")

    # 创建配置，确保使用中文BERT的正确配置
    config = BertConfig(
        vocab_size=21128,  # 中文BERT的词表大小
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

    # 训练参数
    # 根据不同版本的transformers库使用兼容的参数名
    try:
        # 尝试使用新版本的参数名
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            eval_strategy="epoch",  # 新版本使用 eval_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=100,
            report_to="none"  # 禁用wandb等日志
        )
    except TypeError:
        # 如果失败，使用旧版本的参数名
        safe_print("使用旧版本的evaluation_strategy参数...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            evaluation_strategy="epoch",  # 旧版本使用 evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=100,
            report_to="none"  # 禁用wandb等日志
        )

    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
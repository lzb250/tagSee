#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.cross_platform_utils import setup_encoding, safe_print
from src.data_preprocessing import DataPreprocessor
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from src.model_training import train_skill_extraction_model

def main():
    setup_encoding()

    parser = argparse.ArgumentParser(description='简历技能提取模型训练 - 完全优化版')
    parser.add_argument('--synthetic_samples', type=int, default=5000, help='生成合成样本数量')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup比例')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--skill_count_range', type=int, nargs=2, default=[3,8], help='每条样本技能数量范围')
    parser.add_argument('--save_samples', action='store_true', help='保存生成的简历样本')
    parser.add_argument('--test_train', action='store_true', help='测试模式')
    output_dir = datetime.now().strftime('./models/skill_extraction_model_%Y%m%d_%H%M')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='模型输出目录')

    args = parser.parse_args()

    if args.test_train:
        safe_print("🔹 测试模式：快速验证流程")
        args.synthetic_samples = 100
        args.epochs = 2
        args.output_dir += "_test"

    safe_print("="*80)
    safe_print("🔹 开始训练流程")
    safe_print("="*80)
    safe_print(f"训练配置: 样本={args.synthetic_samples}, 轮数={args.epochs}, 批次={args.batch_size}, 输出={args.output_dir}")
    safe_print("="*80)

    safe_print("\n[步骤1] 准备训练数据...")
    preprocessor = DataPreprocessor()
    texts, labels = preprocessor.generate_synthetic_data(num_samples=args.synthetic_samples,
                                                         skill_count_range=tuple(args.skill_count_range),
                                                         save_samples=args.save_samples)

    if len(texts) == 0:
        safe_print("❌ 无法生成训练数据")
        sys.exit(1)
    safe_print(f"✅ 生成 {len(texts)} 个训练样本")

    safe_print("\n[步骤2] 划分数据集...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )
    safe_print(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")

    safe_print("\n[步骤3] 加载模型...")
    local_model_path = "./models/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(local_model_path, local_files_only=True)
    config = BertConfig(
        vocab_size=21128,
        num_labels=3,
        id2label={0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'},
        label2id={'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}
    )
    model = BertForTokenClassification.from_pretrained(local_model_path, config=config, local_files_only=True)
    safe_print("✅ 本地模型加载成功")

    safe_print("\n[步骤4] 开始训练...")
    trainer, final_model_path = train_skill_extraction_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        model_name=local_model_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay
    )

    safe_print("\n" + "="*80)
    safe_print("🔹 训练完成!")
    safe_print(f"✅ 模型保存在: {final_model_path}")
    safe_print("="*80)

if __name__ == "__main__":
    main()

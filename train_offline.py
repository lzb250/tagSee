#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from src.cross_platform_utils import setup_encoding, safe_print
from src.data_preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split

def main():
    setup_encoding()

    parser = argparse.ArgumentParser(description='离线训练简历技能提取模型')
    parser.add_argument('--synthetic_samples', type=int, default=2000,
                        help='生成合成样本数量')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='批次大小')
    parser.add_argument('--output_dir', type=str, default='./models/skill_extraction_model',
                        help='模型输出目录')

    args = parser.parse_args()



    safe_print("开始准备训练数据...")

    # 创建数据预处理器
    preprocessor = DataPreprocessor()

    # 生成合成数据
    texts, labels = preprocessor.generate_synthetic_data(args.synthetic_samples)

    if len(texts) == 0:
        safe_print("错误: 无法生成训练数据")
        sys.exit(1)

    safe_print(f"生成了 {len(texts)} 个训练样本")

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    safe_print(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")

    # 现在加载模型（使用本地路径）
    safe_print("正在加载本地BERT模型...")

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from src.model_training import train_skill_extraction_model

        # 直接使用本地路径，不通过配置
        local_model_path = "./models/bert-base-chinese"

        # 先测试模型是否能加载
        safe_print(f"测试加载模型: {local_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            local_model_path,
            num_labels=3,
            id2label={0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'},
            label2id={'O': 0, 'B-SKILL': 1, 'I-SKILL': 2},
            local_files_only=True
        )
        safe_print("✓ 本地模型加载成功")

        # 训练模型
        trainer, final_model_path = train_skill_extraction_model(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            model_name=local_model_path,  # 直接传入本地路径
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )

        safe_print(f"训练完成! 模型保存在: {final_model_path}")

    except Exception as e:
        safe_print(f"训练过程中出现错误: {e}")
        # 打印详细错误信息
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
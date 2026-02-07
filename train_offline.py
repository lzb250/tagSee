"""

文件路径: train_offline.py

功能解析: 离线训练主脚本



主要功能:

1. 训练流程

   - 使用 DataPreprocessor 生成合成训练数据

   - 划分训练集和验证集（80:20）

   - 加载本地 BERT-base-chinese 模型

   - 训练技能提取模型（Shower 风格 NER）

   - 自动评估和保存最佳模型

   - 生成带时间戳的模型目录



2. 命令行参数

   --synthetic_samples: 生成合成样本数量（默认: 2000）

   --epochs: 训练轮数（默认: 10）

   --batch_size: 批次大小（默认: 12）

   --output_dir: 模型输出目录（默认: 自动生成）



3. 使用示例

   # 使用默认参数训练

   python train_offline.py



   # 自定义参数训练

   python train_offline.py --synthetic_samples 5000 --epochs 20 --batch_size 16



4. 模型输出

   训练过程中会在 models/ 目录下创建：

   - skill_extraction_model_YYYYMMDDHHMM_final （最佳模型）

   - skill_extraction_model_YYYYMMDDHHMM/checkpoint-* （检查点）



使用场景:

- 离线环境模型训练

- 批量模型训练

- 模型实验和对比

---

"""



#!/usr/bin/env python3

# -*- coding: utf-8 -*-


import argparse
import sys

from sklearn.model_selection import train_test_split

from src.cross_platform_utils import setup_encoding, safe_print
from src.data_preprocessing import DataPreprocessor


def main():

    setup_encoding()



    parser = argparse.ArgumentParser(description='简历技能提取模型训练 - 完全优化版')

    parser.add_argument('--synthetic_samples', type=int, default=5000,

                        help='生成合成样本数量（默认5000，建议5000-10000）')

    parser.add_argument('--epochs', type=int, default=15,

                        help='训练轮数（默认15，建议10-20）')

    parser.add_argument('--batch_size', type=int, default=16,

                        help='批次大小（默认16，显存不足可改8）')

    parser.add_argument('--learning_rate', type=float, default=2e-5,

                        help='学习率（默认2e-5）')

    parser.add_argument('--warmup_ratio', type=float, default=0.1,

                        help='Warmup比例（默认10%%）')

    parser.add_argument('--weight_decay', type=float, default=0.01,

                        help='权重衰减（默认0.01）')

    parser.add_argument('--skill_count_range', type=int, nargs=2, default=[3, 8],

                        help='每条样本包含的技能数量范围（默认3-8）')

    parser.add_argument('--save_samples', action='store_true',

                        help='保存生成的简历样本到 data/synthetic_resumes/ 目录')

    parser.add_argument('--test_train', action='store_true',

                        help='测试模式：少量样本快速验证流程')

    from datetime import datetime

    output_dir = datetime.now().strftime('./models/skill_extraction_model_%Y%m%d_%H%M')

    parser.add_argument('--output_dir', type=str, default=output_dir,

                        help='模型输出目录')



    args = parser.parse_args()







    # 测试模式：减少样本快速验证

    if args.test_train:

        safe_print("��� 测试模式：使用少量样本快速验证流程")

        args.synthetic_samples = 100

        args.epochs = 2

        args.output_dir = args.output_dir + "_test"



    safe_print("="*80)

    safe_print("��� 开始训练流程")

    safe_print("="*80)

    safe_print(f"��� 训练配置:")

    safe_print(f"   - 样本数量: {args.synthetic_samples}")

    safe_print(f"   - 训练轮数: {args.epochs}")

    safe_print(f"   - 批次大小: {args.batch_size}")

    safe_print(f"   - 学习率: {args.learning_rate}")

    safe_print(f"   - 技能数量范围: {args.skill_count_range[0]}-{args.skill_count_range[1]}")

    safe_print(f"   - 输出目录: {args.output_dir}")

    safe_print("="*80)



    safe_print("\n[步骤1] 准备训练数据...")



    # 创建数据预处理器

    preprocessor = DataPreprocessor()



    # 生成合成数据（使用新的 skill_count_range 参数）

    skill_range = tuple(args.skill_count_range)

    texts, labels = preprocessor.generate_synthetic_data(

        num_samples=args.synthetic_samples,

        skill_count_range=skill_range,

        save_samples=args.save_samples

    )



    if args.save_samples:

        safe_print(f"��� 简历样本已保存到 data/synthetic_resumes/ 目录")



    if len(texts) == 0:

        safe_print("❌ 错误: 无法生成训练数据")

        sys.exit(1)



    safe_print(f"✅ 生成了 {len(texts)} 个训练样本")



    # 统计信息

    skill_labels = [l for label_list in labels for l in label_list if l != 'O']

    safe_print(f"   - 总token数: {sum(len(t) for t in texts)}")

    safe_print(f"   - 技能token数: {len(skill_labels)}")

    safe_print(f"   - 技能占比: {len(skill_labels)/sum(len(t) for t in texts)*100:.2f}%")



    # 划分训练集和验证集

    safe_print("\n[步骤2] 划分数据集...")

    train_texts, val_texts, train_labels, val_labels = train_test_split(

        texts, labels, test_size=0.2, random_state=42, stratify=None

    )



    safe_print(f"✅ 数据集划分完成:")

    safe_print(f"   - 训练集: {len(train_texts)} 条")

    safe_print(f"   - 验证集: {len(val_texts)} 条")



    # 加载模型（使用本地路径）

    safe_print("\n[步骤3] 加载模型...")



    try:

        from transformers import BertTokenizer, BertForTokenClassification

        from src.model_training import train_skill_extraction_model



        # 直接使用本地路径，不通过配置

        local_model_path = "./models/bert-base-chinese"



        # 先测试模型是否能加载

        safe_print(f"   模型路径: {local_model_path}")

        tokenizer = BertTokenizer.from_pretrained(local_model_path, local_files_only=True)



        # 创建配置，确保使用中文BERT的正确配置

        from transformers import BertConfig

        config = BertConfig(

            vocab_size=21128,  # 中文BERT的词表大小

            num_labels=3,

            id2label={0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'},

            label2id={'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}

        )



        model = BertForTokenClassification.from_pretrained(

            local_model_path,

            config=config,

            local_files_only=True

        )

        safe_print("✅ 本地模型加载成功")



        # 训练模型（使用优化后的参数）

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

            weight_decay=args.weight_decay,

        )



        safe_print("\n" + "="*80)

        safe_print("��� 训练完成!")

        safe_print("="*80)

        safe_print(f"✅ 最终模型保存在: {final_model_path}")

        safe_print("="*80)



    except Exception as e:

        safe_print(f"训练过程中出现错误: {e}")

        # 打印详细错误信息

        import traceback

        traceback.print_exc()

        sys.exit(1)



if __name__ == "__main__":

    main()
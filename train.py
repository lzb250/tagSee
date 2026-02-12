# train.py

import os
from src.skill_registry import SkillRegistry
from src.industrial_dataset_generator import IndustrialResumeGenerator
from src.train_ner import train
import pandas as pd
import re

# ===========================
# NER 数据构建器
# ===========================
def build_ner_dataset(input_csv="data/generated_resumes.csv",
                      output_csv="data/ner_resume_dataset.csv",
                      registry: SkillRegistry = None):
    """
    构建NER训练数据
    - 使用 SkillRegistry 提取技能
    - 支持别名、版本、重复出现
    """
    if registry is None:
        raise ValueError("SkillRegistry is required!")

    df = pd.read_csv(input_csv)

    records = []

    for _, row in df.iterrows():
        text = str(row["resume_text"])
        labels = ["O"] * len(text)

        # 遍历所有识别出的技能
        for skill in registry.extract_from_text(text):
            # 匹配文本中所有出现位置
            for match in re.finditer(re.escape(skill), text, flags=re.IGNORECASE):
                start, end = match.start(), match.end()
                labels[start] = "B-SKILL"
                for i in range(start + 1, end):
                    labels[i] = "I-SKILL"

        records.append({
            "text": text,
            "labels": " ".join(labels)
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ NER数据已保存 {output_csv}")


# ===========================
# 数据生成 + NER + 训练 流水线
# ===========================
def pipeline(num_samples=1000,
             generated_csv="data/generated_resumes.csv",
             ner_csv="data/ner_resume_dataset.csv",
             skills_csv="config/skills.csv"):
    """
    一键流水线：
    1️⃣ 生成简历数据
    2️⃣ 构建NER训练数据
    3️⃣ 训练BERT模型
    """
    # -------------------
    # 1️⃣ 初始化技能注册中心
    # -------------------
    print("🔹 初始化 SkillRegistry ...")
    registry = SkillRegistry(skills_csv)

    # -------------------
    # 2️⃣ 数据生成
    # -------------------
    print(f"🔹 生成 {num_samples} 条简历数据 ...")
    generator = IndustrialResumeGenerator(registry)
    df = generator.generate_dataset(num_samples)
    generator.save_to_csv(df, generated_csv)

    # -------------------
    # 3️⃣ 构建 NER 数据
    # -------------------
    print("🔹 构建 NER 训练数据 ...")
    build_ner_dataset(input_csv=generated_csv,
                      output_csv=ner_csv,
                      registry=registry)

    # -------------------
    # 4️⃣ 训练模型
    # -------------------
    print("🔹 开始训练 BERT NER 模型 ...")
    train(ner_csv)

    print("✅ 流水线完成！")


# ===========================
# 命令行入口
# ===========================
if __name__ == "__main__":
    pipeline(num_samples=1000)

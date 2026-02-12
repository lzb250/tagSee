# src/ner_dataset_builder.py

import pandas as pd
from src.skill_registry import SkillRegistry
import re


def build_ner_dataset(input_csv="data/generated_resumes.csv",
                      output_csv="data/ner_resume_dataset.csv"):
    """
    构建NER训练数据
    - 使用 SkillRegistry 提取技能
    - 支持别名、版本、重复出现
    """

    registry = SkillRegistry("config/skills.csv")

    df = pd.read_csv(input_csv)

    records = []

    for _, row in df.iterrows():
        text = row["resume_text"]
        labels = ["O"] * len(text)

        # 找出所有技能及起止位置
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

    pd.DataFrame(records).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ NER数据已保存 {output_csv}")


if __name__ == "__main__":
    build_ner_dataset()

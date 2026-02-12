# src/ner_dataset_builder.py

import pandas as pd
from skill_schema import ALL_SKILLS


def build_ner_dataset():
    df = pd.read_csv("data/raw_resume_dataset.csv")

    records = []

    for _, row in df.iterrows():
        text = row["resume_text"]
        labels = ["O"] * len(text)

        for skill in ALL_SKILLS:
            start = text.find(skill)
            if start != -1:
                labels[start] = "B-SKILL"
                for i in range(start + 1, start + len(skill)):
                    labels[i] = "I-SKILL"

        records.append({
            "text": text,
            "labels": " ".join(labels)
        })

    pd.DataFrame(records).to_csv(
        "data/ner_resume_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ NER数据已保存 data/ner_resume_dataset.csv")


if __name__ == "__main__":
    build_ner_dataset()

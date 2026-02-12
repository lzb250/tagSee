# src/dataset_generator.py

import random
import pandas as pd
from skill_schema import ALL_SKILLS


TEMPLATES = [
    "本人熟练掌握{skills}，具有多年开发经验。",
    "熟悉{skills}，参与多个大型项目开发。",
    "精通{skills}，具备良好的系统架构设计能力。",
    "在项目中使用{skills}进行系统开发与优化。",
]


def generate_resume():
    skill_count = random.randint(3, 6)
    skills = random.sample(ALL_SKILLS, skill_count)

    template = random.choice(TEMPLATES)
    text = template.format(skills="、".join(skills))

    noise = "同时具备良好的团队合作精神和问题解决能力。"
    return text + noise, "、".join(skills)


def generate_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        text, skills = generate_resume()
        data.append({
            "resume_text": text,
            "skills": skills
        })

    df = pd.DataFrame(data)
    df.to_csv("data/raw_resume_dataset.csv", index=False, encoding="utf-8-sig")
    print("✅ 原始数据已保存 data/raw_resume_dataset.csv")


if __name__ == "__main__":
    generate_dataset(1000)

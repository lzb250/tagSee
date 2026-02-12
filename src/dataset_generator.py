# src/dataset_generator.py

import random
import pandas as pd
from skill_schema import ALL_SKILLS


EDUCATION_BLOCK = """
教育背景：
2018-2022 本科 计算机科学与技术
"""

EXPERIENCE_TEMPLATES = [
    """
工作经历：
2022-至今 后端开发工程师
负责系统架构设计与开发，主要使用{skills}进行企业级项目开发，
优化数据库性能，提高系统并发能力。
""",
    """
项目经验：
参与多个互联网项目开发，使用{skills}完成系统搭建，
实现高可用分布式架构设计。
""",
    """
技术栈：
熟练掌握{skills}，
具备扎实的数据结构与算法基础。
"""
]

SOFT_SKILL_NOISE = [
    "具备良好的团队合作能力。",
    "拥有较强的问题分析能力。",
    "沟通表达能力优秀。",
    "学习能力强，适应新技术快。"
]


def generate_resume():
    skill_count = random.randint(4, 8)
    skills = random.sample(ALL_SKILLS, skill_count)

    template = random.choice(EXPERIENCE_TEMPLATES)

    text = (
        EDUCATION_BLOCK +
        template.format(skills="、".join(skills)) +
        random.choice(SOFT_SKILL_NOISE)
    )

    return text.strip(), "、".join(skills)


def generate_dataset(num_samples=200):
    data = []

    for i in range(num_samples):
        text, skills = generate_resume()
        data.append({
            "resume_id": i,
            "resume_text": text,
            "skills": skills
        })

    df = pd.DataFrame(data)

    df.to_csv(
        "data/raw_resume_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 已生成数据 data/raw_resume_dataset.csv")
    print("\n📌 示例数据：\n")
    print(df.head(3))


if __name__ == "__main__":
    generate_dataset(200)

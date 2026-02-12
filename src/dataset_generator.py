# src/industrial_dataset_generator.py

import random
import pandas as pd
import os
from skill_schema import ALL_SKILLS


# =========================
# 同义词映射
# =========================
SYNONYM_MAP = {
    "机器学习": ["ML", "MachineLearning"],
    "深度学习": ["DL", "DeepLearning"],
    "计算机视觉": ["CV"],
    "自然语言处理": ["NLP"],
    "SpringBoot": ["Spring Boot"],
    "JavaScript": ["JS"],
    "TypeScript": ["TS"],
}


# =========================
# 噪声文本
# =========================
NOISE_PARAGRAPHS = [
    "本人热爱编程，具有良好的代码规范意识。",
    "曾参与多个大型互联网系统架构设计。",
    "熟悉敏捷开发流程，掌握Git版本控制。",
    "在高并发场景下具备性能调优经验。",
    "参与微服务架构设计与分布式系统开发。",
    "具有良好的沟通能力和团队协作精神。",
    "熟悉Linux服务器部署与运维管理。",
]

EDUCATION_BLOCK = """
教育背景：
2017-2021 本科 计算机科学与技术
"""

PROJECT_BLOCK = """
项目经验：
在多个项目中负责核心模块设计与实现，
优化系统性能，提高系统稳定性。
"""


# =========================
# 技能增强函数
# =========================
def random_disturb_skill(skill):
    """
    对技能做扰动：
    - 随机空格
    - 大小写变化
    - 同义词替换
    """

    # 同义词替换
    if skill in SYNONYM_MAP and random.random() < 0.4:
        skill = random.choice(SYNONYM_MAP[skill])

    # 大小写扰动
    if random.random() < 0.3:
        skill = skill.upper()
    elif random.random() < 0.3:
        skill = skill.lower()

    # 随机插入空格
    if len(skill) > 3 and random.random() < 0.3:
        pos = random.randint(1, len(skill)-1)
        skill = skill[:pos] + " " + skill[pos:]

    return skill


def build_long_noise():
    paragraphs = random.sample(NOISE_PARAGRAPHS, random.randint(2, 5))
    return "\n".join(paragraphs)


# =========================
# 生成单条简历
# =========================
def generate_resume(resume_id):
    skill_count = random.randint(5, 12)
    selected_skills = random.sample(ALL_SKILLS, skill_count)

    # 技能增强版本
    disturbed_skills = [random_disturb_skill(s) for s in selected_skills]

    # 技能随机分散到多个段落
    part1 = "熟练掌握：" + "、".join(disturbed_skills[:len(disturbed_skills)//2])
    part2 = "熟悉技术栈：" + "、".join(disturbed_skills[len(disturbed_skills)//2:])

    # 随机重复某些技能
    if random.random() < 0.5:
        repeat_skill = random.choice(disturbed_skills)
        part2 += f"，在项目中多次使用{repeat_skill}"

    resume_text = (
            EDUCATION_BLOCK +
            build_long_noise() + "\n" +
            PROJECT_BLOCK +
            part1 + "\n" +
            build_long_noise() + "\n" +
            part2 + "\n" +
            build_long_noise()
    )

    return {
        "resume_id": resume_id,
        "resume_text": resume_text.strip(),
        "skills": "、".join(selected_skills)  # 原始标准技能
    }


# =========================
# 批量生成
# =========================
def generate_dataset(num_samples=1000, save_path="data/raw_resume_dataset.csv"):
    os.makedirs("data", exist_ok=True)

    data = []
    for i in range(num_samples):
        data.append(generate_resume(i))

    df = pd.DataFrame(data)

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✅ 成功生成 {num_samples} 条工业级增强简历")
    print(f"📁 保存路径: {save_path}")
    print("\n📌 示例预览:\n")
    print(df.head(2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--save", type=str, default="data/raw_resume_dataset.csv")

    args = parser.parse_args()

    generate_dataset(args.num, args.save)

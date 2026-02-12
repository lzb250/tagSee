# src/industrial_dataset_generator.py

import random
import pandas as pd
from tqdm import tqdm
from src.skill_registry import SkillRegistry


class IndustrialResumeGenerator:
    """
    企业级简历数据增强生成器
    - 完全基于 SkillRegistry
    - 不直接读取 skills.csv
    """

    def __init__(self, skill_registry: SkillRegistry):
        self.registry = skill_registry

        # 所有标准技能
        self.all_skills = list(self.registry.standard_index.keys())

        # 分类索引
        self.category_index = self.registry.category_index

    # =====================================
    # 随机技能采样
    # =====================================
    def sample_skills(self, min_skills=4, max_skills=10):
        num = random.randint(min_skills, max_skills)
        return random.sample(self.all_skills, min(num, len(self.all_skills)))

    # =====================================
    # 生成简历文本
    # =====================================
    def generate_resume_text(self, skills):
        templates = [
            "熟练掌握 {skills}，具备良好的系统设计能力。",
            "在多个项目中使用 {skills} 进行开发。",
            "精通 {skills}，具有丰富实战经验。",
            "参与大型系统开发，技术栈包括 {skills}。",
            "熟悉 {skills}，能够独立完成复杂模块开发。"
        ]

        skills_str = "、".join(
            [self.registry.standard_index[s]["standard_name"] for s in skills]
        )

        return random.choice(templates).format(skills=skills_str)

    # =====================================
    # 批量生成
    # =====================================
    def generate_dataset(self, num_samples=1000):
        records = []

        for _ in tqdm(range(num_samples), desc="Generating resumes"):
            skills = self.sample_skills()

            resume_text = self.generate_resume_text(skills)

            # 标准技能名
            standard_skills = [
                self.registry.standard_index[s]["standard_name"]
                for s in skills
            ]

            categories = [
                self.registry.standard_index[s]["category"]
                for s in skills
            ]

            records.append({
                "resume_text": resume_text,
                "skills": "|".join(standard_skills),
                "categories": "|".join(categories)
            })

        return pd.DataFrame(records)

    # =====================================
    # 保存为 CSV
    # =====================================
    def save_to_csv(self, df, output_path):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 数据已保存到 {output_path}")

# src/skill_loader.py

import pandas as pd
from config.settings import SKILL_CSV_PATH


class SkillLoader:

    def __init__(self):
        self.df = pd.read_csv(SKILL_CSV_PATH)
        self.skill_map = self._build_skill_map()

    def _build_skill_map(self):
        skill_map = {}

        for _, row in self.df.iterrows():
            skill = row["skill_name"]
            synonyms = []

            if pd.notna(row["synonyms"]):
                synonyms = str(row["synonyms"]).split("|")

            skill_map[skill] = {
                "category": row["category"],
                "synonyms": synonyms
            }

        return skill_map

    def get_all_skills(self):
        return list(self.skill_map.keys())

    def get_random_skill(self):
        import random
        return random.choice(self.get_all_skills())

    def get_skill_with_variation(self, skill):
        """
        随机返回：
        - 标准名称
        - 或同义词
        """
        import random

        synonyms = self.skill_map[skill]["synonyms"]
        if synonyms and random.random() < 0.4:
            return random.choice(synonyms)

        return skill

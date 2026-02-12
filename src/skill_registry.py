# src/skill_registry.py

import pandas as pd
import re
from collections import defaultdict


class SkillRegistry:
    """
    企业级技能注册中心

    功能：
    - 技能标准化
    - 别名识别
    - 版本识别
    - 技能类别输出
    - 快速匹配
    """

    def __init__(self, csv_path="config/skills.csv"):
        self.csv_path = csv_path
        self.skill_df = pd.read_csv(csv_path)

        self.standard_index = {}
        self.alias_index = {}
        self.version_index = {}
        self.category_index = defaultdict(list)

        self._build_registry()

    # =========================
    # 构建注册中心
    # =========================
    def _build_registry(self):

        for _, row in self.skill_df.iterrows():
            skill = row["skill_name"]
            category = row["category"]

            aliases = []
            versions = []

            if pd.notna(row.get("aliases", "")):
                aliases = str(row["aliases"]).split("|")

            if pd.notna(row.get("versions", "")):
                versions = str(row["versions"]).split("|")

            # 标准索引
            self.standard_index[skill.lower()] = {
                "standard_name": skill,
                "category": category,
                "aliases": aliases,
                "versions": versions
            }

            # 分类索引
            self.category_index[category].append(skill)

            # 别名索引
            for alias in aliases:
                self.alias_index[alias.lower()] = skill

            # 版本索引
            for version in versions:
                self.version_index[version.lower()] = skill

    # =========================
    # 技能标准化
    # =========================
    def normalize(self, skill_text):
        """
        输入任意技能文本，返回标准技能名
        """
        skill_text = skill_text.strip().lower()

        # 1️⃣ 直接匹配标准名
        if skill_text in self.standard_index:
            return self.standard_index[skill_text]["standard_name"]

        # 2️⃣ 匹配别名
        if skill_text in self.alias_index:
            return self.alias_index[skill_text]

        # 3️⃣ 匹配版本号
        if skill_text in self.version_index:
            return self.version_index[skill_text]

        return None

    # =========================
    # 文本批量匹配
    # =========================
    def extract_from_text(self, text):
        """
        从文本中提取所有技能
        """
        found_skills = set()
        text_lower = text.lower()

        # 先匹配版本（防止被标准名覆盖）
        for version, standard in self.version_index.items():
            if version in text_lower:
                found_skills.add(standard)

        # 匹配别名
        for alias, standard in self.alias_index.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                found_skills.add(standard)

        # 匹配标准名
        for standard_lower, data in self.standard_index.items():
            if re.search(r'\b' + re.escape(standard_lower) + r'\b', text_lower):
                found_skills.add(data["standard_name"])

        return list(found_skills)

    # =========================
    # 输出带类别
    # =========================
    def extract_with_category(self, text):
        skills = self.extract_from_text(text)
        result = []
        for skill in skills:
            category = self.standard_index[skill.lower()]["category"]
            result.append({
                "skill": skill,
                "category": category
            })
        return result

    # =========================
    # 获取分类统计
    # =========================
    def category_statistics(self, skills):
        stats = defaultdict(int)
        for skill in skills:
            normalized = self.normalize(skill)
            if normalized:
                category = self.standard_index[normalized.lower()]["category"]
                stats[category] += 1
        return dict(stats)

    # =========================
    # 企业级接口：单技能获取分类
    # =========================
    def get_category(self, skill_text):
        if not skill_text:
            return None
        normalized = self.normalize(skill_text)
        if normalized:
            return self.standard_index[normalized.lower()]["category"]
        return None


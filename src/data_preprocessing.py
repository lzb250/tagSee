# src/data_preprocessing.py
# -*- coding: utf-8 -*-
import random
from .tokenizer import normalize_skill, expand_skill_versions
from config.computer_skills import ALL_COMPUTER_SKILLS

class DataPreprocessor:
    def __init__(self):
        self.all_skills = ALL_COMPUTER_SKILLS

    def generate_synthetic_data(self, num_samples=5000, skill_count_range=(3,8), save_samples=False):
        """
        生成训练数据（文本 + 标签），保证技能+版本号绑定
        """
        texts = []
        labels = []

        for _ in range(num_samples):
            num_skills = random.randint(skill_count_range[0], skill_count_range[1])
            chosen_skills = random.sample(self.all_skills, num_skills)

            tokens = []
            token_labels = []

            for skill in chosen_skills:
                skill_tokens = skill.split(' ')  # 拆空格
                versions = expand_skill_versions(skill)
                # 随机选择版本号
                version = random.choice(versions)
                version_tokens = version.split(' ')

                # 合并技能 + 版本号
                combined_tokens = skill_tokens + version_tokens[len(skill_tokens):]

                for i, t in enumerate(combined_tokens):
                    label = 'B-SKILL' if i == 0 else 'I-SKILL'
                    tokens.append(t)
                    token_labels.append(label)

                tokens.append(',')  # 技能间加逗号
                token_labels.append('O')

            if tokens:
                tokens.pop()  # 去掉最后一个逗号
                token_labels.pop()

            texts.append(tokens)
            labels.append(token_labels)

        return texts, labels

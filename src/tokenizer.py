# src/tokenizer.py
# -*- coding: utf-8 -*-

from config.computer_skills import ALL_COMPUTER_SKILLS, SKILL_NORMALIZATION, SKILL_VERSIONS

def normalize_skill(skill_name: str):
    """技能标准化"""
    key = skill_name.strip().lower()
    return SKILL_NORMALIZATION.get(key, skill_name)

def expand_skill_versions(skill_name: str):
    """返回技能绑定的所有版本"""
    return SKILL_VERSIONS.get(skill_name, [skill_name])

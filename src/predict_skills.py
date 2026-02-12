# src/predict_skills.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import pandas as pd
from src.skill_registry import SkillRegistry
import re

MODEL_PATH = "models/skill_extraction_model"
SKILLS_CSV = "config/skills.csv"

LABELS_LIST = ["O", "B-SKILL", "I-SKILL"]

# ===========================
# NER 预测类
# ===========================
class SkillExtractor:
    def __init__(self, model_path=MODEL_PATH, skills_csv=SKILLS_CSV, device=None):
        self.registry = SkillRegistry(skills_csv)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
            return_offsets_mapping=True
        )
        offsets = tokens.pop("offset_mapping").squeeze(0).tolist()
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()

        # 对齐标签到字符（按offset映射回原文本）
        char_labels = ["O"] * len(text)
        for pred, (start, end) in zip(predictions, offsets):
            if start == end == 0:  # [CLS]/[SEP]/padding
                continue
            label = LABELS_LIST[pred]
            if start < len(char_labels):
                char_labels[start] = label
            for i in range(start + 1, min(end, len(char_labels))):
                if label == "B-SKILL":
                    char_labels[i] = "I-SKILL"
                elif label == "I-SKILL":
                    char_labels[i] = "I-SKILL"

        skills_found = set()
        current_skill = ""
        for char, label in zip(text, char_labels):
            if label == "B-SKILL":
                if current_skill:
                    skills_found.add(current_skill)
                current_skill = char
            elif label == "I-SKILL" and current_skill:
                current_skill += char
            else:
                if current_skill:
                    skills_found.add(current_skill)
                    current_skill = ""
        if current_skill:
            skills_found.add(current_skill)

        # 技能标准化
        skills_normalized = [self.registry.normalize(skill) for skill in skills_found if self.registry.normalize(skill)]

        # 分类统计
        skill_categories = {}
        for skill in skills_normalized:
            cat = self.registry.get_category(skill)
            skill_categories[skill] = cat

        return skills_normalized, skill_categories

# ===========================
# 命令行接口
# ===========================
if __name__ == "__main__":
    extractor = SkillExtractor()
    print("请输入简历文本（Ctrl+D 结束输入）：")
    text = ""
    try:
        while True:
            line = input()
            text += line + "\n"
    except EOFError:
        pass

    skills, categories = extractor.predict(text)
    print("\n🔹 提取技能：")
    for skill, cat in categories.items():
        print(f"{skill} -> {cat}")

    print("\n📊 分类统计：")
    stats = {}
    for cat in categories.values():
        stats[cat] = stats.get(cat, 0) + 1
    print(stats)

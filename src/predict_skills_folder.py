# src/predict_skills_folder.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import pandas as pd
from src.skill_registry import SkillRegistry



MODEL_PATH = "models/skill_extraction_model"
SKILLS_CSV = "config/skills.csv"
INPUT_FOLDER = "data/resumes_txt"  # 多份简历txt文件夹
OUTPUT_CSV = "data/resume_skills_output.csv"

LABELS_LIST = ["O", "B-SKILL", "I-SKILL"]

class SkillExtractor:
    def __init__(self, model_path=MODEL_PATH, skills_csv=SKILLS_CSV, device=None):
        self.registry = SkillRegistry(skills_csv)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_text(self, text: str):
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=256
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

        labels = [LABELS_LIST[i] for i in predictions][:len(text)]
        skills_found = set()
        current_skill = ""
        for char, label in zip(text, labels):
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

        # 技能标准化 + 分类
        skills_normalized = [self.registry.normalize(s) for s in skills_found if self.registry.normalize(s)]
        skill_categories = {s: self.registry.get_category(s) for s in skills_normalized}

        return skills_normalized, skill_categories

if __name__ == "__main__":
    extractor = SkillExtractor()
    results = []

    txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".txt")]
    for idx, filename in enumerate(txt_files, 1):
        filepath = os.path.join(INPUT_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                continue

        skills, categories = extractor.predict_text(text)
        results.append({
            "filename": filename,
            "resume_text": text,
            "skills": ", ".join(skills),
            "skill_categories": ", ".join([f"{s}:{c}" for s, c in categories.items()])
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 批量技能提取完成，结果已保存到 {OUTPUT_CSV}")

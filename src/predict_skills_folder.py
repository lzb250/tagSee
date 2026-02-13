# src/predict_skills_folder.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import pandas as pd
from skill_registry import SkillRegistry



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

        max_length = 256
        stride = 128   # 重叠窗口，防止截断技能

        skills_found = set()

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False
        )

        input_ids = encoding["input_ids"][0]
        offsets = encoding["offset_mapping"][0]

        total_length = input_ids.size(0)

        for start in range(0, total_length, max_length - stride):
            end = min(start + max_length, total_length)

            chunk_input_ids = input_ids[start:end].unsqueeze(0).to(self.device)
            chunk_offsets = offsets[start:end]

            with torch.no_grad():
                outputs = self.model(input_ids=chunk_input_ids)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

            for pred, (s, e) in zip(predictions, chunk_offsets):
                if s == e:
                    continue
                label = LABELS_LIST[pred]
                if label in ("B-SKILL", "I-SKILL"):
                    skill_text = text[s:e]
                    skills_found.add(skill_text)

        # 标准化
        skills_normalized = sorted({
            self.registry.normalize(skill)
            for skill in skills_found
            if self.registry.normalize(skill)
        })

        # 分类
        skill_categories = {
            skill: self.registry.get_category(skill)
            for skill in skills_normalized
        }

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

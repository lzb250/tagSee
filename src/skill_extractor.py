# src/skill_extractor.py

import torch
from transformers import BertTokenizer, BertForTokenClassification
from config import *


class SkillExtractor:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(SAVE_PATH)
        self.model = BertForTokenClassification.from_pretrained(SAVE_PATH)
        self.model.eval()

    def extract(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        skills = []

        current = ""
        for token, label_id in zip(tokens, predictions):
            label = LABEL_LIST[label_id]

            if label == "B-SKILL":
                if current:
                    skills.append(current)
                current = token.replace("##", "")
            elif label == "I-SKILL":
                current += token.replace("##", "")
            else:
                if current:
                    skills.append(current)
                    current = ""

        if current:
            skills.append(current)

        return list(set(skills))

# -*- coding: utf-8 -*-
import os
import sys
import re
from pathlib import Path

# 1. 修复环境冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import jieba
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 确保项目根目录在路径中
project_root = r"C:\Users\LJY\PycharmProjects\TagSee"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.computer_skills import SKILL_NORMALIZATION, ALL_COMPUTER_SKILLS, COMPUTER_SKILLS
from src.cross_platform_utils import get_torch_device

class SkillExtractor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = get_torch_device()

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
        self.normalization_map = SKILL_NORMALIZATION

        # 预载技能库用于保底搜索 (Java Set 风格)
        self.skills_lookup = set(skill.lower() for skill in ALL_COMPUTER_SKILLS)

        # 注入分词词典
        for skill in ALL_COMPUTER_SKILLS:
            jieba.add_word(skill, freq=5000)

    def extract(self, text: str) -> list:
        if not text or not text.strip():
            return []

        # A. 模型推理流
        tokens = list(jieba.cut(text))
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        # B. 解码模型结果
        word_ids = inputs.word_ids()
        model_results = self._decode_and_clean(tokens, word_ids, preds)

        # C. 【保底逻辑】如果模型没扫出来，直接用词典在分词结果里过一遍
        # 这种双路合并（Merge）机制保证了高召回率
        if not model_results:
            fallback_results = [
                t for t in tokens
                if t.lower() in self.skills_lookup or t in ALL_COMPUTER_SKILLS
            ]
            model_results = fallback_results

        # D. 最终链式清洗
        return self._final_pipeline(model_results)

    def _decode_and_clean(self, tokens, word_ids, preds):
        chunks, current = [], []
        for idx, word_id in enumerate(word_ids):
            if word_id is None: continue
            label = self.id2label.get(preds[idx], 'O')
            if label == 'B-SKILL':
                if current: chunks.append("".join(current))
                current = [tokens[word_id]]
            elif label == 'I-SKILL':
                current.append(tokens[word_id])
            else:
                if current: chunks.append("".join(current))
                current = []
        if current: chunks.append("".join(current))
        return chunks

    def _final_pipeline(self, chunks):
        """链式过滤与归一化"""
        return sorted(list(set(
            self.normalization_map.get(c.lower(), c)
            for c in chunks
            if len(c) >= 2  # 过滤单字
            and not re.match(r'^[\u4e00-\u9fff]$', c) # 过滤单个中文字
            and not re.match(r'^[0-9.\W]+$', c) # 过滤纯符号/数字
        )))

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\LJY\PycharmProjects\TagSee\models\skill_extraction_model_final"

    try:
        extractor = SkillExtractor(MODEL_PATH)
        test_text = "本人熟练使用Java编程，掌握Spring Cloud微服务架构。对Vue.js有一定了解，负责过MySQL数据库调优。"

        print("\n" + "🚀 识别结果 " + "="*30)
        print(f"输入: {test_text}")
        print(f"技能标签: {extractor.extract(test_text)}")
        print("="*41)

    except Exception as e:
        print(f"错误: {e}")
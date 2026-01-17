# src/inference.py
import re
from typing import List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from config.computer_skills import SKILL_NORMALIZATION
from .cross_platform_utils import safe_path, get_torch_device
import jieba

class SkillExtractor:
    """技能提取推理器（支持本地训练模型）"""

    def __init__(self, model_path: str):
        self.model_path = safe_path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self.device = get_torch_device()
        print(f"使用设备: {self.device}")

        # 加载本地 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        # 标签映射
        self.id2label = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
        self.normalization_map = SKILL_NORMALIZATION

    def extract_skills(self, resume_text: str) -> List[str]:
        """从简历文本中提取技能"""

        if not resume_text.strip():
            return []

        # 使用 jieba 分词
        tokens = list(jieba.cut(resume_text))
        if not tokens:
            return []

        # 编码
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # ⚠ 解决旧版本 transformers 无 word_ids() 报错
        if hasattr(inputs, "word_ids"):
            word_ids = inputs.word_ids()
        else:
            # 兼容旧版本
            batch_word_ids = []
            for i in range(inputs["input_ids"].size(0)):
                if hasattr(self.tokenizer, "batch_encode_plus"):
                    enc = self.tokenizer.batch_encode_plus([tokens])
                    batch_word_ids.append(enc.word_ids(batch_index=0))
            word_ids = batch_word_ids[0] if batch_word_ids else [None]*len(tokens)

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        # 提取技能
        skills = []
        current_skill_tokens = []

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            label_id = predictions[idx]
            label = self.id2label.get(label_id, 'O')
            token = tokens[word_id] if word_id < len(tokens) else ""

            if label == 'B-SKILL':
                if current_skill_tokens:
                    skill_str = ''.join(current_skill_tokens)
                    if self._is_valid_skill(skill_str):
                        skills.append(skill_str)
                    current_skill_tokens = []
                current_skill_tokens.append(token)
            elif label == 'I-SKILL':
                current_skill_tokens.append(token)
            else:
                if current_skill_tokens:
                    skill_str = ''.join(current_skill_tokens)
                    if self._is_valid_skill(skill_str):
                        skills.append(skill_str)
                    current_skill_tokens = []

        # 处理最后一个技能
        if current_skill_tokens:
            skill_str = ''.join(current_skill_tokens)
            if self._is_valid_skill(skill_str):
                skills.append(skill_str)

        # 标准化技能名称
        normalized_skills = [self.normalization_map.get(s.lower(), s) for s in skills]

        # 去重
        final_skills = list(set(filter(self._is_valid_skill, normalized_skills)))

        return final_skills

    def _is_valid_skill(self, skill: str) -> bool:
        """验证技能是否有效"""
        if not skill or len(skill.strip()) < 2:
            return False
        skill = skill.strip()
        if not re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', skill):
            return False
        if re.match(r'^[\d\s\W]+$', skill):
            return False
        invalid_patterns = [r'^[a-z]$', r'^[A-Z]$', r'^\d+$', r'^[+\-*/=<>|&^%$#@!~`]+$']
        for pattern in invalid_patterns:
            if re.match(pattern, skill):
                return False
        return True


# 使用示例
if __name__ == "__main__":
    model_path = "./models/skill_extraction_model_final"
    extractor = SkillExtractor(model_path)

    test_text = "我精通Python和React，熟悉Docker容器化技术"
    skills = extractor.extract_skills(test_text)
    print("提取技能:", skills)

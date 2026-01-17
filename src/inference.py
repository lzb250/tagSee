# src/inference.py
import re
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from config.computer_skills import SKILL_NORMALIZATION
from .cross_platform_utils import safe_path, get_torch_device


class SkillExtractor:
    """技能提取推理器"""

    def __init__(self, model_path: str):
        self.model_path = safe_path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self.device = get_torch_device()
        print(f"使用设备: {self.device}")

        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        # 标签映射
        self.id2label = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}
        self.normalization_map = SKILL_NORMALIZATION


    def extract_skills(self, resume_text: str) -> List[str]:
        """从简历文本中提取技能（使用 jieba 分词）"""
        if not resume_text.strip():
            return []

        import jieba
        # 使用 jieba 进行中文分词
        tokens = list(jieba.cut(resume_text))
        if not tokens:
            return []

        # 编码（注意：tokenizer 需要处理分词后的 tokens）
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # 解码预测结果
        predicted_labels = []
        for pred in predictions[0]:
            label_id = pred.item()
            if label_id in self.id2label:
                predicted_labels.append(self.id2label[label_id])
            else:
                predicted_labels.append('O')

        # 提取技能实体（基于分词结果）
        skills = []
        current_skill_tokens = []

        # 只处理实际的 tokens（忽略填充和特殊token）
        word_ids = inputs.word_ids()
        token_index = 0

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if token_index >= len(predicted_labels):
                break

            label = predicted_labels[token_index]
            token_index += 1

            if word_id < len(tokens):
                token = tokens[word_id]
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
        normalized_skills = []
        for skill in skills:
            normalized = self.normalization_map.get(skill.lower(), skill)
            normalized_skills.append(normalized)

        # 去重并过滤
        final_skills = []
        for skill in normalized_skills:
            if self._is_valid_skill(skill):
                final_skills.append(skill)

        return list(set(final_skills))

    def _is_valid_skill(self, skill: str) -> bool:
        """验证技能是否有效"""
        if not skill or len(skill.strip()) < 2:
            return False

        skill = skill.strip()

        # 必须包含字母、数字或中文
        if not re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', skill):
            return False

        # 不能全是数字或符号
        if re.match(r'^[\d\s\W]+$', skill):
            return False

        # 过滤明显无效的字符组合
        invalid_patterns = [r'^[a-z]$', r'^[A-Z]$', r'^\d+$', r'^[+\-*/=<>|&^%$#@!~`]+$']
        for pattern in invalid_patterns:
            if re.match(pattern, skill):
                return False

        return True

# 使用示例
if __name__ == "__main__":
    # 测试代码...
    pass
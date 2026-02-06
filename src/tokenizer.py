# src/tokenizer.py

"""
文件路径: src/tokenizer.py
功能解析: 中文分词器模块

主要功能:
1. CrossPlatformTokenizer 类
   - 基于 jieba 分词器，支持中文文本分词
   - 自动注入计算机专业词汇到词典
   - 支持自定义词典加载
   - 带词性的分词标注

2. 核心方法
   - tokenize(): 基础中文分词
   - tokenize_with_pos(): 带词性标注的分词
   - find_skills_in_text(): 直接在文本中查找技能（不依赖分词）

3. 特性
   - 预注入计算机技能词汇（如 Python、Java、Django 等）
   - 预注入常见技术术语（如 机器学习、深度学习等）
   - 支持完整单词匹配验证

使用场景:
- 文本预处理
- 技能关键词匹配
- 中文 NLP 任务
---
"""

import jieba
import jieba.posseg as pseg
from typing import List, Tuple
from pathlib import Path
from .cross_platform_utils import safe_path
from config.computer_skills import ALL_COMPUTER_SKILLS

class CrossPlatformTokenizer:
    """跨平台中文分词器"""

    def __init__(self, custom_dict_path: str = None):
        # 添加计算机专业词汇到 jieba 词典
        for skill in ALL_COMPUTER_SKILLS:
            jieba.add_word(skill, freq=1000, tag='nz')

        # 添加常见技术术语
        tech_terms = [
            '机器学习', '深度学习', '自然语言处理', '计算机视觉',
            '微服务', '容器化', '自动化部署', '持续集成',
            '敏捷开发', '测试驱动', '代码审查', '性能优化'
        ]
        for term in tech_terms:
            jieba.add_word(term, freq=1000, tag='nz')

        # 加载自定义词典（如果提供）
        if custom_dict_path:
            dict_path = safe_path(custom_dict_path)
            if dict_path.exists():
                jieba.load_userdict(str(dict_path))

    def tokenize(self, text: str, cut_all: bool = False) -> List[str]:
        """中文分词"""
        if not text.strip():
            return []

        tokens = jieba.lcut(text, cut_all=cut_all)
        # 过滤空字符串
        return [token for token in tokens if token.strip()]

    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """带词性的分词"""
        if not text.strip():
            return []

        words = pseg.cut(text)
        return [(word, flag) for word, flag in words if word.strip()]

    def find_skills_in_text(self, text: str) -> List[str]:
        """直接在文本中查找技能（不依赖分词）"""
        found_skills = []
        text_lower = text.lower()

        for skill in ALL_COMPUTER_SKILLS:
            skill_lower = skill.lower()
            if skill_lower in text_lower:
                # 验证是否为完整匹配
                if self._is_complete_match(text_lower, skill_lower):
                    found_skills.append(skill)

        return list(set(found_skills))

    def _is_complete_match(self, text: str, skill: str) -> bool:
        """检查是否为完整单词匹配"""
        import re
        pattern = r'\b' + re.escape(skill) + r'\b'
        return bool(re.search(pattern, text))

# 使用示例
if __name__ == "__main__":
    tokenizer = CrossPlatformTokenizer()
    test_text = "我精通Python和React，熟悉Docker容器化技术"
    tokens = tokenizer.tokenize(test_text)
    print("分词结果:", tokens)
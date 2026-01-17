# src/data_preprocessing.py
import json
import random
from typing import List, Tuple

from config.computer_skills import ALL_COMPUTER_SKILLS, SKILL_NORMALIZATION
from .cross_platform_utils import safe_path


class DataPreprocessor:
    """数据预处理和标注生成器"""

    def __init__(self):
        self.skills_set = set(ALL_COMPUTER_SKILLS)
        self.normalization_map = SKILL_NORMALIZATION
        self.label2id = {'O': 0, 'B-SKILL': 1, 'I-SKILL': 2}
        self.id2label = {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'}

    def create_bio_annotations(self, text: str) -> List[Tuple[str, str]]:
        """
        从原始文本创建BIO标注（改进版）
        """
        if not text.strip():
            return []

        import jieba
        tokens = list(jieba.cut(text))
        labels = ['O'] * len(tokens)

        # 创建文本到token的映射
        token_positions = []
        current_pos = 0
        for token in tokens:
            start_pos = current_pos
            end_pos = current_pos + len(token)
            token_positions.append((start_pos, end_pos))
            current_pos = end_pos

        # 在原始文本中查找每个技能
        text_lower = text.lower()
        for skill in sorted(self.skills_set, key=len, reverse=True):  # 长技能优先
            skill_lower = skill.lower()
            start = 0
            while start < len(text_lower):
                pos = text_lower.find(skill_lower, start)
                if pos == -1:
                    break

                # 验证是否为完整单词边界
                if self._is_word_boundary(text, pos, len(skill)):
                    # 找到对应的token范围
                    skill_end = pos + len(skill)
                    skill_tokens = []
                    for i, (tok_start, tok_end) in enumerate(token_positions):
                        if tok_start >= pos and tok_end <= skill_end:
                            skill_tokens.append(i)
                        elif tok_start < skill_end and tok_end > pos:
                            # Token与技能有重叠，也包含进来
                            skill_tokens.append(i)

                    # 标注BIO
                    if skill_tokens:
                        labels[skill_tokens[0]] = 'B-SKILL'
                        for idx in skill_tokens[1:]:
                            labels[idx] = 'I-SKILL'

                start = pos + 1

        # 过滤空 token
        filtered_pairs = [(token, label) for token, label in zip(tokens, labels) if token.strip()]
        return filtered_pairs

    def _is_valid_skill_in_token(self, token: str, skill: str) -> bool:
        """检查技能是否在 token 中有效"""
        if skill == token:
            return True
        # 允许版本号包含，如 "Python 3.9" 包含 "Python"
        if skill in token and (token.startswith(skill) or f" {skill}" in token):
            return True
        return False

        def _is_valid_skill_position(self, text: str, pos: int, length: int) -> bool:
            """检查技能位置是否有效"""
            # 检查前一个字符
            if pos > 0:
                prev_char = text[pos - 1]
                if prev_char.isalnum() or prev_char in '_':
                    return False

            # 检查后一个字符
            if pos + length < len(text):
                next_char = text[pos + length]
                if next_char.isalnum() or next_char in '_':
                    return False

            return True

        def _is_word_boundary(self, text: str, pos: int, length: int) -> bool:
            """检查是否为单词边界"""
            # 检查前一个字符
            if pos > 0:
                prev_char = text[pos - 1]
                if prev_char.isalnum() or prev_char in '_':
                    return False

            # 检查后一个字符
            if pos + length < len(text):
                next_char = text[pos + length]
                if next_char.isalnum() or next_char in '_':
                    return False

            return True

        def save_dataset(self, texts: List[List[str]], labels: List[List[str]],
                         output_path: str, format: str = 'json'):
            """保存数据集"""
            output_path = safe_path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == 'json':
                dataset = []
                for text, label in zip(texts, labels):
                    dataset.append({
                        'tokens': text,
                        'labels': label
                    })

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

            elif format == 'conll':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for text, label in zip(texts, labels):
                        for token, lbl in zip(text, label):
                            f.write(f"{token} {lbl}\n")
                        f.write("\n")  # 句子间空行

        def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[List[List[str]], List[List[str]]]:
            """生成合成训练数据"""
            templates = [
                "我有{years}年{field}开发经验，熟练掌握{skills}，熟悉{frameworks}框架。",
                "毕业于{university}计算机专业，精通{skills}，在{projects}项目中有丰富经验。",
                "擅长{skills}开发，使用{frameworks}和{databases}构建高性能应用。",
                "具备{years}年{domain}领域经验，主要技术栈包括{skills}和{cloud_services}。"
            ]

            fields = ['Web', '移动', '后端', '前端', '全栈', '数据', 'AI', '区块链']
            universities = ['清华大学', '北京大学', '上海交通大学', '浙江大学', '复旦大学']
            projects = ['电商', '社交', '金融', '医疗', '教育', '游戏', '物联网']
            domains = ['金融科技', '电子商务', '社交网络', '医疗健康', '教育科技', '智能硬件']

            texts = []
            labels = []

            for _ in range(num_samples):
                # 随机选择模板和参数
                template = random.choice(templates)

                # 随机选择技能
                selected_skills = []
                all_skills_list = list(self.skills_set)
                num_skills = random.randint(2, 5)
                selected_skills = random.sample(all_skills_list, min(num_skills, len(all_skills_list)))

                # 构建简历文本
                resume_text = template.format(
                    years=random.randint(1, 10),
                    field=random.choice(fields),
                    university=random.choice(universities),
                    projects=random.choice(projects),
                    domain=random.choice(domains),
                    skills='、'.join(selected_skills[:2]),
                    frameworks='、'.join(random.sample(
                        [s for s in all_skills_list if
                         '框架' in s or any(f in s for f in ['React', 'Vue', 'Django', 'Spring'])],
                        min(2, len(all_skills_list))
                    )),
                    databases='、'.join(random.sample(
                        [s for s in all_skills_list if
                         any(db in s for db in ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis'])],
                        min(2, len(all_skills_list))
                    )),
                    cloud_services='、'.join(random.sample(
                        [s for s in all_skills_list if
                         any(cloud in s for cloud in ['AWS', 'Azure', 'Google', '阿里云'])],
                        min(1, len(all_skills_list))
                    ))
                )

                # 创建标注
                annotations = self.create_bio_annotations(resume_text)
                if annotations:  # 确保有标注数据
                    tokens, token_labels = zip(*annotations)
                    texts.append(list(tokens))
                    labels.append(list(token_labels))

            return texts, labels

    # 使用示例
    if __name__ == "__main__":
        preprocessor = DataPreprocessor()
        # 生成合成数据...
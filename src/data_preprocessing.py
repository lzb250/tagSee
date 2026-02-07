"""
文件路径: src/data_preprocessing.py
功能解析: 数据预处理和标注生成模块

主要功能:
1. DataPreprocessor 类
   - 生成合成训练数据（模拟真实简历文本）
   - 创建 BIO 格式的序列标注
   - 保存数据集为 JSON 或 CoNLL 格式

2. 核心方法
   - create_bio_annotations(): 从原始文本创建 BIO 标注
   - generate_synthetic_data(): 生成合成训练数据
   - save_dataset(): 保存数据集

使用场景:
- 训练数据生成和增强
- 序列标注数据准备
- 跨平台数据处理
---
"""

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
        从原始文本创建BIO标注（增强版）

        改进点：
        1. 支持带版本号的技能标注（如 Java 11, Vue 3）
        2. 支持缩写技能标注（如 S3, EC2）
        3. 智能识别技能边界

        使用策略：直接在原始文本中匹配技能，然后反向映射到 token
        """
        if not text.strip():
            return []

        import jieba

        # 先注入完整的技能到分词器
        self._ensure_skills_in_tokenizer()

        # 使用增强的分词器进行分词
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

        # 增强的技能词典（包含带版本号的变体）
        enhanced_skills = []

        # 1. 基础技能
        for skill in self.skills_set:
            enhanced_skills.append((skill, skill))  # (搜索文本, 原始技能名)

        # 2. 带版本号的技能
        skill_versions = {
            'Java': ['Java 8', 'Java 11', 'Java 17', 'Java 21', 'Java SE'],
            'Python': ['Python 2.7', 'Python 3.6', 'Python 3.7',  'Python 3.8',
                       'Python 3.9', 'Python 3.10', 'Python 3.11', 'Python 3.12'],
            'Go': ['Go 1.16', 'Go 1.17', 'Go 1.18', 'Go 1.19', 'Go 1.20', 'Go 1.21'],
            'Node.js': ['Node.js 10', 'Node.js 12', 'Node.js 14', 'Node.js 16', 'Node.js 18', 'Node.js 20'],
            'Vue': ['Vue 2', 'Vue 3', 'Vue.js 2', 'Vue.js 3'],
            'React': ['React 15', 'React 16', 'React 17', 'React 18'],
            'Angular': ['AngularJS', 'Angular 2', 'Angular 4', 'Angular 6',
                        'Angular 8', 'Angular 10', 'Angular 12', 'Angular 14', 'Angular 16', 'Angular 17'],
            'Spring': ['Spring 4', 'Spring 5', 'Spring 6',
                       'Spring Boot 1.x', 'Spring Boot 2.0', 'Spring Boot 2.1',
                       'Spring Boot 2.2', 'Spring Boot 2.3', 'Spring Boot 2.4',
                       'Spring Boot 2.5', 'Spring Boot 2.6', 'Spring Boot 2.7',
                       'Spring Boot 3.0', 'Spring Boot 3.1', 'Spring Boot 3.2',
                       'Spring Cloud 2020', 'Spring Cloud 2021', 'Spring Cloud 2022', 'Spring Cloud 2023'],
            'MySQL': ['MySQL 5.5', 'MySQL 5.6', 'MySQL 5.7', 'MySQL 8.0'],
            'PostgreSQL': ['PostgreSQL 9.x', 'PostgreSQL 10', 'PostgreSQL 11',
                           'PostgreSQL 12', 'PostgreSQL 13', 'PostgreSQL 14', 'PostgreSQL 15', 'PostgreSQL 16'],
            'MongoDB': ['MongoDB 3.x', 'MongoDB 4.0', 'MongoDB 4.2', 'MongoDB 4.4',
                        'MongoDB 5.0', 'MongoDB 6.0', 'MongoDB 7.0'],
            'Redis': ['Redis 3.x', 'Redis 4.x', 'Redis 5.0', 'Redis 6.0', 'Redis 6.2', 'Redis 7.0', 'Redis 7.2'],
            'Docker': ['Docker 17.x', 'Docker 18.x', 'Docker 19.x', 'Docker 20.x', 'Docker 23.x', 'Docker 24.x'],
            'Kubernetes': ['Kubernetes 1.18', 'Kubernetes 1.19', 'Kubernetes 1.20', 'Kubernetes 1.21',
                           'Kubernetes 1.22', 'Kubernetes 1.23', 'Kubernetes 1.24', 'Kubernetes 1.25',
                           'Kubernetes 1.26', 'Kubernetes 1.27', 'Kubernetes 1.28', 'Kubernetes 1.29'],
            'CentOS': ['CentOS 6', 'CentOS 7', 'CentOS 8', 'CentOS Stream'],
            'Ubuntu': ['Ubuntu 14.04', 'Ubuntu 16.04', 'Ubuntu 18.04', 'Ubuntu 20.04', 'Ubuntu 22.04', 'Ubuntu 24.04'],
            'Debian': ['Debian 8', 'Debian 9', 'Debian 10', 'Debian 11', 'Debian 12'],
            'Windows': ['Windows 7', 'Windows 8', 'Windows 10', 'Windows 11', 'Windows Server'],
        }

        for base_skill, versions in skill_versions.items():
            for version in versions:
                if base_skill in self.skills_set:
                    enhanced_skills.append((version, version))

        # 3. 缩写技能
        abbreviation_skills = {
            'S3': 'Amazon S3',
            'EC2': 'Amazon EC2',
            'K8s': 'Kubernetes',
            'K8S': 'Kubernetes',
            'TS': 'TypeScript',
            'JS': 'JavaScript',
            'ES': 'Elasticsearch',
        }

        for abbr, full_name in abbreviation_skills.items():
            if full_name in self.skills_set or abbr in self.skills_set:
                enhanced_skills.append((abbr, abbr))

        # 按长度降序排列，优先匹配长技能
        enhanced_skills.sort(key=lambda x: len(x[0]), reverse=True)

        # 在原始文本中查找每个技能
        text_lower = text.lower()
        labeled_positions = set()  # 已标注的位置，避免重复标注

        for search_text, original_skill in enhanced_skills:
            search_text_lower = search_text.lower()
            start = 0

            while start < len(text_lower):
                pos = text_lower.find(search_text_lower, start)
                if pos == -1:
                    break

                # 检查是否已经被标注过
                is_overlapped = any(pos < end and pos + len(search_text) > start
                                    for start, end in labeled_positions)
                if is_overlapped:
                    start = pos + 1
                    continue

                # 验证是否为完整单词边界
                if self._is_word_boundary(text, pos, len(search_text)):
                    # 找到对应的token范围
                    skill_end = pos + len(search_text)
                    skill_tokens = []

                    for i, (tok_start, tok_end) in enumerate(token_positions):
                        # token 与技能有重叠或包含
                        if tok_start < skill_end and tok_end > pos:
                            skill_tokens.append(i)

                    # 标注BIO
                    if skill_tokens:
                        labels[skill_tokens[0]] = 'B-SKILL'
                        for idx in skill_tokens[1:]:
                            labels[idx] = 'I-SKILL'
                        labeled_positions.add((pos, skill_end))

                start = pos + 1

        # 过滤空 token
        filtered_pairs = [(token, label) for token, label in zip(tokens, labels) if token.strip()]
        return filtered_pairs

    def _ensure_skills_in_tokenizer(self):
        """确保技能词汇已注入到分词器"""
        import jieba

        # 注入带版本号的技能
        skill_versions = {
            'Java': ['Java 8', 'Java 11', 'Java 17', 'Java 21', 'Java SE'],
            'Python': ['Python 2.7', 'Python 3.6', 'Python 3.7', 'Python 3.8',
                       'Python 3.9', 'Python 3.10', 'Python 3.11', 'Python 3.12'],
            'Go': ['Go 1.16', 'Go 1.17', 'Go 1.18', 'Go 1.19', 'Go 1.20', 'Go 1.21'],
            'Vue': ['Vue 2', 'Vue 3', 'Vue.js 2', 'Vue.js 3'],
            'React': ['React 15', 'React 16', 'React 17', 'React 18'],
            'Spring': ['Spring 4', 'Spring 5', 'Spring 6',
                       'Spring Boot 1.x', 'Spring Boot 2.0', 'Spring Boot 2.1',
                       'Spring Boot 2.2', 'Spring Boot 2.3', 'Spring Boot 2.4',
                       'Spring Boot 2.5', 'Spring Boot 2.6', 'Spring Boot 2.7',
                       'Spring Boot 3.0', 'Spring Boot 3.1', 'Spring Boot 3.2',
                       'Spring Cloud 2020', 'Spring Cloud 2021', 'Spring Cloud 2022', 'Spring Cloud 2023'],
            'MySQL': ['MySQL 5.5', 'MySQL 5.6', 'MySQL 5.7', 'MySQL 8.0'],
            'PostgreSQL': ['PostgreSQL 9.x', 'PostgreSQL 10', 'PostgreSQL 11',
                           'PostgreSQL 12', 'PostgreSQL 13', 'PostgreSQL 14', 'PostgreSQL 15', 'PostgreSQL 16'],
            'MongoDB': ['MongoDB 3.x', 'MongoDB 4.0', 'MongoDB 4.2', 'MongoDB 4.4',
                        'MongoDB 5.0', 'MongoDB 6.0', 'MongoDB 7.0'],
            'Redis': ['Redis 3.x', 'Redis 4.x', 'Redis 5.0', 'Redis 6.0', 'Redis 6.2', 'Redis 7.0', 'Redis 7.2'],
            'Docker': ['Docker 17.x', 'Docker 18.x', 'Docker 19.x', 'Docker 20.x', 'Docker 23.x', 'Docker 24.x'],
            'Kubernetes': ['Kubernetes 1.18', 'Kubernetes 1.19', 'Kubernetes 1.20', 'Kubernetes 1.21',
                           'Kubernetes 1.22', 'Kubernetes 1.23', 'Kubernetes 1.24', 'Kubernetes 1.25',
                           'Kubernetes 1.26', 'Kubernetes 1.27', 'Kubernetes 1.28', 'Kubernetes 1.29'],
            'CentOS': ['CentOS 6', 'CentOS 7', 'CentOS 8', 'CentOS Stream'],
            'Ubuntu': ['Ubuntu 14.04', 'Ubuntu 16.04', 'Ubuntu 18.04', 'Ubuntu 20.04', 'Ubuntu 22.04', 'Ubuntu 24.04'],
        }

        for base_skill, versions in skill_versions.items():
            if base_skill in self.skills_set:
                for version in versions:
                    words = version.split()
                    for word in words:
                        jieba.add_word(word, freq=2000, tag='nz')
                    # 同时添加完整版本号组合
                    jieba.add_word(version, freq=3000, tag='nz')

        # 注入缩写
        abbreviations = ['S3', 'EC2', 'K8s', 'K8S', 'S 3', 'EC 2']
        for abbr in abbreviations:
            jieba.add_word(abbr, freq=3000, tag='nz')

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

    def generate_synthetic_data(self, num_samples: int = 1000, skill_count_range: Tuple[int, int] = (3, 8),
                                save_samples: bool = False, save_dir: str = "data/synthetic_resumes") -> Tuple[List[List[str]], List[List[str]]]:
        """
        生成合成训练数据（完全优化版）

        改进点：
        1. 包含带版本号的技能（如 Java 11, Vue 3）
        2. 包含缩写技能（如 S3, EC2）
        3. 包含混合上下文和负样本
        4. 模拟真实简历的复杂句式
        5. 增加技能数量的随机性和多样性
        6. 更多正负样本比例控制

        Args:
            num_samples: 生成样本数量
            skill_count_range: 每条样本技能数量范围
            save_samples: 是否保存生成的简历文本到文件
            save_dir: 保存目录
        """
        # 增强：带版本号的技能变体
        skill_with_versions = {
            'Java': ['Java 8', 'Java 11', 'Java 17'],
            'Python': ['Python 3.8', 'Python 3.9', 'Python 3.10'],
            'Vue': ['Vue 2', 'Vue 3', 'Vue.js 3'],
            'Spring': ['Spring 5', 'Spring Boot 2.7', 'Spring Cloud 2022'],
            'MySQL': ['MySQL 5.7', 'MySQL 8.0'],
            'MongoDB': ['MongoDB 5.0', 'MongoDB 6.0'],
            'Redis': ['Redis 6.0', 'Redis 7.0'],
            'CentOS': ['CentOS 7', 'CentOS 8'],
            'Ubuntu': ['Ubuntu 20.04', 'Ubuntu 22.04'],
            'Docker': ['Docker 20', 'Docker 23'],
            'Kubernetes': ['Kubernetes 1.24', 'Kubernetes 1.28'],
            'Node.js': ['Node.js 14', 'Node.js 16', 'Node.js 18'],
            'React': ['React 17', 'React 18'],
            'Angular': ['Angular 12', 'Angular 14', 'Angular 16'],
            'Go': ['Go 1.18', 'Go 1.19', 'Go 1.20'],
        }

        # 缩写技能映射
        abbreviation_skills = {
            'S3': 'Amazon S3',
            'EC2': 'Amazon EC2',
            'ES': 'Elasticsearch',
            'K8s': 'Kubernetes',
            'TS': 'TypeScript',
            'JS': 'JavaScript',
        }

        # 丰富的模板（模拟真实简历）
        templates = [
            # 基础描述（高权重）
            "熟练使用{skills}进行开发，掌握{frameworks}框架。",
            "精通{skills}，有{years}年相关开发经验。",
            "熟悉{databases}数据库设计和优化。",
            "使用{cloud_services}进行云原生架构设计。",

            # 版本号混合（重要！让模型学习版本号）
            "精通{skills_with_version}，熟悉{frameworks}，对比过{db_versions}。",
            "主要使用Java 11和Spring Boot 2.7，后端框架是{frameworks}。",
            "前端开发采用Vue 3和TypeScript，配合React 18构建应用。",
            "数据库使用MySQL 8.0和Redis 7.0，部署在Kubernetes 1.28上。",

            # 复杂句式
            "在工作中，我主要负责{skills}开发，配合{frameworks}框架，使用{databases}存储数据。",
            "技术栈包括{skills}，后端使用{frameworks}，部署在{cloud_services}上。",
            "从{year}年至今，参与过{projects}项目，使用{skills}和{tools}。",

            # 程序员项目经验格式
            "项目：{projects}开发，技术栈：{skills}，使用{databases}数据库。",
            "负责{projects}模块开发，采用{frameworks}架构，使用{tools}工具。",

            # 包含负样本（混淆词 - 重要！让模型学会区分）
            "担任{role}职位，负责{skills}开发。需要具备良好的{soft_skills}能力。",
            "在{company}工作期间，使用{skills}完成{projects}项目。",
            "善于沟通和学习，掌握{skills}，有丰富的{soft_skills}经验。",

            # 缩写技能（重要！）
            "使用AWS的S3和EC2服务，熟悉Docker容器化部署。",
            "熟练使用K8s进行容器编排，了解S3对象存储。",
            "掌握ES（Elasticsearch）全文搜索，配合K8s使用。",

            # 技能组合
            "前端使用{frontend_skills}，后端使用{backend_skills}，数据库选择{databases}。",
            "云服务使用{cloud_services}，容器化采用{kubernetes_version}。",

            # 真实简历风格
            "本人毕业于{year}年，主修{projects}方向，熟练掌握{skills}。",
            "在{company}担任{role}，使用{skills}和{frameworks}开发{projects}系统。",
        ]

        fields = ['Web', '移动', '后端', '前端', '全栈', '数据', 'AI', '区块链']
        companies = ['阿里集团', '腾讯', '字节跳动', '美团', '京东', '华为', '微软', 'Google']
        roles = ['高级工程师', '技术负责人', '架构师', '开发工程师', '全栈工程师']
        soft_skills = ['沟通', '团队协作', '学习', '问题解决', '创新']  # 这些不是技能，是负样本
        year_options = [2018, 2019, 2020, 2021, 2022]

        texts = []
        labels = []

        for _ in range(num_samples):
            # 随机选择模板
            template = random.choice(templates)
            all_skills_list = list(self.skills_set)

            # 构建参数字典
            params = {}

            # 随机选择基础技能（增加数量范围）
            num_skills = random.randint(skill_count_range[0], skill_count_range[1])
            selected_skills = random.sample(all_skills_list, min(num_skills, len(all_skills_list)))
            params['skills'] = '、'.join(selected_skills)

            # 带版本号的技能
            with_version = []
            for skill_base, versions in skill_with_versions.items():
                if skill_base in selected_skills:
                    with_version.append(random.choice(versions))
                elif random.random() < 0.1:  # 10%概率额外添加带版本号的技能
                    with_version.append(random.choice(versions))
            params['skills_with_version'] = '、'.join(with_version) if with_version else params['skills']

            # 前端/后端分类
            frontend = ['React', 'Vue', 'Angular', 'TypeScript', 'JavaScript', 'Next.js']
            backend = ['Java', 'Python', 'Go', 'Spring', 'Django', 'FastAPI', 'Node.js']

            params['frontend_skills'] = random.sample(
                [s for s in all_skills_list if s in frontend],
                k=min(2, len([s for s in all_skills_list if s in frontend]))
            )
            params['backend_skills'] = random.sample(
                [s for s in all_skills_list if s in backend],
                k=min(2, len([s for s in all_skills_list if s in backend]))
            )
            params['frontend_skills'] = '、'.join(params['frontend_skills'])
            params['backend_skills'] = '、'.join(params['backend_skills'])

            # 框架
            params['frameworks'] = '、'.join(random.sample(
                [s for s in all_skills_list if '框架' in s or any(f in s for f in ['React', 'Vue', 'Django', 'Spring', 'Flask'])],
                min(2, len(all_skills_list))
            ))

            # 数据库（包含版本）
            db_with_version = []
            for db in ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis']:
                if db in selected_skills and db in skill_with_versions:
                    db_with_version.append(random.choice(skill_with_versions[db]))
            params['databases'] = '、'.join(db_with_version) if db_with_version else '、'.join(
                random.sample([s for s in all_skills_list if any(d in s for d in ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis'])],
                              min(2, len(all_skills_list)))
            )
            params['db_versions'] = params['databases']

            # 云服务
            params['cloud_services'] = '、'.join(random.sample(
                ['AWS', 'Azure', '阿里云', '腾讯云', 'Kubernetes', 'Docker'],
                min(2, 6)
            ))

            # Kubernetes版本
            if 'Kubernetes' in selected_skills:
                params['kubernetes_version'] = random.choice(['Kubernetes 1.24', 'Kubernetes 1.28', 'K8s 1.28'])
            else:
                params['kubernetes_version'] = 'Kubernetes'

            # 工具
            params['tools'] = '、'.join(random.sample(
                ['Git', 'Jenkins', 'Docker', 'VS Code', 'Postman', 'Swagger'],
                min(2, 6)
            ))

            # 其他参数
            params['years'] = random.randint(1, 10)
            params['year'] = random.choice(year_options)
            params['projects'] = random.choice(['电商', '社交', '金融', '医疗', '教育', '游戏', '物联网'])
            params['company'] = random.choice(companies)
            params['role'] = random.choice(roles)
            params['soft_skills'] = random.choice(soft_skills)

            # 生成文本
            try:
                resume_text = template.format(**params)
            except KeyError as e:
                # 如果模板使用了未定义的参数，跳过
                continue

            # 创建标注
            annotations = self.create_bio_annotations(resume_text)
            if annotations:
                tokens, token_labels = zip(*annotations)
                texts.append(list(tokens))
                labels.append(list(token_labels))

                # 如果需要保存简历文本，在这里保存原始文本
                if save_samples:
                    from pathlib import Path
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)

                    # 保存原始文本（不是token列表）
                    sample_file = save_path / f"resume_{len(texts):04d}.txt"
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        f.write(resume_text)

        return texts, labels

    def _generate_resume_text(self, template: str, params: dict) -> str:
        """生成单条简历文本的辅助方法"""
        try:
            return template.format(**params)
        except KeyError:
            return None

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # 生成合成数据...
    print(f"生成了训练数据")
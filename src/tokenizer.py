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
# src/tokenizer.py
import jieba
import jieba.posseg as pseg
from typing import List, Tuple
from pathlib import Path
from .cross_platform_utils import safe_path
from config.computer_skills import ALL_COMPUTER_SKILLS
class CrossPlatformTokenizer:
    """跨平台中文分词器"""
    def __init__(self, custom_dict_path: str = None):
        # 1. 添加基础计算机专业词汇
        for skill in ALL_COMPUTER_SKILLS:
            jieba.add_word(skill, freq=1000, tag='nz')
        # 2. 添加带版本号的技能变体（重要！让分词器能识别完整技能）
        skill_versions = {
            # 编程语言
            'Java': ['Java 8', 'Java 11', 'Java 17', 'Java 21', 'Java SE'],
            'Python': ['Python 2.7', 'Python 3.6', 'Python 3.7', 'Python 3.8',
                       'Python 3.9', 'Python 3.10', 'Python 3.11', 'Python 3.12'],
            'Go': ['Go 1.16', 'Go 1.17', 'Go 1.18', 'Go 1.19', 'Go 1.20', 'Go 1.21'],
            'Node.js': ['Node.js 10', 'Node.js 12', 'Node.js 14', 'Node.js 16', 'Node.js 18', 'Node.js 20'],
            # 前端框架
            'Vue': ['Vue 2', 'Vue 3', 'Vue.js 2', 'Vue.js 3'],
            'React': ['React 15', 'React 16', 'React 17', 'React 18'],
            'Angular': ['AngularJS', 'Angular 2', 'Angular 4', 'Angular 6',
                        'Angular 8', 'Angular 10', 'Angular 12', 'Angular 14', 'Angular 16', 'Angular 17'],
            # 后端框架
            'Spring': ['Spring 4', 'Spring 5', 'Spring 6',
                       'Spring Boot 1.x', 'Spring Boot 2.0', 'Spring Boot 2.1',
                       'Spring Boot 2.2', 'Spring Boot 2.3', 'Spring Boot 2.4',
                       'Spring Boot 2.5', 'Spring Boot 2.6', 'Spring Boot 2.7',
                       'Spring Boot 3.0', 'Spring Boot 3.1', 'Spring Boot 3.2',
                       'Spring Cloud 2020', 'Spring Cloud 2021', 'Spring Cloud 2022', 'Spring Cloud 2023'],
            'Django': ['Django 2.x', 'Django 3.x', 'Django 4.x', 'Django 5.x'],
            'Flask': ['Flask 1.x', 'Flask 2.0', 'Flask 2.1', 'Flask 2.2', 'Flask 2.3', 'Flask 3.0'],
            # 数据库
            'MySQL': ['MySQL 5.5', 'MySQL 5.6', 'MySQL 5.7', 'MySQL 8.0'],
            'PostgreSQL': ['PostgreSQL 9.x', 'PostgreSQL 10', 'PostgreSQL 11',
                           'PostgreSQL 12', 'PostgreSQL 13', 'PostgreSQL 14', 'PostgreSQL 15', 'PostgreSQL 16'],
            'MongoDB': ['MongoDB 3.x', 'MongoDB 4.0', 'MongoDB 4.2', 'MongoDB 4.4',
                        'MongoDB 5.0', 'MongoDB 6.0', 'MongoDB 7.0'],
            'Redis': ['Redis 3.x', 'Redis 4.x', 'Redis 5.0', 'Redis 6.0', 'Redis 6.2', 'Redis 7.0', 'Redis 7.2'],
            'Elasticsearch': ['Elasticsearch 5.x', 'Elasticsearch 6.x', 'Elasticsearch 7.x', 'Elasticsearch 8.x'],
            # 云服务和容器
            'Docker': ['Docker 17.x', 'Docker 18.x', 'Docker 19.x', 'Docker 20.x', 'Docker 23.x', 'Docker 24.x'],
            'Kubernetes': ['Kubernetes 1.18', 'Kubernetes 1.19', 'Kubernetes 1.20', 'Kubernetes 1.21',
                           'Kubernetes 1.22', 'Kubernetes 1.23', 'Kubernetes 1.24', 'Kubernetes 1.25',
                           'Kubernetes 1.26', 'Kubernetes 1.27', 'Kubernetes 1.28', 'Kubernetes 1.29',
                           'K8s', 'K8s 1.24', 'K8s 1.28'],
            # 云服务缩写
            'Amazon S3': ['S3', 'S 3', 'Amazon S3'],
            'Amazon EC2': ['EC2', 'EC 2', 'Amazon EC2'],
            'AWS': ['AWS', 'Amazon Web Services'],
            # 操作系统
            'CentOS': ['CentOS 6', 'CentOS 7', 'CentOS 8', 'CentOS Stream'],
            'Ubuntu': ['Ubuntu 14.04', 'Ubuntu 16.04', 'Ubuntu 18.04', 'Ubuntu 20.04', 'Ubuntu 22.04', 'Ubuntu 24.04'],
            'Debian': ['Debian 8', 'Debian 9', 'Debian 10', 'Debian 11', 'Debian 12'],
            'Windows': ['Windows 7', 'Windows 8', 'Windows 10', 'Windows 11', 'Windows Server'],
        }
        for base_skill, versions in skill_versions.items():
            if base_skill in ALL_COMPUTER_SKILLS:
                for version in versions:
                    jieba.add_word(version, freq=2000, tag='nz')
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
# src/computer_resume_processor.py
import re
from typing import List, Dict
from config.computer_skills import ALL_COMPUTER_SKILLS

class ComputerResumeProcessor:
    """计算机专业简历专用处理器"""

    def __init__(self):
        self.skill_patterns = {
            # 版本号模式
            'version_pattern': r'\b([A-Za-z][\w\s]*?)\s*(?:v|version|ver)?\s*(\d+(?:\.\d+)*)(?!\w)',
            # 框架版本模式
            'framework_version': r'\b(React|Vue|Angular|Django|Spring|Flask|Express|Laravel)\s*(?:JS|\.js)?\s*(?:v|version)?\s*(\d+(?:\.\d+)*)',
            # 编程语言 + 框架组合
            'lang_framework': r'\b(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|C#)\s+(?:with|and|using)\s+(Django|Flask|FastAPI|Spring|React|Vue|Angular|Express|NestJS)',
            # 云服务模式
            'cloud_services': r'\b(AWS|Azure|Google\s+Cloud|阿里云|腾讯云|华为云)\s+(?:services?|platform|ecosystem|cloud)',
            # 数据库模式
            'database_pattern': r'\b(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Oracle|SQL\s+Server)\s+(?:database|db|cluster|instance)',
            # 容器和编排
            'container_pattern': r'\b(Docker|Kubernetes|K8s|Helm|Podman)\s+(?:container|orchestration|deployment|cluster)',
            # CI/CD 工具
            'cicd_pattern': r'\b(Jenkins|GitLab\s+CI|GitHub\s+Actions|CircleCI|Travis\s+CI)\s+(?:pipeline|workflow|automation)'
        }

    def enhance_skill_extraction(self, resume_text: str, base_skills: List[str]) -> List[str]:
        """增强技能提取，处理计算机专业特有的模式"""
        enhanced_skills = set(base_skills)
        text_lower = resume_text.lower()

        # 处理版本号
        version_matches = re.finditer(self.skill_patterns['version_pattern'], resume_text, re.IGNORECASE)
        for match in version_matches:
            skill_name = match.group(1).strip()
            if self._is_known_computer_skill(skill_name):
                versioned_skill = f"{skill_name} {match.group(2)}"
                enhanced_skills.add(versioned_skill)
                enhanced_skills.add(skill_name)

        # 处理框架组合
        framework_matches = re.finditer(self.skill_patterns['lang_framework'], resume_text, re.IGNORECASE)
        for match in framework_matches:
            lang, framework = match.groups()
            enhanced_skills.add(lang.strip())
            enhanced_skills.add(framework.strip())
            enhanced_skills.add(f"{lang.strip()} + {framework.strip()}")

        # 处理云服务
        cloud_matches = re.finditer(self.skill_patterns['cloud_services'], resume_text, re.IGNORECASE)
        for match in cloud_matches:
            cloud_service = re.sub(r'\s+', '', match.group(1))
            enhanced_skills.add(cloud_service)

        # 处理数据库
        db_matches = re.finditer(self.skill_patterns['database_pattern'], resume_text, re.IGNORECASE)
        for match in db_matches:
            db_name = match.group(1).strip()
            enhanced_skills.add(db_name)

        # 处理容器技术
        container_matches = re.finditer(self.skill_patterns['container_pattern'], resume_text, re.IGNORECASE)
        for match in container_matches:
            container_tech = match.group(1).strip()
            enhanced_skills.add(container_tech)

        return list(enhanced_skills)

    def _is_known_computer_skill(self, skill: str) -> bool:
        """检查是否为已知的计算机技能"""
        skill_clean = skill.strip().lower()
        return (skill_clean in [s.lower() for s in ALL_COMPUTER_SKILLS] or
                any(skill_clean in known_skill.lower() for known_skill in ALL_COMPUTER_SKILLS))

    def extract_technical_context(self, resume_text: str) -> Dict[str, List[str]]:
        """提取技术上下文信息"""
        context = {
            'experience_level': [],
            'project_types': [],
            'domains': [],
            'methodologies': []
        }

        # 经验水平
        experience_patterns = {
            'expert': r'\b(精通|专家|资深|expert|senior|lead|principal)\b',
            'proficient': r'\b(熟练|熟悉|proficient|experienced|intermediate)\b',
            'basic': r'\b(了解|基础|basic|familiar|junior|entry)\b'
        }

        for level, pattern in experience_patterns.items():
            if re.search(pattern, resume_text, re.IGNORECASE):
                context['experience_level'].append(level)

        # 项目类型
        project_types = ['web应用', '移动应用', '数据分析', '机器学习', '微服务', '区块链', '物联网', '桌面应用']
        for proj_type in project_types:
            if proj_type in resume_text:
                context['project_types'].append(proj_type)

        # 领域
        domains = ['金融科技', '电子商务', '社交网络', '医疗健康', '教育科技', '游戏开发', '企业软件', '政府项目']
        for domain in domains:
            if domain in resume_text:
                context['domains'].append(domain)

        # 开发方法论
        methodologies = ['敏捷开发', 'Scrum', 'Kanban', '瀑布模型', 'DevOps', '持续集成', '测试驱动']
        for method in methodologies:
            if method in resume_text:
                context['methodologies'].append(method)

        return context

# 使用示例
if __name__ == "__main__":
    processor = ComputerResumeProcessor()
    # 测试代码...
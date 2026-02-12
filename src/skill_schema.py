# src/skill_schema.py

SKILL_CATEGORIES = {
    "LANGUAGE": [
        "Python", "Java", "C++", "C", "Go", "Rust",
        "JavaScript", "TypeScript", "Kotlin", "PHP"
    ],
    "FRAMEWORK": [
        "Spring", "SpringBoot", "SpringCloud",
        "Django", "Flask",
        "Vue", "React", "Angular",
        "MyBatis", "Hibernate"
    ],
    "DATABASE": [
        "MySQL", "PostgreSQL", "MongoDB",
        "Redis", "Oracle", "SQLite"
    ],
    "AI": [
        "PyTorch", "TensorFlow", "Keras",
        "机器学习", "深度学习", "计算机视觉",
        "NLP", "大模型"
    ],
    "BIGDATA": [
        "Hadoop", "Spark", "Flink", "Hive"
    ],
    "CLOUD": [
        "Docker", "Kubernetes", "Linux",
        "AWS", "阿里云", "腾讯云"
    ]
}

ALL_SKILLS = []
for v in SKILL_CATEGORIES.values():
    ALL_SKILLS.extend(v)

ALL_SKILLS = list(set(ALL_SKILLS))

import json
import random
import os
from resume_to_ner import convert_resume_to_ner

SAVE_PATH = "data/skill_dataset.json"

SKILL_LIBRARY = [
    "Java", "Python", "Spring Boot", "Flask",
    "Redis", "MySQL", "PostgreSQL",
    "MongoDB", "Elasticsearch",
    "Docker", "Kubernetes",
    "Vue", "React",
    "Linux", "Ubuntu", "CentOS"
]

TEMPLATES = [
    "熟悉{skill}开发",
    "使用{skill}构建系统",
    "精通{skill}",
    "掌握{skill}技术",
    "基于{skill}进行项目开发",
    "参与{skill}相关系统建设",
    "在项目中使用{skill}"
]

def create_synthetic_sample():
    skill = random.choice(SKILL_LIBRARY)
    template = random.choice(TEMPLATES)
    sentence = template.format(skill=skill)

    tokens = list(sentence)
    labels = ["O"] * len(tokens)

    start = sentence.index(skill)
    for i in range(len(skill)):
        labels[start+i] = "B-SKILL" if i == 0 else "I-SKILL"

    return {"tokens": tokens, "labels": labels}

def generate_dataset(num_synthetic=5000):

    dataset = []

    # 1️⃣ 合成数据
    for _ in range(num_synthetic):
        dataset.append(create_synthetic_sample())

    # 2️⃣ 加载真实简历增强
    raw_resume_path = "data/raw_resumes"
    if os.path.exists(raw_resume_path):
        for file in os.listdir(raw_resume_path):
            with open(os.path.join(raw_resume_path, file), "r", encoding="utf-8") as f:
                text = f.read()
                ner_data = convert_resume_to_ner(text, SKILL_LIBRARY)
                dataset.extend(ner_data)

    return dataset


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    dataset = generate_dataset(5000)

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ 数据生成完成: {SAVE_PATH}")
    print(f"样本总数: {len(dataset)}")

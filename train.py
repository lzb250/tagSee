# train.py

import os
import argparse
import pandas as pd
import re

from src.skill_registry import SkillRegistry
from src.industrial_dataset_generator import IndustrialResumeGenerator
from src.train_ner import train


def _is_ascii_word_term(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_#.+-]+", term))


def _find_term_spans(text: str, term: str):
    escaped = re.escape(term)
    if _is_ascii_word_term(term):
        pattern = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    else:
        pattern = escaped
    return list(re.finditer(pattern, text, flags=re.IGNORECASE))


def _collect_skill_spans(text: str, registry: SkillRegistry):
    spans = []
    for _, data in registry.standard_index.items():
        terms = [data["standard_name"], *data.get("aliases", []), *data.get("versions", [])]
        for term in terms:
            term = str(term).strip()
            if not term:
                continue
            for match in _find_term_spans(text, term):
                spans.append((match.start(), match.end()))

    # 长片段优先，避免短片段覆盖长技能名
    spans.sort(key=lambda x: (x[1] - x[0], -x[0]), reverse=True)
    return spans


# ===========================
# NER 数据构建器
# ===========================
def build_ner_dataset(
    input_csv="data/generated_resumes.csv",
    output_csv="data/ner_resume_dataset.csv",
    registry: SkillRegistry = None,
):
    """
    构建NER训练数据
    - 使用 SkillRegistry 提取技能
    - 支持别名、版本、重复出现
    """
    if registry is None:
        raise ValueError("SkillRegistry is required!")

    df = pd.read_csv(input_csv)

    records = []

    for _, row in df.iterrows():
        text = str(row["resume_text"])
        labels = ["O"] * len(text)

        # 直接基于标准名/别名/版本打标，避免只匹配标准名导致漏标
        for start, end in _collect_skill_spans(text, registry):
            if not all(tag == "O" for tag in labels[start:end]):
                continue
            labels[start] = "B-SKILL"
            for i in range(start + 1, end):
                labels[i] = "I-SKILL"

        records.append({"text": text, "labels": " ".join(labels)})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"✅ NER数据已保存 {output_csv}")


# ===========================
# 数据生成 + NER + 训练 流水线
# ===========================
def pipeline(
    num_samples=1000,
    generated_csv="data/generated_resumes.csv",
    ner_csv="data/ner_resume_dataset.csv",
    skills_csv="config/skills.csv",
    model_output_dir="models/skill_extraction_model",
    model_path="models/bert-base-chinese",
    max_length=256,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
):
    """
    一键流水线：
    1️⃣ 生成简历数据
    2️⃣ 构建NER训练数据
    3️⃣ 训练BERT模型
    """
    print("🔹 初始化 SkillRegistry ...")
    registry = SkillRegistry(skills_csv)

    print(f"🔹 生成 {num_samples} 条简历数据 ...")
    generator = IndustrialResumeGenerator(registry)
    df = generator.generate_dataset(num_samples)
    generator.save_to_csv(df, generated_csv)

    print("🔹 构建 NER 训练数据 ...")
    build_ner_dataset(input_csv=generated_csv, output_csv=ner_csv, registry=registry)

    print("🔹 开始训练 BERT NER 模型 ...")
    train(
        ner_csv=ner_csv,
        output_dir=model_output_dir,
        model_path=model_path,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
    )

    print("✅ 流水线完成！")


def parse_args():
    parser = argparse.ArgumentParser(description="Resume NER pipeline trainer")
    parser.add_argument("--num-samples", type=int, default=1000, help="生成训练简历数量")
    parser.add_argument("--generated-csv", default="data/generated_resumes.csv")
    parser.add_argument("--ner-csv", default="data/ner_resume_dataset.csv")
    parser.add_argument("--skills-csv", default="config/skills.csv")
    parser.add_argument("--model-output-dir", default="models/skill_extraction_model")
    parser.add_argument("--model-path", default="models/bert-base-chinese")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


# ===========================
# 命令行入口
# ===========================
if __name__ == "__main__":
    args = parse_args()
    pipeline(
        num_samples=args.num_samples,
        generated_csv=args.generated_csv,
        ner_csv=args.ner_csv,
        skills_csv=args.skills_csv,
        model_output_dir=args.model_output_dir,
        model_path=args.model_path,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
    )

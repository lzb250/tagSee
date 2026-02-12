# src/generate_skills_csv.py

import os
import pandas as pd
from config.computer_skills import (
    COMPUTER_SKILLS,
    SKILL_NORMALIZATION,
    SKILL_VERSIONS
)


def build_alias_map():
    """
    把 SKILL_NORMALIZATION 转成：
    标准技能 -> [别名列表]
    """

    alias_map = {}

    for alias, standard in SKILL_NORMALIZATION.items():
        if standard not in alias_map:
            alias_map[standard] = []

        alias_map[standard].append(alias)

    return alias_map


def generate_skills_dataframe():
    alias_map = build_alias_map()

    records = []

    for category, skills in COMPUTER_SKILLS.items():
        for skill in skills:

            aliases = alias_map.get(skill, [])
            versions = SKILL_VERSIONS.get(skill, [])

            record = {
                "skill_name": skill,
                "category": category,
                "aliases": "|".join(aliases) if aliases else "",
                "versions": "|".join(versions) if versions else ""
            }

            records.append(record)

    df = pd.DataFrame(records)

    return df


def main(save_path="config/skills.csv"):
    os.makedirs("config", exist_ok=True)

    df = generate_skills_dataframe()

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ 技能库已生成: {save_path}")
    print(f"📊 共生成 {len(df)} 个技能")
    print("\n📌 示例预览:")
    print(df.head(10))


if __name__ == "__main__":
    main()

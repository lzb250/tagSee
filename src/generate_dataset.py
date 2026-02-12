# src/generate_dataset.py

from src.skill_registry import SkillRegistry
from src.industrial_dataset_generator import IndustrialResumeGenerator


def main():

    registry = SkillRegistry("config/skills.csv")

    generator = IndustrialResumeGenerator(registry)

    df = generator.generate_dataset(num_samples=500)

    generator.save_to_csv(df, "data/generated_resumes.csv")


if __name__ == "__main__":
    main()

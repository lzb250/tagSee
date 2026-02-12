# predict.py

from src.skill_extractor import SkillExtractor

if __name__ == "__main__":
    extractor = SkillExtractor()

    text = """
    本人熟练掌握Python、SpringBoot、MySQL，
    熟悉Docker和Kubernetes，
    参与过基于PyTorch的大模型训练项目。
    """

    skills = extractor.extract(text)

    print("🎯 提取技能:")
    for s in skills:
        print(s)

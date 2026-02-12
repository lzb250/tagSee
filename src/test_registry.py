# test_registry.py

from src.skill_registry import SkillRegistry

registry = SkillRegistry()

text = """
熟练掌握 python3、Java 17、Spring Boot 3.0，
熟悉 k8s、docker、machine learning、
使用 React 18 和 Vue 3 开发项目。
"""

print("🔍 提取技能：")
skills = registry.extract_with_category(text)

for item in skills:
    print(item)

print("\n📊 分类统计：")
stats = registry.category_statistics([x["skill"] for x in skills])
print(stats)

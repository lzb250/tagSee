#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.inference import SkillExtractor
from src.resume_parser import ResumeParser
from src.computer_resume_processor import ComputerResumeProcessor

def test_model(model_path, resume_path):
    """测试模型效果"""
    try:
        # 解析简历
        parser = ResumeParser()
        raw_text = parser.extract_text(resume_path)
        cleaned_text = parser.clean_text(raw_text)

        print(f"📄 处理文件: {os.path.basename(resume_path)}")
        print(f"📝 文本长度: {len(cleaned_text)} 字符")

        # 提取技能
        extractor = SkillExtractor(model_path)
        base_skills = extractor.extract_skills(cleaned_text)

        # 增强处理
        processor = ComputerResumeProcessor()
        enhanced_skills = processor.enhance_skill_extraction(cleaned_text, base_skills)
        final_skills = sorted(list(set(enhanced_skills)))

        print(f"\n🎯 提取结果 ({len(final_skills)} 个技能):")
        for i, skill in enumerate(final_skills, 1):
            print(f"  {i:2d}. {skill}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python test_model.py <model_path> <resume_file>")
        print("示例: python test_model.py ./models/safe_model_v1_final resume1.txt")
        sys.exit(1)

    model_path = sys.argv[1]
    resume_path = sys.argv[2]

    success = test_model(model_path, resume_path)
    sys.exit(0 if success else 1)
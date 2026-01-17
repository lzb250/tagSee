# extract_skills.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from src.cross_platform_utils import setup_encoding, safe_print, safe_path
from src.resume_parser import ResumeParser
from src.tokenizer import CrossPlatformTokenizer
from src.inference import SkillExtractor
from src.computer_resume_processor import ComputerResumeProcessor

class ResumeSkillExtractorApp:
    def __init__(self, model_path: str = None):
        self.parser = ResumeParser()
        self.tokenizer = CrossPlatformTokenizer()
        self.computer_processor = ComputerResumeProcessor()
        self.model_path = safe_path(model_path) if model_path else None
        self.skill_extractor = None

        # 初始化模型（如果提供）
        if self.model_path and self.model_path.exists():
            try:
                self.skill_extractor = SkillExtractor(str(self.model_path))
                safe_print(f"✓ 加载模型: {self.model_path}")
            except Exception as e:
                safe_print(f"⚠ 警告: 无法加载模型 {e}, 使用词典匹配模式")
                self.skill_extractor = None
        else:
            safe_print("ℹ 使用词典匹配模式（未提供模型路径）")

    def process_single_file(self, file_path: str) -> Dict:
        """处理单个简历文件"""
        file_path = safe_path(file_path)

        if not file_path.exists():
            return {"error": f"文件不存在: {file_path}"}

        try:
            # 1. 提取文本
            raw_text = self.parser.extract_text(file_path)
            if not raw_text:
                return {"error": "无法提取文本内容"}

            # 2. 清理文本
            cleaned_text = self.parser.clean_text(raw_text)

            # 3. 提取技能
            if self.skill_extractor:
                # 使用深度学习模型
                base_skills = self.skill_extractor.extract_skills(cleaned_text)
            else:
                # 使用词典匹配
                base_skills = self.tokenizer.find_skills_in_text(cleaned_text)

            # 4. 增强处理（计算机专业）
            enhanced_skills = self.computer_processor.enhance_skill_extraction(
                cleaned_text, base_skills
            )

            # 5. 提取上下文
            context = self.computer_processor.extract_technical_context(cleaned_text)

            return {
                "file": str(file_path.name),
                "full_path": str(file_path),
                "skills": sorted(list(set(enhanced_skills))),
                "context": context,
                "text_length": len(cleaned_text),
                "skill_count": len(set(enhanced_skills))
            }

        except Exception as e:
            return {"error": f"处理文件时出错: {str(e)}", "file": str(file_path.name)}

    def process_directory(self, directory_path: str, output_file: str = None) -> List[Dict]:
        """批量处理简历目录"""
        directory_path = safe_path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            safe_print(f"错误: 目录不存在或不是目录: {directory_path}")
            return []

        supported_extensions = {'.txt', '.pdf', '.docx'}
        results = []

        safe_print(f"正在扫描目录: {directory_path}")

        # 递归查找所有支持的文件
        files_to_process = []
        for file_path in directory_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                files_to_process.append(file_path)

        if not files_to_process:
            safe_print(f"警告: 在 {directory_path} 中未找到支持的文件 (.txt, .pdf, .docx)")
            return []

        safe_print(f"找到 {len(files_to_process)} 个文件待处理")

        # 处理每个文件
        for i, file_path in enumerate(files_to_process, 1):
            safe_print(f"[{i}/{len(files_to_process)}] 处理: {file_path.name}")
            result = self.process_single_file(str(file_path))
            results.append(result)

            if "error" not in result:
                safe_print(f"  ✓ 提取到 {result['skill_count']} 个技能")
            else:
                safe_print(f"  ✗ 错误: {result['error']}")

        # 保存结果（如果指定了输出文件）
        if output_file:
            output_path = safe_path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            safe_print(f"\n结果已保存到: {output_path}")

        return results
def main():
    setup_encoding()

    parser = argparse.ArgumentParser(description='简历技能提取工具')
    parser.add_argument('input', type=str, help='输入文件或目录路径')
    parser.add_argument('--model', type=str, default='./models/skill_extraction_model_final',
                        help='模型路径 (默认: ./models/skill_extraction_model_final)')
    parser.add_argument('--output', type=str, help='输出JSON文件路径 (批量处理时)')
    parser.add_argument('--list-skills', action='store_true',
                        help='仅列出技能，不显示详细信息')

    args = parser.parse_args()

    # 创建提取器
    extractor = ResumeSkillExtractorApp(model_path=args.model)

    input_path = safe_path(args.input)

    if input_path.is_file():
        # 处理单个文件
        result = extractor.process_single_file(str(input_path))

        if "error" in result:
            safe_print(f"错误: {result['error']}")
            sys.exit(1)

        if args.list_skills:
            safe_print(", ".join(result['skills']))
        else:
            safe_print("\n=== 技能提取结果 ===")
            safe_print(f"文件: {result['file']}")
            safe_print(f"技能 ({result['skill_count']}): {', '.join(result['skills'])}")
            safe_print(f"上下文: {result['context']}")

    elif input_path.is_dir():
        # 批量处理目录
        results = extractor.process_directory(str(input_path), args.output)

        if not results:
            sys.exit(1)

        if args.list_skills:
            all_skills = set()
            for result in results:
                if "skills" in result:
                    all_skills.update(result['skills'])
            safe_print(", ".join(sorted(all_skills)))
        else:
            success_count = sum(1 for r in results if "error" not in r)
            error_count = len(results) - success_count

            safe_print(f"\n=== 处理完成 ===")
            safe_print(f"成功: {success_count}, 失败: {error_count}")

            if error_count > 0:
                safe_print("\n错误详情:")
                for result in results:
                    if "error" in result:
                        safe_print(f"  {result['file']}: {result['error']}")

    else:
        safe_print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
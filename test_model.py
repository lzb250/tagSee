#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# 修复 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

import torch
from pathlib import Path
from transformers import BertTokenizer, BertForTokenClassification


sys.path.append(str(Path(__file__).parent))

from src.resume_parser import ResumeParser
from src.computer_resume_processor import ComputerResumeProcessor

def test_model(model_path, resume_files):
    """测试模型效果，可处理单个或多个简历文件"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 加载模型至: {device.upper()}")

        # 加载 tokenizer 和模型
        tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        model = BertForTokenClassification.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()

        # 使用模型配置自动获取标签映射
        id2label = model.config.id2label

        parser = ResumeParser()
        processor = ComputerResumeProcessor()

        # 处理单个文件或列表
        if isinstance(resume_files, str):
            resume_files = [resume_files]

        for resume_path in resume_files:
            raw_text = parser.extract_text(resume_path)
            cleaned_text = parser.clean_text(raw_text)
            print(f"\n📄 处理文件: {os.path.basename(resume_path)}")
            print(f"📝 文本长度: {len(cleaned_text)} 字符")

            # 模型推理
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            preds = predictions[0].cpu().tolist()

            # 基于 B/I-SKILL 标签解析技能
            base_skills = []
            current_skill = ""
            for token, pred in zip(tokens, preds):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                label = id2label.get(pred, "O")
                if label == "B-SKILL":
                    if current_skill:
                        base_skills.append(current_skill)
                    current_skill = token.replace("##", "")
                elif label == "I-SKILL" and current_skill:
                    current_skill += token.replace("##", "")
                else:
                    if current_skill:
                        base_skills.append(current_skill)
                    current_skill = ""
            if current_skill:
                base_skills.append(current_skill)

            # 规则增强，保持版本号绑定
            enhanced_skills = processor.enhance_skill_extraction(cleaned_text, base_skills)
            final_skills = sorted(list(set([s.strip() for s in enhanced_skills if len(s.strip()) > 1])))

            # 输出对比
            print(f"🔹 模型提取技能 ({len(base_skills)}): {', '.join(base_skills)}")
            print(f"🔹 增强后技能 ({len(final_skills)}): {', '.join(final_skills)}")

        return True

    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python test_model.py <model_path> <resume_file1> [resume_file2 ...]")
        sys.exit(1)

    model_path = sys.argv[1]
    resume_files = sys.argv[2:]
    success = test_model(model_path, resume_files)
    sys.exit(0 if success else 1)

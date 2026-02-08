#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import torch
from pathlib import Path
from transformers import BertTokenizer, BertForTokenClassification

# 1. 环境修复与路径设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(str(Path(__file__).parent))

from src.resume_parser import ResumeParser
from src.computer_resume_processor import ComputerResumeProcessor

def test_model(model_path, resume_path):
    """测试模型效果"""
    try:
        # --- [步骤1] 解析简历文本 ---
        parser = ResumeParser()
        raw_text = parser.extract_text(resume_path)
        cleaned_text = parser.clean_text(raw_text)

        print(f"📄 处理文件: {os.path.basename(resume_path)}")
        print(f"📝 文本长度: {len(cleaned_text)} 字符")

        # --- [步骤2] 加载 BERT 模型 (针对 RTX 5070) ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 正在加载模型至: {device.upper()}")

        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForTokenClassification.from_pretrained(model_path).to(device)
        model.eval()

        # --- [步骤3] 模型推理 (NER 提取) ---
        # 限制长度防止长文本报错
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # 解析 ID 还原为文字
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        preds = predictions[0].cpu().tolist()

        base_skills = []
        current_skill = ""

        for token, pred in zip(tokens, preds):
            if token in ["[CLS]", "[SEP]", "[PAD]"]: continue

            # 假设你的标签映射是: 1: B-SKILL, 2: I-SKILL
            if pred == 1:
                if current_skill: base_skills.append(current_skill)
                current_skill = token.replace("##", "")
            elif pred == 2 and current_skill:
                current_skill += token.replace("##", "")
            else:
                if current_skill: base_skills.append(current_skill)
                current_skill = ""
        if current_skill: base_skills.append(current_skill)

        # --- [步骤4] 规则增强处理 ---
        processor = ComputerResumeProcessor()
        # 即使模型没提取出来，规则库也会兜底
        enhanced_skills = processor.enhance_skill_extraction(cleaned_text, base_skills)
        final_skills = sorted(list(set([s.strip() for s in enhanced_skills if len(s.strip()) > 1])))

        # --- [步骤5] 结果展示 ---
        print(f"\n🎯 提取结果 ({len(final_skills)} 个技能):")
        if not final_skills:
            print("  (未识别到技能)")
        else:
            for i, skill in enumerate(final_skills, 1):
                print(f"  {i:2d}. {skill}")

        return True

    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python test_model.py <model_path> <resume_file>")
        sys.exit(1)

    m_path = sys.argv[1]
    r_path = sys.argv[2]

    success = test_model(m_path, r_path)
    sys.exit(0 if success else 1)
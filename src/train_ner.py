# src/train_ner.py

import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

MODEL_PATH = "models/bert-base-chinese"  # 本地BERT路径

# ===========================
# 数据集类
# ===========================
class NERDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.labels_list = ["O", "B-SKILL", "I-SKILL"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        labels_str = str(self.df.iloc[idx]["labels"]).split()
        encodings = self.tokenizer(text, truncation=True, padding="max_length", max_length=256, return_offsets_mapping=True)
        # 对齐标签
        labels = [0] * len(encodings["input_ids"])
        offset_mapping = encodings["offset_mapping"]
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end != 0:
                labels[i] = self.labels_list.index(labels_str[start])
        encodings["labels"] = labels
        return {k: torch.tensor(v) for k, v in encodings.items()}

# ===========================
# 训练函数
# ===========================
def train(ner_csv="data/ner_resume_dataset.csv"):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    dataset = NERDataset(ner_csv, tokenizer)
    model = BertForTokenClassification.from_pretrained(MODEL_PATH, num_labels=3)

    training_args = TrainingArguments(
        output_dir="models/skill_extraction_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        evaluation_strategy="no",
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("models/skill_extraction_model")
    print("✅ BERT NER 模型训练完成")

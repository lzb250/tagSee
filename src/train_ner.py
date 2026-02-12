# src/train_ner.py

import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

MODEL_PATH = "models/bert-base-chinese"  # 本地BERT路径

class NERDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=256):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_list = ["O", "B-SKILL", "I-SKILL"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        labels_str = str(self.df.iloc[idx]["labels"]).split()

        # 字符级 BIO
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True
            # 不返回 offset_mapping
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # 对齐 labels 到 token
        labels = [0] * self.max_length
        for i, char_idx in enumerate(range(min(len(labels_str), self.max_length))):
            char_label = labels_str[char_idx]
            labels[i] = self.labels_list.index(char_label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }


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

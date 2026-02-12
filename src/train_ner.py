# src/train_ner.py

import inspect
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
            return_attention_mask=True,
            return_offsets_mapping=True
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        offsets = tokenized["offset_mapping"]

        # 对齐 labels 到 token：
        # - 特殊token和padding(label=-100)不参与loss
        # - 每个token使用其起始字符位置对应的字符级标签
        labels = [-100] * self.max_length
        for i, (start, end) in enumerate(offsets):
            if attention_mask[i] == 0:
                continue
            if start == end == 0:  # [CLS]/[SEP] 等特殊token
                continue
            if start >= len(labels_str):
                labels[i] = self.labels_list.index("O")
                continue
            char_label = labels_str[start]
            labels[i] = self.labels_list.index(char_label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


# ===========================
# 训练函数
# ===========================
def train(
    ner_csv="data/ner_resume_dataset.csv",
    output_dir="models/skill_extraction_model",
    model_path=MODEL_PATH,
    max_length=256,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    dataset = NERDataset(ner_csv, tokenizer, max_length=max_length)
    model = BertForTokenClassification.from_pretrained(model_path, num_labels=3)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        evaluation_strategy="no",
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ BERT NER 模型训练完成，模型已保存到 {output_dir}")

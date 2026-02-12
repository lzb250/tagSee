# src/train_ner.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from config import *


def load_dataset():
    df = pd.read_csv("data/ner_resume_dataset.csv")

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        label_ids = [LABEL_LIST.index(l) for l in example["labels"].split()]
        label_ids = label_ids[:MAX_LENGTH]
        label_ids += [0] * (MAX_LENGTH - len(label_ids))

        tokens["labels"] = label_ids
        return tokens

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize)

    return dataset


def train():
    dataset = load_dataset()

    model = BertForTokenClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(LABEL_LIST)
    )

    args = TrainingArguments(
        output_dir=SAVE_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained(SAVE_PATH)
    print("✅ 模型训练完成并保存")


if __name__ == "__main__":
    train()

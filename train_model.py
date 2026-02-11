import json
import os
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import precision_recall_fscore_support

DATA_PATH = "data/skill_dataset.json"
MODEL_SAVE_PATH = "models/skill_extraction_model"

LABEL_LIST = ["O", "B-SKILL", "I-SKILL"]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    return dataset.train_test_split(test_size=0.1)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["labels"][word_idx]])
        else:
            label = example["labels"][word_idx]
            if label.startswith("B-"):
                label = label.replace("B-", "I-")
            labels.append(label2id[label])

        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_labels.append(l_)
                true_predictions.append(p_)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="weighted"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    dataset = load_dataset()
    dataset = dataset.map(tokenize_and_align_labels)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("🚀 开始训练...")
    trainer.train()

    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("====================================")
    print("✅ 训练完成")
    print("====================================")

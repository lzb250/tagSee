# src/model_training.py
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.metrics import f1_score

# Dataset
class SkillDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {'O':0, 'B-SKILL':1, 'I-SKILL':2}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_len,
                                  return_tensors='pt')

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # ignore padding
            else:
                label_ids.append(self.label2id[labels[word_idx]])
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label_ids)
        return item

# Train function
def train_skill_extraction_model(train_texts, train_labels,
                                 val_texts, val_labels,
                                 model_name, output_dir,
                                 num_epochs=10, batch_size=16,
                                 learning_rate=2e-5,
                                 warmup_ratio=0.1,
                                 weight_decay=0.01,
                                 max_len=128):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)

    train_dataset = SkillDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = SkillDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * num_epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(total_steps*warmup_ratio),
                                                num_training_steps=total_steps)

    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Evaluation
        f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} - F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(best_model_path)

    return None, best_model_path

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    id2label = {0:'O', 1:'B-SKILL', 2:'I-SKILL'}
    with torch.no_grad():
        for batch in data_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            for i in range(preds.size(0)):
                pred_i = preds[i]
                label_i = batch['labels'][i]
                for j in range(len(label_i)):
                    if label_i[j] != -100:
                        all_preds.append(id2label[pred_i[j].item()])
                        all_labels.append(id2label[label_i[j].item()])

    return f1_score(all_labels, all_preds, average='micro')

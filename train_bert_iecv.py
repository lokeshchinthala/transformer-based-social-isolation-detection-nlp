#!/usr/bin/env python
"""
Social Isolation Detection Model - Internal-External Cross-Validation (IECV)

This script performs Internal-External Cross-Validation (IECV) for social isolation
detection using BERT-large across multiple healthcare sites.

Author: Lokesh Chinthala
Date: 09/05/2025
License: MIT License
"""

# Load the libraries
import os
import re
import json
import string
import random
import numpy as np
import pandas as pd
from collections import Counter
from random import choices

import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


# Random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set random seed
seed_value = 43
set_seed(seed_value)

# Clear GPU Memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# Text cleaning
def clean_text(text):
    """Clean narrative text: remove symbols, bullets, URLs, punctuation, etc."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\n", " ", text).strip()
    text = text.upper()
    return text


# Load json file
def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected JSON format in {file_path}")
        return data


# Prepare data
def prepare_data(file_path, valid_labels):
    data = load_json_file(file_path)
    labelled, unlabelled = [], []
    for entry in data:
        file_name = entry["file_name"]
        text = entry["text"]

        # Skip documents that contain the below
        if "HERE ARE SOME EXAMPLES" in text.upper():
            continue
        if "BOREDOM OR LONELINESS" in text.upper():
            continue

        span = entry["span"]
        label = entry["label"]
        span_text = clean_text(span)

        if label in valid_labels:
            if not span_text.strip():
                continue
            labelled.append({"file_name": file_name, "sentence": span_text, "label": label})
        else:
            unlabelled.append({"file_name": file_name, "sentence": clean_text(text)})
    return labelled, unlabelled


# Define valid labels
valid_labels = ['social isolation', 'no social isolation', 'social support']

sites = {
    "site1_data": "/Data/site1_social_isolation_complete_annotated_data_fulltext.json",
    "site2_data": "/Data/site2_social_isolation_complete_annotated_data_fulltext.json",
    "site3_data": "/Data/site3_social_isolation_complete_annotated_data_fulltext.json",
}

site_data = {}
for site, path in sites.items():
    labelled, _ = prepare_data(path, valid_labels)
    site_data[site] = labelled
    print(f"{site}: {len(labelled)} samples")

# Label encoding
label2id = {l: i for i, l in enumerate(valid_labels)}
id2label = {i: l for l, i in label2id.items()}

for data in site_data.values():
    for d in data:
        d["label_id"] = label2id[d["label"]]


# IECV helper function
def train_eval(train_sites, test_site):
    print(f"\nTraining on {train_sites}, Testing on {test_site}")

    # Combine training data
    train_data = sum([site_data[s] for s in train_sites], [])
    holdout_data = site_data[test_site]

    # Drop duplicates
    df_train = pd.DataFrame(train_data).drop_duplicates(subset=["sentence", "label"])
    df_holdout = pd.DataFrame(holdout_data).drop_duplicates(subset=["sentence", "label"])

    # Convert back to Hugging Face Dataset
    formatted_labelled_data = Dataset.from_pandas(df_train, preserve_index=False)

    # Split clean train data into train and validation dataset for model evaluation during training
    split_data = formatted_labelled_data.train_test_split(test_size=0.2, seed=seed_value)

    # Convert to Hugging Face datasets
    train_ds = split_data['train']
    val_dataset = split_data['test']

    holdout_dataset = Dataset.from_pandas(df_holdout, preserve_index=False)

    # Oversample to balance classes
    counts = Counter([x["label"] for x in train_ds])
    max_count = max(counts.values())
    oversampled_train = []
    for label in counts:
        samples = [ex for ex in train_ds if ex["label"] == label]
        oversampled_train.extend(choices(samples, k=max_count))

    print("Balanced training size:", len(oversampled_train))
    print("Class distribution:", Counter([x["label"] for x in oversampled_train]))

    # Convert train dataset to Hugging Face datasets
    train_dataset = Dataset.from_list(oversampled_train)

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained("/llm_model_files/bert_uncased_large/")

    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=256)

    # Tokenization
    tokenized_train = train_dataset.map(tokenize)
    tokenized_test = val_dataset.map(tokenize)
    tokenized_holdout = holdout_dataset.map(tokenize)

    tokenized_train = tokenized_train.rename_column("label_id", "labels")
    tokenized_test = tokenized_test.rename_column("label_id", "labels")
    tokenized_holdout = tokenized_holdout.rename_column("label_id", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_holdout.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "/llm_model_files/bert_uncased_large/",
        num_labels=len(valid_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=f"bert_iecv_{test_site}_holdout",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"bert_logs_{test_site}",
        seed=seed_value,
        dataloader_num_workers=64,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Get predictions
    preds = trainer.predict(tokenized_holdout)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=valid_labels))
    print(f"Macro F1 for {test_site}: {f1:.4f}")
    return f1


def main():
    # Run IECV
    results = {}
    results["site2_data"] = train_eval(["site1_data", "site3_data"], "site2_data")
    results["site1_data"] = train_eval(["site2_data", "site3_data"], "site1_data")
    results["site3_data"] = train_eval(["site1_data", "site2_data"], "site3_data")

    # Aggregate results
    f1_scores = list(results.values())
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print("\n==============================")
    print("IECV RESULTS SUMMARY")
    print("==============================")
    for site, score in results.items():
        print(f"Holdout {site}: F1 = {score:.4f}")
    print(f"\nMean ± SD F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()

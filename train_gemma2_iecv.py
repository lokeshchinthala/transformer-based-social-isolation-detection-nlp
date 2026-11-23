#!/usr/bin/env python
"""
Social Isolation Detection Model - Cross-Site Validation

This script performs Internal-External cross-validation (IECV) for social isolation detection
using Gemma-2-2b across multiple healthcare sites.

Author: Lokesh Chinthala
Date: 09/05/2025
License: MIT License
"""

# Load the libraries
import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import gc
from collections import Counter
from random import choices

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# --- Configuration ---
model_id = "/llm_model_files/gemma-2-2b" 

valid_labels = ['social isolation', 'no social isolation', 'social support']
label2id = {l: i for i, l in enumerate(valid_labels)}
id2label = {i: l for l, i in label2id.items()}

# Function to set random seed for reproducibility
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
def clear_gpu_memory():
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if torch.cuda.is_available():
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem/1024**3
    total_gb = total_mem/1024**3
    used_gb = total_gb - free_gb    
    print(f"GPU Memory: {used_gb:.2f} GB used/ {total_gb:.2f}GB total ({free_gb:.2f}GB free)")

# Text cleaning
def clean_text(text):
    """Clean narrative text: remove symbols, bullets, URLs, punctuation, etc."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\n", " ", text).strip()
    text = text.upper()
    return text


# Prepare data
def prepare_data(file_path, valid_labels):
    with open(file_path, "r") as f:
        data = json.load(f)
        
    labelled = []
    for entry in data:
        if any(
            x in entry["text"].upper() 
            for x in [
                "HERE ARE SOME EXAMPLES", 
                "�BOREDOM OR LONELINESS"
            ]
        ):
            continue
        
        span_text = clean_text(entry["span"])
        
        if entry["label"] in valid_labels and span_text.strip():
            labelled.append({
                "file_name": entry["file_name"],
                "sentence": span_text,
                "label": entry["label"],
                "label_id": label2id[entry["label"]],                
            })
    return labelled

# --- Data Setup ---
sites = {
    "site1_data": "/Data/site1_social_isolation_complete_annotated_data_fulltext.json",
    "site2_data": "/Data/site2_social_isolation_complete_annotated_data_fulltext.json",
    "site3_data": "/Data/site3_social_isolation_complete_annotated_data_fulltext.json",
}

site_data = {}

for site, path in sites.items():
    site_data[site] = prepare_data(path, valid_labels)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
tokenizer.padding_side = "right"

def tokenize(batch):
    return tokenizer(
        batch["sentence"], 
        truncation=True, 
        max_length=256
    )


# IECV helper function
def train_eval(train_sites, test_site):
    print(f"\n>>> Training on {train_sites}, Testing on {test_site}")
    
    # Build train and test sets
    train_pool = sum([site_data[s] for s in train_sites], [])
    holdout_data = site_data[test_site]

    # Drop duplicates
    df_train = pd.DataFrame(train_pool).drop_duplicates(subset=["sentence", "label"])
    df_holdout = pd.DataFrame(holdout_data).drop_duplicates(subset=["sentence", "label"])

    # Convert back to Hugging Face Dataset
    formatted_labelled_data = Dataset.from_pandas(df_train, preserve_index=False)
    
    # Split clean train data into train and validation dataset for model evaluation during training
    split_data = formatted_labelled_data.train_test_split(test_size=0.2, seed = seed_value)
    
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
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = val_dataset.map(tokenize, batched=True)
    tokenized_holdout = holdout_dataset.map(tokenize, batched=True)

    # Rename labels
    tokenized_train = tokenized_train.rename_column("label_id", "labels")
    tokenized_test = tokenized_test.rename_column("label_id", "labels")
    tokenized_holdout = tokenized_holdout.rename_column("label_id", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_holdout.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(valid_labels),
        id2label=id2label,
        label2id=label2id,
        device_map="auto"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"/results/gemma_2b_{test_site}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # Execute
    trainer.train()
    
    # Evaluate on held-out test site
    preds = trainer.predict(tokenized_holdout)
    
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=valid_labels))
    print(f"Macro F1 for {test_site}: {macro_f1:.4f}")
    
    clear_gpu_memory()
    return macro_f1

def main():
    # Run IECV
    results = {}

    for holdout in ["site2_data", "site1_data", "site3_data"]:
        train_set = [s for s in sites.keys() if s != holdout]
        results[holdout] = train_eval(train_set, holdout)

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

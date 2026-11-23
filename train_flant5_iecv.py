#!/usr/bin/env python
"""
FLAN-T5 IECV for Social Isolation Detection - Cross-Site Validation

This script performs Internal-External cross-validation (IECV) for social isolation detection
using FLAN-T5-large across multiple healthcare sites.

Author: Lokesh Chinthala
Date: 09/05/2025
License: MIT License
"""
# Load the libraries
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

import os
import re
import json
import pandas as pd
import numpy as np
import random
import gc
from random import choices

from datasets import Dataset
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


# Function to set random seed for reproducibility
def set_seed(seed):
    """Set seed for reproducibility across libraries (random, numpy, torch)."""
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

# Define valid labels
valid_labels = ['social isolation', 'no social isolation', 'social support']


# Text cleaning
def clean_text(text):
    """Clean narrative text: remove symbols, bullets, URLs, punctuation, etc."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\n", " ", text).strip()
    text = text.upper()
    return text

# Load json files
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
        if "�BOREDOM OR LONELINESS" in text.upper():
            continue
        
        span = entry["span"]
        label = entry["label"]
        span_text = clean_text(span)
        
        if label in valid_labels:
            if not span_text.strip():
                continue
            labelled.append({"file_name":file_name, "sentence": span_text, "label": label})
        
        else:
            unlabelled.append({"file_name":file_name, "sentence": clean_text(text)})
    return labelled, unlabelled


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


def tokenize_batch(examples, tokenizer):
    """Tokenize batch for FLAN-T5"""
    inputs = []
    for sentence in examples['sentence']:
        input_text = (
            f"Classify the text into exactly one category: 'social isolation', "
            f"'no social isolation', or 'social support'.\n"
            f"Text: {sentence}\n"
            f"Category: "
        )
        inputs.append(input_text)
    
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['label'],
            padding="max_length",
            truncation=True,
            max_length=10
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def predict_batch(texts, model, tokenizer, batch_size=8):
    """Predict batch of texts"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    predicted_labels = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        input_texts = []
        for text in batch_texts:
            input_text = (
                f"Classify the text into exactly one category: 'social isolation', "
                f"'no social isolation', or 'social support'.\n"
                f"Text: {text}\n"
                f"Category: "
            )
            input_texts.append(input_text)
        
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=20,
                num_beams=1,
                do_sample=False
            )
        
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted_labels.extend(batch_preds)
    
    return predicted_labels


# IECV helper function
def train_eval(train_sites, test_site):
    print(f"\nTraining on {train_sites}, Testing on {test_site}")
    
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

    # Tokenization for FLAN-T5
    model_name = '/llm_model_files/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenize_batch(examples, tokenizer)
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "label", "file_name"]
    )
    
    tokenized_test = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "label", "file_name"]
    )
    

    # Model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Training args
    training_args = TrainingArguments(
        output_dir=f"/results/flan_t5_iecv_{test_site}_holdout",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"/logs/flan_t5_logs_{test_site}",
        seed=seed_value,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    trainer.train()

    true_labels = df_holdout["label"].tolist()
    test_sentences = df_holdout["sentence"].tolist()
    predicted_labels = predict_batch(test_sentences, model, tokenizer)
    
    # Calculate metrics
    y_true = [valid_labels.index(label) for label in true_labels]
    y_pred = [valid_labels.index(label) if label in valid_labels else 0 for label in predicted_labels]
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=valid_labels))
    print(f"Macro F1 for {test_site}: {macro_f1:.4f}")
    
    clear_gpu_memory()
    return macro_f1


def main():
    # --- Execution ---
    results = {}

    for holdout in ["site3_data", "site2_data", "site1_data"]:
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

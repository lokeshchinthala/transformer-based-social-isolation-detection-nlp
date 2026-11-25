#!/usr/bin/env python
"""
FLAN-T5 IECV for Social Isolation Detection using the combined multisite dataset without oversampling of minority classes

This script performs full fine-tuning of FLAN-T5 IECV for social isolation detection across multiple healthcare sites.

Author: Lokesh Chinthala
Date: 03/10/2026
License: MIT License
"""

# Load libraries
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

import os
import re
import string
import json
import pandas as pd
import numpy as np
import glob
import random
import gc
import csv
from random import choices

from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from collections import Counter

import nltk

# Classifiction
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report


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


# Clear GPU Memory
def clear_gpu_memory():
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# #Helper functions


def clean_text(text):
    """Clean narrative text: remove symbols, bullets, URLs, punctuation, etc."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\n", " ", text).strip()
    text = text.upper()
    return text


def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected JSON format in {file_path}")
        return data


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


def train_eval(site_data, valid_labels, seed_value):
    
    # Build train and test sets
    df_all = pd.concat(
        {site: pd.DataFrame(data) for site, data in site_data.items()}, 
        names = ['site']
    ).reset_index(level='site')
     
    # Drop duplicates
    df_unique = df_all.drop_duplicates(subset=["sentence", "label"], keep = 'first').copy()
    
    # Split on unique spans for training and holdout
    finetune_df, holdout_df = train_test_split(
        df_unique,
        test_size=0.2,
        random_state=seed_value,
        stratify=df_unique['label']
    )
    
    print(f"Finetune spans: {len(finetune_df)}, Holdout spans: {len(holdout_df)}")
    
    # Split on unique spans for finetuning
    train_df, test_df = train_test_split(
        finetune_df,
        test_size=0.2,
        random_state=seed_value,
        stratify=finetune_df['label']
    )
    
    print(f"Training spans: {len(train_df)}, Test spans: {len(test_df)}")
    
    # Convert to Hugging Face datasets
    train_data = Dataset.from_pandas(train_df, preserve_index=False)
    test_data = Dataset.from_pandas(test_df, preserve_index=False)
#     holdout_data = Dataset.from_pandas(holdout_df)

    # Tokenization for FLAN-T5
    model_name = '/llm_model_files/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenize_batch(examples, tokenizer)
    
    tokenized_train = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["site", "sentence", "label", "file_name"]
    )
    
    tokenized_test = test_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["site", "sentence", "label", "file_name"]
    )

    # Model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Training args
    training_args = TrainingArguments(
        output_dir=f"/results/flan_t5_all_data_wo_oversampling",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"/logs/flan_t5_all_data_wo_oversampling",
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
    
    # Evaluate on held-out test site
    true_labels = holdout_df["label"].tolist()
    test_sentences = holdout_df["sentence"].tolist()
    predicted_labels = predict_batch(test_sentences, model, tokenizer)
    
    # Calculate metrics
    y_true = [valid_labels.index(label) for label in true_labels]
    y_pred = [valid_labels.index(label) if label in valid_labels else 0 
              for label in predicted_labels]
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=valid_labels))
    print(f"Macro F1: {macro_f1:.4f}")
    
    # Error analysis
    header = ['True_label', 'Predicted_label', 'Span']
    
    with open('FLAN_T5_wo_oversampling_errors.csv', 'w', encoding="utf-8", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t, p, s in zip(true_labels, predicted_labels, test_sentences):
            if (p == "social isolation" and t != "social isolation") or \
            (p == "social support" and t != "social support") or \
            (p == "no social isolation" and t != "no social isolation"):
                s = s.lower()
                print(f"FP -> Predicted: {p}, Actual: {t}, Text: {s}")
                writer.writerow([t, p, s])

    # Manually save the model and tokenizer
    save_dir = '/finetuned_models/manual_save/flan_t5_all_data_final_wo_oversampling'
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    clear_gpu_memory()
    return macro_f1


# Execution
def main():
    # Define valid labels
    valid_labels = ['social isolation', 'no social isolation', 'social support']

    # Set random seed
    seed_value = 43
    set_seed(seed_value)

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

    results = train_eval(site_data, valid_labels, seed_value)

if __name__ == "__main__":
    main()
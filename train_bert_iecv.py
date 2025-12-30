#!/usr/bin/env python
"""
Social Isolation Detection Model - Cross-Site Validation

This script performs Internal-External cross-validation (IECV) for social isolation detection
using BERT-large across multiple healthcare sites.
"""

import os
import re
import json
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


def set_seed(seed: int = 43) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    """
    Clean narrative text.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
     
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Convert to uppercase for consistency
    text = text.upper()
    
    return text


def load_json_file(file_path: str) -> list:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data entries
        
    Raises:
        ValueError: If JSON format is unexpected
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r") as file:
        data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected JSON format in {file_path}. Expected list.")
        return data


def prepare_data(file_path: str, valid_labels: list) -> tuple:
    """
    Prepare labelled and unlabelled data from JSON file.
    
    Args:
        file_path: Path to JSON data file
        valid_labels: List of valid label categories
        
    Returns:
        Tuple of (labelled_data, unlabelled_data)
    """
    data = load_json_file(file_path)
    labelled, unlabelled = [], []
    
    for entry in data:
        file_name = entry.get("file_name", "")
        text = entry.get("text", "")
        span = entry.get("span", "")
        label = entry.get("label", "")
        
        # Skip documents with example content
        text_upper = text.upper()
        if "HERE ARE SOME EXAMPLES" in text_upper:
            continue
        if "�BOREDOM OR LONELINESS" in text_upper:
            continue
        
        span_text = clean_text(span)
        
        if label in valid_labels:
            if not span_text.strip():
                continue
            labelled.append({
                "file_name": file_name,
                "sentence": span_text,
                "label": label
            })
        else:
            unlabelled.append({
                "file_name": file_name,
                "sentence": clean_text(text)
            })

    return labelled, unlabelled

def train_eval(
    train_sites: list,
    test_site: str,
    site_data: dict,
    valid_labels: list,
    label2id: dict,
    id2label: dict,
    model_path: str,
    seed_value: int
) -> float:
    """
    Train model on training sites and evaluate on test site.
    
    Args:
        train_sites: List of site names for training
        test_site: Site name for testing
        site_data: Dictionary containing data for all sites
        valid_labels: List of valid labels
        label2id: Label to ID mapping
        id2label: ID to label mapping
        model_path: Path to pre-trained BERT model
        seed_value: Random seed for reproducibility
        
    Returns:
        Macro F1 score for the test site
    """
    print(f"\n{'='*60}")
    print(f"Training on: {train_sites}")
    print(f"Testing on: {test_site}")
    print(f"{'='*60}")
    
    # Combine training data from all training sites
    train_data = []
    for site in train_sites:
        train_data.extend(site_data[site])
    
    test_data = site_data[test_site]
    
    # Prepare training DataFrame with deduplication
    df_train = pd.DataFrame(train_data).drop_duplicates(
        subset=["file_name", "sentence", "label"]
    )
    
    # Handle class imbalance
    no_si_train = df_train[df_train['label'] == 'no social isolation']
    no_si_train = no_si_train.drop_duplicates(subset=["sentence", "label"])
    
    others_train = df_train[df_train['label'] != 'no social isolation']
    df_train_balanced = pd.concat([no_si_train, others_train], ignore_index=True)
    
    # Convert to Hugging Face Dataset
    train_data_balanced = Dataset.from_pandas(df_train_balanced, preserve_index=False)
    
    # Oversample minority classes
    counts = Counter([x["label"] for x in train_data_balanced])
    max_count = max(counts.values())
    oversampled = []
    
    for label in counts:
        samples = [ex for ex in train_data_balanced if ex["label"] == label]
        oversampled.extend(choices(samples, k=max_count))
    
    print(f"\nTraining Statistics:")
    print(f"Original size: {len(train_data_balanced)}")
    print(f"Balanced size: {len(oversampled)}")
    print(f"Class distribution: {Counter([x['label'] for x in oversampled])}")
    
    # Prepare datasets
    train_dataset = Dataset.from_list(oversampled)
    test_dataset = Dataset.from_list(test_data)
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    def tokenize(batch):
        """Tokenize a batch of sentences."""
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    
    # Rename columns for compatibility with Trainer
    tokenized_train = tokenized_train.rename_column("label_id", "labels")
    tokenized_test = tokenized_test.rename_column("label_id", "labels")
    
    # Set format for PyTorch
    tokenized_train.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    tokenized_test.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # Initialize model
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(valid_labels),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"/results/bert_iecv_{test_site}_holdout",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"/logs/bert_logs_{test_site}",
        seed=seed_value,
        dataloader_num_workers=64,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )
    
    # Train model
    print(f"\nStarting training for {test_site} holdout...")
    trainer.train()
    
    # Evaluate model
    print(f"\nEvaluating on {test_site}...")
    preds = trainer.predict(tokenized_test)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    # Print detailed classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=valid_labels))
    print(f"\nMacro F1 Score for {test_site}: {f1:.4f}")
    print("="*60)
    
    return f1


def main():
    """
    Main function to run the IECV experiment.
    """
    # Set random seed for reproducibility
    seed_value = 43
    set_seed(seed_value)
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Define valid labels
    valid_labels = ['social isolation', 'no social isolation', 'social support']
    
    # Define label mappings
    label2id = {label: i for i, label in enumerate(valid_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Define data paths
    sites = {
        "site1_data": "/data/site1_annotated_data.json",
        "site2_data": "/data/site2_annotated_data.json",
        "site3_data": "/data/site3_annotated_data.json",
    }
    
    # Path to pre-trained BERT model
    model_path = "/llm_model_files/bert_uncased_large/"
    
    # Load and prepare data for all sites
    print("Loading and preparing data...")
    site_data = {}
    
    for site_name, file_path in sites.items():
        try:
            labelled, unlabelled = prepare_data(file_path, valid_labels)
            
            # Add label IDs
            for entry in labelled:
                entry["label_id"] = label2id[entry["label"]]
            
            site_data[site_name] = labelled
            
            print(f"{site_name}: {len(labelled)} labelled samples")
            print(f"{site_name}: {len(unlabelled)} unlabelled samples")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping {site_name}.")
            site_data[site_name] = []
        except Exception as e:
            print(f"Error loading {site_name}: {e}")
            site_data[site_name] = []
    
    # Run Internal-External Cross-Validation (IECV)
    print("\n" + "="*60)
    print("STARTING INTERNAL-EXTERNAL CROSS-VALIDATION (IECV)")
    print("="*60)
    
    results = {}
    
    # Site 2 as holdout
    results["site2_data"] = train_eval(
        ["site1_data", "site3_data"],
        "site2_data",
        site_data,
        valid_labels,
        label2id,
        id2label,
        model_path,
        seed_value
    )
    
    # Site 1 as holdout
    results["site1_data"] = train_eval(
        ["site2_data", "site3_data"],
        "site1_data",
        site_data,
        valid_labels,
        label2id,
        id2label,
        model_path,
        seed_value
    )
    
    # Site 3 as holdout
    results["site3_data"] = train_eval(
        ["site1_data", "site2_data"],
        "site3_data",
        site_data,
        valid_labels,
        label2id,
        id2label,
        model_path,
        seed_value
    )
    
    # Aggregate and display results
    print("\n" + "="*60)
    print("IECV RESULTS SUMMARY")
    print("="*60)
    
    f1_scores = []
    for site, score in results.items():
        print(f"Holdout {site}: F1 = {score:.4f}")
        f1_scores.append(score)
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*60)
    
    # Save results to file
    results_summary = {
        "individual_scores": results,
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
        "seed": seed_value,
        "model": model_path
    }
    
    os.makedirs("/results", exist_ok=True)
    with open("/results/bert_iecv_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nResults saved to /results/bert_iecv_results.json")


if __name__ == "__main__":
    main()
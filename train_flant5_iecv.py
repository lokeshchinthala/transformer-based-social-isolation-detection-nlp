#!/usr/bin/env python
"""
FLAN-T5 IECV for Social Isolation Detection - Cross-Site Validation

This script performs Internal-External cross-validation (IECV) for social isolation detection
using FLAN-T5-large across multiple healthcare sites.
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
    T5Tokenizer,
    T5ForConditionalGeneration,
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


def tokenize_batch(examples: dict, tokenizer: T5Tokenizer) -> dict:
    """
    Tokenize a batch of examples for FLAN-T5.
    
    Args:
        examples: Batch of examples with 'sentence' and 'label'
        tokenizer: T5 tokenizer
        
    Returns:
        Tokenized batch with input_ids and labels
    """
    inputs = []
    for sentence in examples['sentence']:
        input_text = (
            f"Classify the text into exactly one category: 'social isolation', "
            f"'no social isolation' or 'social support'.\n"
            f"Text: {sentence}\n"
            f"Category:"
        )
        inputs.append(input_text)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
    # Tokenize labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['label'],
            padding="max_length",
            truncation=True,
            max_length=10
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def predict_batch(texts: list, model: T5ForConditionalGeneration, 
                  tokenizer: T5Tokenizer) -> list:
    """
    Generate predictions for a batch of texts.
    
    Args:
        texts: List of input texts
        model: Fine-tuned FLAN-T5 model
        tokenizer: T5 tokenizer
        
    Returns:
        List of predicted labels
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Prepare inputs
    input_texts = []
    for text in texts:
        input_text = (
            f"Classify the text into exactly one category: 'social isolation', "
            f"'no social isolation' or 'social support'.\n"
            f"Text: {text}\n"
            f"Category:"
        )
        input_texts.append(input_text)
    
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=20,
            num_beams=1,
            do_sample=False
        )
    
    # Decode predictions
    predicted_labels = []
    for output in outputs:
        predicted_label = tokenizer.decode(output, skip_special_tokens=True)
        predicted_labels.append(predicted_label)
    
    return predicted_labels


def train_eval(
    train_sites: list,
    test_site: str,
    site_data: dict,
    valid_labels: list,
    model_name: str,
    seed_value: int
) -> float:
    """
    Train FLAN-T5 model on training sites and evaluate on test site.
    
    Args:
        train_sites: List of site names for training
        test_site: Site name for testing
        site_data: Dictionary containing data for all sites
        valid_labels: List of valid labels
        model_name: Path to pre-trained FLAN-T5 model
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
    
    # Handle class imbalance - remove duplicates from 'no social isolation' class
    no_si_train = df_train[df_train['label'] == 'no social isolation']
    no_si_train = no_si_train.drop_duplicates(subset=["sentence", "label"])
    
    others_train = df_train[df_train['label'] != 'no social isolation']
    df_train_balanced = pd.concat([no_si_train, others_train], ignore_index=True)
    
    # Convert to Hugging Face Dataset
    train_data_balanced = Dataset.from_pandas(df_train_balanced, preserve_index=False)
    
    # Calculate class distribution for oversampling
    counts = Counter([x["label"] for x in train_data_balanced])
    max_count = max(counts.values())
    
    # Oversample minority classes
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
    print("\nLoading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Tokenize datasets
    print("Tokenizing datasets")
    
    def tokenize_function(examples):
        return tokenize_batch(examples, tokenizer)
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "label", "file_name"]
    )
    
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "label", "file_name"]
    )
    
    # Initialize model
    print("Loading FLAN-T5 model")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Set up training arguments
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
        dataloader_num_workers=64,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        push_to_hub=False,
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
    print(f"\nStarting training for {test_site} holdout")
    trainer.train()
    
    # Evaluate model
    print(f"\nEvaluating on {test_site}")
    
    # Get true labels
    true_labels = [item["label"] for item in test_data]
    
    # Get predictions
    test_sentences = [item["sentence"] for item in test_data]
    predicted_labels = predict_batch(test_sentences, model, tokenizer)
    
    # Calculate metrics
    f1 = f1_score(true_labels, predicted_labels, average="macro", 
                  labels=valid_labels, zero_division=0)
    
    # Print detailed classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, predicted_labels, target_names=valid_labels, zero_division=0))
    
    # Print false positives for analysis
    print("\n" + "="*60)
    print("FALSE POSITIVES ANALYSIS (predicted as 'social isolation')")
    print("="*60)
    
    fp_count = 0
    for true, pred, text in zip(true_labels, predicted_labels, test_sentences):
        if pred == "social isolation" and true != "social isolation":
            fp_count += 1
            if fp_count <= 10:  # Show first 10 FPs
                print(f"\nFP {fp_count}:")
                print(f"Predicted: {pred}")
                print(f"Actual: {true}")
                print(f"Text: {text[:100]}")
    
    print(f"\nTotal false positives: {fp_count}")
    print(f"Macro F1 Score for {test_site}: {f1:.4f}")
    print("="*60)
    
    # Save predictions for detailed analysis
    results_df = pd.DataFrame({
        'text': test_sentences,
        'true_label': true_labels,
        'predicted_label': predicted_labels
    })
    
    results_dir = f"/results/{test_site}_predictions"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f"{results_dir}/predictions.csv", index=False)
    
    # Save the fine-tuned model
    model_save_dir = f"/saved_models/flan_t5_{test_site}_holdout"
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    
    return f1


def main():
    """
    Main function to run the IECV experiment with FLAN-T5.
    """
    # Set random seed for reproducibility
    seed_value = 43
    set_seed(seed_value)
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Define valid labels
    valid_labels = ['social isolation', 'no social isolation', 'social support']
    
    # Define data paths
    sites = {
        "site1_data": "/data/site1_annotated_data.json",
        "site2_data": "/data/site2_annotated_data.json",
        "site3_data": "/data/site3_annotated_data.json",
    }
    
    # Path to pre-trained FLAN-T5 model
    model_name = "/llm_model_files/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/"
    
    # Create output directories
    os.makedirs("/results", exist_ok=True)
    os.makedirs("/logs", exist_ok=True)
    os.makedirs("/saved_models", exist_ok=True)
    
    # Load and prepare data for all sites
    print("="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    site_data = {}
    
    for site_name, file_path in sites.items():
        try:
            labelled, unlabelled = prepare_data(file_path, valid_labels)
            site_data[site_name] = labelled
            
            print(f"\n{site_name}:")
            print(f"Labelled samples: {len(labelled)}")
            print(f"Unlabelled samples: {len(unlabelled)}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping {site_name}.")
            site_data[site_name] = []
        except Exception as e:
            print(f"Error loading {site_name}: {e}")
            site_data[site_name] = []
    
    # Run Internal-External Cross-Validation (IECV)
    print("\n" + "="*60)
    print("STARTING Internal-External CROSS-VALIDATION (IECV) WITH FLAN-T5")
    print("="*60)
    
    results = {}
    
    # Site 2 as holdout
    results["site2_data"] = train_eval(
        ["site1_data", "site3_data"],
        "site2_data",
        site_data,
        valid_labels,
        model_name,
        seed_value
    )
    
    # Site 1 as holdout
    results["site1_data"] = train_eval(
        ["site2_data", "site3_data"],
        "site1_data",
        site_data,
        valid_labels,
        model_name,
        seed_value
    )
    
    # Site 3 as holdout
    results["site3_data"] = train_eval(
        ["site1_data", "site2_data"],
        "site3_data",
        site_data,
        valid_labels,
        model_name,
        seed_value
    )
    
    # Aggregate and display results
    print("\n" + "="*60)
    print("IECV RESULTS SUMMARY - FLAN-T5")
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
        "model": "FLAN-T5-large",
        "individual_scores": results,
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
        "seed": seed_value,
        "valid_labels": valid_labels,
        "training_sites": list(sites.keys())
    }
    
    with open("/results/flan_t5_iecv_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Create comparative analysis
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    # Calculate per-class F1 scores for each site
    for site in sites.keys():
        if site in results and results[site] > 0:
            print(f"\n{site} holdout completed successfully.")
        else:
            print(f"\n{site} holdout had issues.")
    
    print(f"\nOverall model performance:")
    print(f"Best site: {max(results, key=results.get)} (F1 = {results[max(results, key=results.get)]:.4f})")
    print(f"Worst site: {min(results, key=results.get)} (F1 = {results[min(results, key=results.get)]:.4f})")
    print(f"Performance range: {max(f1_scores)-min(f1_scores):.4f}")
    
    print("\nResults saved to:")
    print("/results/flan_t5_iecv_results.json")
    print("/results/[site]_predictions/predictions.csv for each site")
    print("/saved_models/flan_t5_[site]_holdout/ for each model")

if __name__ == "__main__":
    main()

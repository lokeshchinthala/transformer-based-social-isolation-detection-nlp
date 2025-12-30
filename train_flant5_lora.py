#!/usr/bin/env python
"""
FLAN-T5 with LoRA for Social Isolation Detection

This script fine-tunes FLAN-T5-large with LoRA using all labelled data from all sites.
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
from peft import LoraConfig, get_peft_model, TaskType
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
    text = re.sub(r"\s+", " ", text).strip()
    text = text.upper()
    return text

def load_json_file(file_path: str) -> list:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data entries
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected JSON format in {file_path}")
        return data

def prepare_data(file_path: str, valid_labels: list) -> list:
    """
    Prepare labelled data from JSON file.
    
    Args:
        file_path: Path to JSON data file
        valid_labels: List of valid label categories
        
    Returns:
        List of labelled data
    """
    data = load_json_file(file_path)
    labelled = []
    
    for entry in data:
        file_name = entry["file_name"]
        text = entry["text"]
        
        # Skip documents with example content
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
            labelled.append({
                "file_name": file_name,
                "sentence": span_text,
                "label": label
            })    
    return labelled

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

def predict_batch(texts: list, model, tokenizer: T5Tokenizer, batch_size: int = 8) -> list:
    """
    Generate predictions for a batch of texts.
    
    Args:
        texts: List of input texts
        model: Fine-tuned FLAN-T5 model with LoRA
        tokenizer: T5 tokenizer
        batch_size: Batch size for prediction
        
    Returns:
        List of predicted labels
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    predicted_labels = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Prepare inputs
        input_texts = []
        for text in batch_texts:
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
            max_length=128
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=20,
                num_beams=1,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode predictions
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted_labels.extend(batch_preds)
    
    return predicted_labels


def main():
    """
    Main function to train FLAN-T5 with LoRA on all data.
    """
    # Set random seed for reproducibility
    seed_value = 43
    set_seed(seed_value)
    
    # Clear GPU memory
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
    
    # Load and prepare data from all sites
    print("="*60)
    print("LOADING AND CONCATENATING DATA FROM ALL SITES")
    print("="*60)
    
    all_labelled_data = []
    
    for site_name, file_path in sites.items():
        try:
            labelled = prepare_data(file_path, valid_labels)
            all_labelled_data.extend(labelled)
            print(f"{site_name}: {len(labelled)} labelled samples")
        except Exception as e:
            print(f"Error loading {site_name}: {e}")
    
    print(f"\nTotal labelled samples from all sites: {len(all_labelled_data)}")
    
    # Convert to Hugging Face Dataset
    formatted_data = Dataset.from_list(all_labelled_data)
    
    # Calculate statistics
    file_count = len(set(item["file_name"] for item in all_labelled_data))
    class_count = Counter([item["label"] for item in formatted_data])
    total_samples = len(formatted_data)
    
    print(f"\nDataset Statistics:")
    print(f"Unique files: {file_count}")
    print(f"Total samples: {total_samples}")
    print(f"Class distribution:")
    for label, count in class_count.items():
        print(f"{label}: {count}")
    
    # Split into training and validation (80/20)
    print(f"\nSplitting data (80/20 train/test)")
    split_data = formatted_data.train_test_split(
        test_size=0.2,
        seed=seed_value
    )
    
    train_data_raw = split_data['train']
    test_data = split_data['test']
    
    print(f"Training samples: {len(train_data_raw)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Handle class imbalance in training data
    print(f"\nHandling class imbalance...")
    df_train = train_data_raw.to_pandas()
    
    # Remove duplicates from 'no social isolation' class
    no_si_train = df_train[df_train['label'] == 'no social isolation']
    no_si_train = no_si_train.drop_duplicates(subset=["sentence", "label"])
    
    others_train = df_train[df_train['label'] != 'no social isolation']
    df_balanced = pd.concat([no_si_train, others_train], ignore_index=True)
    
    # Convert back to Dataset
    train_dataset = Dataset.from_pandas(df_balanced, preserve_index=False)
    
    # Calculate class distribution for oversampling
    class_count_train = Counter([item["label"] for item in train_dataset])
    max_count_train = max(class_count_train.values())
    
    print(f"\nAfter deduplication:")
    for label, count in class_count_train.items():
        print(f"  {label}: {count}")
    
    # Oversample minority classes
    oversampled_train = []
    for label in class_count_train:
        label_samples = [ex for ex in train_dataset if ex["label"] == label]
        oversampled_label_samples = choices(label_samples, k=max_count_train)
        oversampled_train.extend(oversampled_label_samples)
    
    # Final training dataset
    train_data_balanced = Dataset.from_list(oversampled_train)
    
    final_class_count = Counter([item["label"] for item in train_data_balanced])
    print(f"\nFinal training dataset after oversampling:")
    print(f"Total samples: {len(train_data_balanced)}")
    print(f"Class distribution:")
    for label, count in final_class_count.items():
        print(f"{label}: {count}")
    
    # Initialize tokenizer
    print(f"\n" + "="*60)
    print("INITIALIZING MODEL AND TOKENIZER")
    print("="*60)
    
    print(f"Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Tokenize datasets
    print(f"Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenize_batch(examples, tokenizer)
    
    tokenized_train = train_data_balanced.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "label", "file_name"]
    )
    
    tokenized_test = test_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "label", "file_name"]
    )
    
    # Initialize model with LoRA
    print(f"Loading FLAN-T5 model with LoRA")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="results/flan_t5_lora",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="/logs/flan_t5_lora",
        seed=seed_value,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=10,
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
    print(f"\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    trainer.train()
    
    # Evaluate model
    print(f"\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Generate predictions for test set
    print(f"\nGenerating predictions...")
    true_labels = [item["label"] for item in test_data]
    test_sentences = [item["sentence"] for item in test_data]
    predicted_labels = predict_batch(test_sentences, model, tokenizer, batch_size=8)
    
    # Print classification report
    print(f"\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(classification_report(true_labels, predicted_labels, target_names=valid_labels, zero_division=0))
    
    # Calculate F1 score
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", labels=valid_labels, zero_division=0)
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print("="*60)
    
    # Save model and tokenizer
    print(f"\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    save_dir = "results/flan_t5_lora"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model saved to: {save_dir}")
    
    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        "text": test_sentences,
        "true_label": true_labels,
        "predicted_label": predicted_labels
    })
    
    predictions_df.to_csv("/results/flant5_lora_predictions.csv", index=False)
    print(f"Predictions saved to: flant5_lora_predictions.csv")
    
    # Save training summary
    results_summary = {
        "model": "FLAN-T5-large with LoRA",
        "training_samples": len(train_data_balanced),
        "test_samples": len(test_data),
        "class_distribution_training": dict(final_class_count),
        "class_distribution_test": dict(Counter(true_labels)),
        "macro_f1": float(macro_f1),
        "seed": seed_value,
        "valid_labels": valid_labels,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q", "v"]
        }
    }
    
    with open("/results/flant5_lora_training_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Training summary saved to: flant5_lora_training_summary.json")
    print(f"\n" + "="*60)


if __name__ == "__main__":
    main()
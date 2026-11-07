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
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


# Configuration
seed_value = 43
model_id = "/llm_model_files/gemma-2-2b" 

valid_labels = ['social isolation', 'no social isolation', 'social support']
label2id = {l: i for i, l in enumerate(valid_labels)}
id2label = {i: l for l, i in label2id.items()}

# Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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


from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant=True)


# Text cleaning
def clean_text(text):
    """Clean narrative text: remove symbols, bullets, URLs, punctuation, etc."""
    if not isinstance(text, str):
        return ""
#     text = re.sub(r"[✔✓•■□▪▫●○★☆☑☐❍❏❑⦿⦾•�]", " ", text)
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


# Data Setup
sites = {
    "site1_data": "/Data/site1_annotated_data.json",
    "site2_data": "/Data/site2_annotated_data.json",
    "site3_data": "/Data/site3_annotated_data.json",
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
        max_length=512
    )


# IECV helper function
def train_eval(train_sites, test_site):
    print(f"\n Training on {train_sites}, Testing on {test_site}")
    
    # Build train and test sets
    train_pool = sum([site_data[s] for s in train_sites], [])
    test_data = site_data[test_site]
    
    df_train = pd.DataFrame(train_pool)
    
    # Remove duplicates
    df_train = df_train.drop_duplicates(subset=["file_name", "sentence", "label"])
    
    # reduce repeated negatives
    no_si_train = df_train[df_train['label'] == 'no social isolation']

    no_si_train = no_si_train.drop_duplicates(subset=["sentence", "label"])

    others_train = df_train[df_train['label'] != 'no social isolation']

    df_train = pd.concat([no_si_train, others_train], ignore_index = True)

    
    # Internal Train/Validation Split
    train_data, val_data = train_test_split(
        df_train.to_dict("records"), 
        test_size=0.15, 
        stratify=df_train['label_id'], 
        random_state=seed_value
    )
    
    train_df = pd.DataFrame(train_data)
    
    # Oversample to balance classes
    counts = Counter(train_df["label"])
    
    print("\nOriginal Training Distribution")
    print(counts)
    
    max_count = max(counts.values())
    
    oversampled = []
    
    for label in valid_labels:
        subset = train_df[train_df["label"] == label].to_dict("records")
        if len(subset) > 0:
            oversampled.extend(choices(subset, k=max_count))

    print("Balanced training size:", len(oversampled))
    print("Class distribution:", Counter([x["label"] for x in oversampled]))
    
    # Convert back to Hugging Face Dataset
    train_ds = Dataset.from_list(oversampled)
    val_ds = Dataset.from_list(val_data)
    test_ds = Dataset.from_list(test_data)
    
    # Tokenize
    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_val = val_ds.map(tokenize, batched=True)
    tokenized_test = test_ds.map(tokenize, batched=True)

    # Rename labels
    tokenized_train = tokenized_train.rename_column("label_id", "labels")
    tokenized_val = tokenized_val.rename_column("label_id", "labels")
    tokenized_test = tokenized_test.rename_column("label_id", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load Model with QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(valid_labels),
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"gemma_lora_{test_site}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        gradient_checkpointing=True,
        eval_accumulation_steps=1, 
        optim="paged_adamw_8bit",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    # Execute
    trainer.train()
    
    # Evaluate on held-out test site
    preds = trainer.predict(tokenized_test) 
    
    logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
    
    y_pred = np.argmax(logits, axis=1)
    y_true = preds.label_ids
    
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nResults for {test_site}:")
    print(classification_report(y_true, y_pred, target_names=valid_labels))
    print(f"\nMacro F1 for {test_site}: {macro_f1:.4f}")
    
    clear_gpu_memory()
    return macro_f1


# Run IECV
results = {}
for holdout in ["site1_data", "site2_data", "site3_data"]:
    train_set = [s for s in sites.keys() if s != holdout]
    results[holdout] = train_eval(train_set, holdout)

print("\nFinal IECV Macro F1 Mean:", np.mean(list(results.values())))


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
print("==============================")
# Fine-Tuned Large Language Models for Detecting Social Isolation from Unstructured Clinical Notes

This repository contains the code used in the paper “Fine-Tuned Large Language Models for Detecting Social Isolation from Unstructured Clinical Notes”, which investigates transformer-based approaches for social isolation detection.

## Overview

The script performs:
- Text preprocessing and cleaning of clinical narratives
- Internal-External Cross-Validation (IECV) across three healthcare sites
- Fine-tuning of BERT, RoBERTa, and FLAN-T5 for multi-class classification
- LoRA-based parameter-efficient fine-tuning (PEFT) of FLAN-T5 for multi-class classification
- Evaluation using macro F1 score and detailed classification reports

### 1. Fine-tune BERT
```
python train_bert_iecv.py
```

### 2. Fine-tune RoBERTa
```
python train_roberta_iecv.py
```

### 3. Fine-tune FLAN-T5
```
python train_flant5_iecv.py
```

### 4. Fine-tune FLAN-T5 with LoRA
```
python train_flant5_lora.py
```

Annotated clinical data and pretrained model weights are not distributed through this repository due to privacy and licensing constraints.

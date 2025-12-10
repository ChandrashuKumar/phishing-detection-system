# DistilBERT Model Architecture

## Overview

A transformer-based deep learning model for phishing email/SMS detection that leverages DistilBERT (a distilled version of BERT) to analyze text content for phishing indicators using self-attention mechanisms.

## Model Architecture

### Input Processing

- **Input Type**: Tokenized text sequences (email subject + body or SMS content)
- **Maximum Sequence Length**: 128 tokens
- **Input Shape**: (batch_size, 128)
- **Tokenizer**: DistilBERT tokenizer (distilbert-base-uncased)

### DistilBERT Encoder

| Component | Type | Parameters |
|-----------|------|------------|
| Base Model | DistilBERT | distilbert-base-uncased (6 transformer layers) |
| Hidden Size | 768 | - |
| Attention Heads | 12 | - |
| Vocabulary Size | 30,522 | - |
| Frozen Layers | Embeddings + Layers 0-2 | (First 3 transformer layers frozen) |
| Trainable Layers | Layers 3-5 | (Last 3 transformer layers fine-tuned) |

**Purpose**: Extracts contextual embeddings from input text using multi-head self-attention, capturing semantic meaning, long-range dependencies, and linguistic patterns indicative of phishing attempts.

**Output**: (batch_size, 128, 768) - Contextualized token representations

### Pooling Strategy

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| CLS Token Extraction | Indexing | (batch, 128, 768) → (batch, 768) | Extracts [CLS] token representation |

**Purpose**: Aggregates sequence-level information from the special [CLS] token, which has attended to all tokens through self-attention layers.

**Output**: (batch, 768) - Sentence-level representation

### Classification Head

| Layer | Type | Input → Output | Activation | Dropout |
|-------|------|----------------|------------|---------|
| Dropout | Dropout | (batch, 768) → (batch, 768) | - | 0.3 |
| FC1 | Linear | (batch, 768) → (batch, 256) | ReLU | - |
| Dropout | Dropout | (batch, 256) → (batch, 256) | - | 0.3 |
| FC2 | Linear | (batch, 256) → (batch, 1) | None (logits) | - |

**Purpose**: Transforms the sentence representation into a binary classification decision through non-linear transformations with regularization.

**Output**: Raw logit score (converted to probability via sigmoid during inference)

## Data Flow Summary
```
Input Text
    ↓
Tokenization (batch, 128)
    ↓
DistilBERT Embeddings (batch, 128, 768)
    ↓
DistilBERT Transformer Layer 0 (Frozen)
    ↓
DistilBERT Transformer Layer 1 (Frozen)
    ↓
DistilBERT Transformer Layer 2 (Frozen)
    ↓
DistilBERT Transformer Layer 3 (Trainable)
    ↓
DistilBERT Transformer Layer 4 (Trainable)
    ↓
DistilBERT Transformer Layer 5 (Trainable)
    ↓
Last Hidden State (batch, 128, 768)
    ↓
Extract [CLS] Token (batch, 768)
    ↓
Dropout (0.3)
    ↓
Linear (768 → 256) + ReLU
    ↓
Dropout (0.3)
    ↓
Linear (256 → 1)
    ↓
Output Logit
```

## Training Configuration

### Optimizer
- **Type**: AdamW (with weight decay)
- **Learning Rates**:
  - DistilBERT layers: 2e-5 (discriminative fine-tuning)
  - Classification head: 1e-3
- **Weight Decay**: 0.01

### Loss Function
- **Type**: BCEWithLogitsLoss
- **Class Weighting**: None (balanced dataset)

### Training Strategy
- **Batch Size**: 12
- **Gradient Accumulation**: 3 steps (effective batch size: 36)
- **Mixed Precision**: FP16 (Automatic Mixed Precision enabled)
- **Early Stopping**: Patience of 2 epochs
- **Epochs**: 5 (with early stopping)

### Regularization
- **Dropout**: 0.3 (in classification head)
- **Layer Freezing**: First 50% of DistilBERT layers frozen
- **Weight Decay**: 0.01

## Model Statistics

- **Total Parameters**: 67,149,313
- **Trainable Parameters**: 22,050,049 (32.8%)
- **Frozen Parameters**: 45,099,264 (67.2%)
- **DistilBERT Parameters**: ~66.9M
- **Classification Head Parameters**: ~249K

## Architecture Highlights

1. **Transformer Architecture**: Relies on self-attention mechanisms for capturing long-range dependencies and contextual relationships
2. **Transfer Learning**: Leverages pre-trained DistilBERT for strong text understanding
3. **Discriminative Fine-Tuning**: Different learning rates for BERT (2e-5) and classification head (1e-3)
4. **Selective Layer Freezing**: Freezes lower BERT layers (general linguistic features), fine-tunes upper layers (task-specific phishing patterns)
5. **Memory Efficient**: DistilBERT has 40% fewer parameters than BERT while retaining 97% performance
6. **Mixed Precision Training**: FP16 reduces memory usage and speeds up training
7. **Multi-Head Attention**: 12 attention heads enable the model to focus on different linguistic patterns simultaneously

## Use Cases

This model is optimized for detecting phishing in email and SMS messages by identifying:
- **Social Engineering Tactics**: Urgency, fear, authority impersonation
- **Suspicious Keywords**: "verify account", "urgent action", "click here"
- **Deceptive Language Patterns**: Grammatical errors, unusual phrasing
- **Phishing Indicators**: Requests for personal information, suspicious links
- **Impersonation**: Brand impersonation, executive impersonation (BEC)
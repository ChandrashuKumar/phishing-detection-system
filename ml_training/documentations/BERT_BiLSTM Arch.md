# DistilBERT-BiLSTM Model Architecture

## Overview

A transformer-based deep learning model for phishing email/SMS detection that combines DistilBERT (a distilled version of BERT) with Bidirectional LSTM networks to analyze text content for phishing indicators.

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

**Purpose**: Extracts contextual embeddings from input text, capturing semantic meaning and linguistic patterns indicative of phishing attempts.

**Output**: (batch_size, 128, 768) - Contextualized token representations

### Sequential Processing

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| BiLSTM | LSTM | (batch, 128, 768) → (batch, 128, 256) | hidden_size=128, num_layers=1, bidirectional=True, dropout=0.3 |

**Purpose**: Captures sequential dependencies and long-range patterns in the contextualized embeddings from DistilBERT.

**Output**:
- LSTM output: (batch, 128, 256)
- Hidden states: (2, batch, 128) - Forward and backward states

### Classification Head

| Layer | Type | Input → Output | Activation | Dropout |
|-------|------|----------------|------------|---------|
| Concatenation | Concat | 2 × (batch, 128) → (batch, 256) | - | - |
| Dropout | Dropout | (batch, 256) → (batch, 256) | - | 0.3 |
| FC | Linear | (batch, 256) → (batch, 1) | None (logits) | - |

**Purpose**: Combines forward and backward LSTM hidden states and produces final classification logit.

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
Bidirectional LSTM (batch, 128, 256)
    ↓
Extract Final Hidden States
Forward: (batch, 128)
Backward: (batch, 128)
    ↓
Concatenate (batch, 256)
    ↓
Dropout (0.3)
    ↓
Linear Classifier (batch, 1)
    ↓
Output Logit
```

## Training Configuration

### Optimizer
- **Type**: AdamW (with weight decay)
- **Learning Rates**:
  - DistilBERT layers: 2e-5 (discriminative fine-tuning)
  - LSTM layers: 1e-3
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
- **Dropout**: 0.3 (after BiLSTM and before classifier)
- **Layer Freezing**: First 50% of DistilBERT layers frozen
- **Weight Decay**: 0.01

## Model Statistics

- **Total Parameters**: 67,282,689
- **Trainable Parameters**: 22,183,425 (33.0%)
- **Frozen Parameters**: 45,099,264 (67.0%)
- **DistilBERT Parameters**: ~66.9M
- **LSTM Parameters**: ~329K
- **Classifier Parameters**: ~513

## Architecture Highlights

1. **Transfer Learning**: Leverages pre-trained DistilBERT for strong text understanding
2. **Discriminative Fine-Tuning**: Different learning rates for BERT (2e-5) and LSTM (1e-3) layers
3. **Selective Layer Freezing**: Freezes lower BERT layers (general features), fine-tunes upper layers (task-specific)
4. **Sequential Modeling**: BiLSTM captures temporal patterns in contextualized embeddings
5. **Memory Efficient**: DistilBERT has 40% fewer parameters than BERT while retaining 97% performance
6. **Mixed Precision Training**: FP16 reduces memory usage and speeds up training

## Use Cases

This model is optimized for detecting phishing in email and SMS messages by identifying:
- **Social Engineering Tactics**: Urgency, fear, authority impersonation
- **Suspicious Keywords**: "verify account", "urgent action", "click here"
- **Deceptive Language Patterns**: Grammatical errors, unusual phrasing
- **Phishing Indicators**: Requests for personal information, suspicious links
- **Impersonation**: Brand impersonation, executive impersonation (BEC)






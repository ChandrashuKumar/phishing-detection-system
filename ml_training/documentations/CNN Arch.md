# Phishing Detection CNN Model Architecture

## Overview

A Convolutional Neural Network (CNN) model for phishing URL detection using character-level analysis with parallel multi-scale convolutions and attention mechanisms.

## Model Architecture

### Input Processing

- **Input Type**: Character-level URL sequences
- **Maximum Sequence Length**: 200 characters
- **Input Shape**: (batch_size, 200)

### Embedding Layer

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Embedding | Embedding | (batch, 200) → (batch, 200, 128) | vocab_size=225, embedding_dim=128, padding_idx=0 |

**Purpose**: Converts character indices to dense 128-dimensional vectors.

### Parallel Convolutional Layers

The model uses three parallel 1D convolutions with different kernel sizes to capture multi-scale patterns:

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Conv3 | Conv1D | (batch, 128, 200) → (batch, 256, 200) | kernel_size=3, padding=1 |
| Conv5 | Conv1D | (batch, 128, 200) → (batch, 256, 200) | kernel_size=5, padding=2 |
| Conv7 | Conv1D | (batch, 128, 200) → (batch, 256, 200) | kernel_size=7, padding=3 |

**Purpose**: 
- **Conv3**: Detects short character patterns (trigrams)
- **Conv5**: Detects medium patterns (5-grams)
- **Conv7**: Detects longer patterns (7-grams)

### Feature Fusion

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Concatenation | Concat | 3 × (batch, 256, 200) → (batch, 768, 200) | Combines parallel conv outputs |
| Dropout | Dropout | (batch, 768, 200) → (batch, 768, 200) | p=0.5 |
| Conv Combine | Conv1D | (batch, 768, 200) → (batch, 512, 200) | kernel_size=3, padding=1 |

**Purpose**: Merges multi-scale features into a unified 512-dimensional representation.

### Attention Mechanism

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Attention | Linear + Softmax | (batch, 200, 512) → (batch, 200, 512) | 512 → 1 weights |
| Max Pooling | Global Max Pool | (batch, 200, 512) → (batch, 512) | Aggregates sequence |

**Purpose**: Identifies and emphasizes the most discriminative character patterns while suppressing irrelevant features.

### Classification Head

| Layer | Type | Input → Output | Activation | Dropout |
|-------|------|----------------|------------|---------|
| FC1 | Linear | (batch, 512) → (batch, 256) | ReLU | 0.5 |
| FC2 | Linear | (batch, 256) → (batch, 128) | ReLU | 0.3 |
| FC3 | Linear | (batch, 128) → (batch, 1) | None (logits) | - |

**Output**: Raw logit score (converted to probability via sigmoid during inference)

## Training Configuration

### Optimizer
- **Type**: Adam
- **Learning Rate**: 0.001
- **Scheduler**: ReduceLROnPlateau
  - Mode: max
  - Factor: 0.5
  - Patience: 2

### Loss Function
- **Type**: BCEWithLogitsLoss

### Class Imbalance Handling
- **Method**: WeightedRandomSampler
- **Purpose**: Oversamples minority class (phishing) during training to achieve balanced batches
- **Effect**: Phishing samples are seen ~2.13× more often per epoch

### Regularization
- **Dropout Rates**: 
  - Post-concatenation: 0.5
  - After FC1: 0.5
  - After FC2: 0.3

## Model Statistics

- **Total Parameters**: 1,866,114
- **Trainable Parameters**: 1,866,114
- **Character Vocabulary Size**: 225
- **Embedding Dimension**: 128
- **Convolution Filters**: 256 per kernel size
- **Combined Features**: 512

## Architecture Highlights

1. **Multi-Scale Pattern Detection**: Parallel convolutions with kernel sizes 3, 5, and 7 capture patterns at different granularities
2. **Attention Mechanism**: Automatically learns to focus on suspicious character sequences
3. **Deep Feature Extraction**: Four convolutional layers progressively extract higher-level features
4. **Heavy Regularization**: Multiple dropout layers (0.3-0.5) prevent overfitting
5. **Class Imbalance Handling**: WeightedRandomSampler ensures balanced training batches

## Data Flow Summary
```
Input (batch, 200)
    ↓
Embedding (batch, 200, 128)
    ↓
┌───────────┬───────────┬───────────┐
Conv3       Conv5       Conv7
(256)       (256)       (256)
└───────────┴───────────┴───────────┘
    ↓
Concatenate (batch, 768, 200)
    ↓
Dropout (0.5)
    ↓
Conv Combine (batch, 512, 200)
    ↓
Attention (batch, 200, 512)
    ↓
Max Pooling (batch, 512)
    ↓
FC1 → Dropout → FC2 → Dropout → FC3
    ↓
Output (batch, 1)
```

## Use Case

This model is optimized for detecting phishing URLs by analyzing character-level patterns such as:
- Suspicious top-level domains (e.g., `.tk`, `.ml`)
- Typosquatting (e.g., `paypa1` instead of `paypal`)
- URL structure anomalies
- Obfuscation techniques
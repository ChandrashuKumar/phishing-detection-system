# Dual CNN with Cross-Attention Model Architecture

## Overview

A multi-modal deep learning model for phishing detection that processes both URL (character-level) and HTML (word-level) content using parallel CNNs with self-attention and cross-attention mechanisms for URL↔HTML interaction.

## Model Architecture

### Input Processing

**URL Branch (Character-level)**
- **Input Type**: Character-level URL sequences
- **Maximum Sequence Length**: 200 characters
- **Vocabulary Size**: 225 characters
- **Input Shape**: (batch_size, 200)

**HTML Branch (Word-level)**
- **Input Type**: Word-level HTML tokens
- **Maximum Sequence Length**: 500 words
- **Vocabulary Size**: 10,000 words
- **Input Shape**: (batch_size, 500)

### URL Branch Architecture

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Embedding | Embedding | (batch, 200) → (batch, 200, 128) | vocab_size=225, embedding_dim=128, padding_idx=0 |
| Conv3 | Conv1D | (batch, 128, 200) → (batch, 256, 200) | kernel_size=3, padding=1 |
| Conv5 | Conv1D | (batch, 128, 200) → (batch, 256, 200) | kernel_size=5, padding=2 |
| Conv7 | Conv1D | (batch, 128, 200) → (batch, 256, 200) | kernel_size=7, padding=3 |
| Concatenation | Concat | 3 × (batch, 256, 200) → (batch, 768, 200) | Combines parallel convs |
| Conv Combine | Conv1D | (batch, 768, 200) → (batch, 128, 200) | kernel_size=3, padding=1 |
| Self-Attention | MultiheadAttention | (batch, 200, 128) → (batch, 200, 128) | num_heads=8, embed_dim=128 |
| Residual | Add | (batch, 200, 128) | Residual connection |

**Purpose**: Extracts multi-scale character patterns from URLs and applies self-attention to capture long-range dependencies.

### HTML Branch Architecture

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Embedding | Embedding | (batch, 500) → (batch, 500, 128) | vocab_size=10000, embedding_dim=128, padding_idx=0 |
| Conv3 | Conv1D | (batch, 128, 500) → (batch, 256, 500) | kernel_size=3, padding=1 |
| Conv5 | Conv1D | (batch, 128, 500) → (batch, 256, 500) | kernel_size=5, padding=2 |
| Conv7 | Conv1D | (batch, 128, 500) → (batch, 256, 500) | kernel_size=7, padding=3 |
| Concatenation | Concat | 3 × (batch, 256, 500) → (batch, 768, 500) | Combines parallel convs |
| Conv Combine | Conv1D | (batch, 768, 500) → (batch, 128, 500) | kernel_size=3, padding=1 |
| Self-Attention | MultiheadAttention | (batch, 500, 128) → (batch, 500, 128) | num_heads=8, embed_dim=128 |
| Residual | Add | (batch, 500, 128) | Residual connection |

**Purpose**: Extracts semantic patterns from HTML content and applies self-attention to identify important page elements.

### Cross-Attention Fusion

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| URL→HTML Cross-Attention | MultiheadAttention | (batch, 200, 128) → (batch, 200, 128) | num_heads=4, embed_dim=128 |
| URL Residual | Add | (batch, 200, 128) | Residual connection |
| HTML→URL Cross-Attention | MultiheadAttention | (batch, 500, 128) → (batch, 500, 128) | num_heads=4, embed_dim=128 |
| HTML Residual | Add | (batch, 500, 128) | Residual connection |

**Purpose**: Enables bidirectional interaction between URL and HTML features. URL attends to HTML to learn which page content is relevant, and HTML attends to URL to contextualize content based on domain.

### Feature Aggregation & Classification

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| URL Max Pooling | Global Max Pool | (batch, 200, 128) → (batch, 128) | Aggregates URL features |
| HTML Max Pooling | Global Max Pool | (batch, 500, 128) → (batch, 128) | Aggregates HTML features |
| Concatenation | Concat | 2 × (batch, 128) → (batch, 256) | Fuses both modalities |
| FC1 | Linear | (batch, 256) → (batch, 128) | ReLU + Dropout(0.3) |
| FC2 | Linear | (batch, 128) → (batch, 64) | ReLU + Dropout(0.2) |
| FC3 | Linear | (batch, 64) → (batch, 1) | Sigmoid (inference) |

**Output**: Phishing probability score (0-1)

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
  - After FC1: 0.3
  - After FC2: 0.2
- **Residual Connections**: Applied after self-attention and cross-attention layers

## Model Statistics

- **Total Parameters**: 3,188,865
- **Trainable Parameters**: 3,188,865
- **URL Vocabulary Size**: 225 characters
- **HTML Vocabulary Size**: 10,000 words
- **Embedding Dimension**: 128
- **Self-Attention Heads**: 8
- **Cross-Attention Heads**: 4

## Architecture Highlights

1. **Multi-Modal Learning**: Processes both URL and HTML content independently before fusion
2. **Multi-Scale CNNs**: Parallel convolutions (kernel sizes 3, 5, 7) capture patterns at different granularities
3. **Self-Attention**: 8-head attention in each branch captures long-range dependencies
4. **Cross-Attention**: 4-head bidirectional attention enables URL↔HTML interaction
5. **Residual Connections**: Stabilizes training and enables gradient flow
6. **Class Imbalance Handling**: WeightedRandomSampler ensures balanced training batches

## Data Flow Summary
```
URL Input (batch, 200)                  HTML Input (batch, 500)
    ↓                                        ↓
Char Embedding (batch, 200, 128)        Word Embedding (batch, 500, 128)
    ↓                                        ↓
┌────────┬────────┬────────┐           ┌────────┬────────┬────────┐
Conv3    Conv5    Conv7                Conv3    Conv5    Conv7
(256)    (256)    (256)                (256)    (256)    (256)
└────────┴────────┴────────┘           └────────┴────────┴────────┘
    ↓                                        ↓
Concat (batch, 768, 200)                Concat (batch, 768, 500)
    ↓                                        ↓
Conv Combine (batch, 128, 200)          Conv Combine (batch, 128, 500)
    ↓                                        ↓
Self-Attention (8 heads)                Self-Attention (8 heads)
+ Residual                              + Residual
    ↓                                        ↓
    └──────── Cross-Attention (4 heads) ─────┘
              (Bidirectional)
                ↓         ↓
            URL Features  HTML Features
             (batch, 128)  (batch, 128)
                ↓         ↓
            Concatenate (batch, 256)
                    ↓
        FC1 → Dropout → FC2 → Dropout → FC3
                    ↓
            Output (batch, 1)
```

## Use Case

This model is optimized for detecting phishing websites by analyzing both URL and HTML content:

**URL Analysis**:
- Suspicious top-level domains (e.g., `.tk`, `.ml`)
- Typosquatting (e.g., `paypa1` instead of `paypal`)
- URL structure anomalies
- Obfuscation techniques

**HTML Analysis**:
- Login form patterns
- Branding inconsistencies
- Suspicious links and redirects
- Page structure anomalies

**Cross-Modal Interaction**:
- Detecting mismatch between URL domain and page content
- Identifying legitimate-looking pages on suspicious domains
- Recognizing phishing patterns that span both URL and content

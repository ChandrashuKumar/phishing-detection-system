# CNN-LSTM Hybrid Model Architecture

## Overview

A dual-branch deep learning model for phishing detection that combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to analyze both URL and HTML content.

## Model Architecture

### Input Processing

The model processes two types of input simultaneously:
- **URL Input**: Character-level sequences (max length: 200)
- **HTML Input**: Word-level sequences (max length: 2000)

### Branch 1: URL Processing (Character-Level)

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Embedding | Embedding | (batch, 200) → (batch, 200, 16) | vocab_size=225, embedding_dim=16 |
| Conv1D-1 | 1D Convolution | (batch, 16, 200) → (batch, 256, 200) | kernel_size=3, padding=1 |
| Conv1D-2 | 1D Convolution | (batch, 256, 200) → (batch, 256, 200) | kernel_size=5, padding=2 |
| Conv1D-3 | 1D Convolution | (batch, 256, 200) → (batch, 128, 200) | kernel_size=7, padding=3 |
| Attention | Linear + Softmax | (batch, 200, 128) → (batch, 200, 128) | 128 → 1 weights |
| Max Pooling | Global Max Pool | (batch, 200, 128) → (batch, 128) | Aggregates sequence |

**Purpose**: Detects character-level patterns such as suspicious TLDs (`.tk`), typosquatting, and URL structure anomalies.

### Branch 2: HTML Processing (Word-Level)

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Embedding | Embedding | (batch, 2000) → (batch, 2000, 16) | vocab_size=10000, embedding_dim=16 |
| Conv1D-1 | 1D Convolution | (batch, 16, 2000) → (batch, 256, 2000) | kernel_size=3, padding=1 |
| Conv1D-2 | 1D Convolution | (batch, 256, 2000) → (batch, 256, 2000) | kernel_size=5, padding=2 |
| Conv1D-3 | 1D Convolution | (batch, 256, 2000) → (batch, 128, 2000) | kernel_size=7, padding=3 |
| Attention | Linear + Softmax | (batch, 2000, 128) → (batch, 2000, 128) | 128 → 1 weights |
| Max Pooling | Global Max Pool | (batch, 2000, 128) → (batch, 128) | Aggregates sequence |

**Purpose**: Identifies word-level phishing indicators such as urgency keywords ("verify", "urgent"), login forms, and social engineering tactics.

### Feature Fusion

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Concatenation | Concat | (batch, 128) + (batch, 128) → (batch, 256) | Combines URL and HTML features |
| Reshape | Unsqueeze | (batch, 256) → (batch, 1, 256) | Prepares for LSTM |

### Sequential Processing

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Bi-LSTM | LSTM | (batch, 1, 256) → (batch, 1, 256) | hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3 |

**Purpose**: Learns contextual relationships between URL and HTML features to identify sophisticated phishing patterns.

### Classification Head

| Layer | Type | Input → Output | Activation | Dropout |
|-------|------|----------------|------------|---------|
| FC1 | Linear | (batch, 256) → (batch, 128) | ReLU | 0.3 |
| FC2 | Linear | (batch, 128) → (batch, 64) | ReLU | 0.2 |
| FC3 | Linear | (batch, 64) → (batch, 1) | None (logits) | - |

**Output**: Raw logit score (converted to probability via sigmoid during inference)

## Data Flow Summary
```
URL Input (batch, 200)                    HTML Input (batch, 2000)
    ↓                                         ↓
URL Embedding (batch, 200, 16)            HTML Embedding (batch, 2000, 16)
    ↓                                         ↓
Transpose (batch, 16, 200)                Transpose (batch, 16, 2000)
    ↓                                         ↓
URL Conv1 (batch, 256, 200)               HTML Conv1 (batch, 256, 2000)
    ↓                                         ↓
URL Conv2 (batch, 256, 200)               HTML Conv2 (batch, 256, 2000)
    ↓                                         ↓
URL Conv3 (batch, 128, 200)               HTML Conv3 (batch, 128, 2000)
    ↓                                         ↓
Transpose (batch, 200, 128)               Transpose (batch, 2000, 128)
    ↓                                         ↓
URL Attention (batch, 200, 128)           HTML Attention (batch, 2000, 128)
    ↓                                         ↓
URL Max Pool (batch, 128)                 HTML Max Pool (batch, 128)
    ↓                                         ↓
    └─────────────┬───────────────────────────┘
                  ↓
          Concatenate (batch, 256)
                  ↓
          Unsqueeze (batch, 1, 256)
                  ↓
      Bidirectional LSTM (batch, 1, 256)
                  ↓
          Squeeze (batch, 256)
                  ↓
      FC1 → Dropout → FC2 → Dropout → FC3
                  ↓
          Output (batch, 1)
```

## Training Configuration

### Optimizer
- **Type**: Adam
- **Learning Rate**: 0.001
- **Scheduler**: ReduceLROnPlateau (mode='max', factor=0.5, patience=2)

### Loss Function
- **Type**: BCEWithLogitsLoss
- **Positive Weight**: 2.13 (addresses class imbalance)

### Regularization
- Dropout layers: 0.3 (after FC1), 0.2 (after FC2)
- LSTM dropout: 0.3 (between layers)

## Model Statistics

- **Total Parameters**: 2,135,571
- **Trainable Parameters**: 2,135,571
- **Vocabulary Sizes**:
  - URL (characters): 225
  - HTML (words): 10,000
- **Embedding Dimensions**: 16 for both branches
- **Feature Dimensions**:
  - After CNN: 128 per branch
  - Combined: 256
  - LSTM output: 256 (bidirectional)

## Key Features

1. **Dual-Branch Architecture**: Separate processing pipelines for URL and HTML content
2. **Multi-Scale Pattern Detection**: Multiple kernel sizes (3, 5, 7) capture n-grams of varying lengths
3. **Attention Mechanism**: Focuses on important features while suppressing noise
4. **Bidirectional LSTM**: Captures sequential dependencies in combined features
5. **Class Imbalance Handling**: Weighted loss function prioritizes phishing detection
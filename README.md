# Phishing Detection System

Machine learning system for detecting phishing attacks through URL analysis and email/SMS content classification.

## Overview

This project implements multiple models for phishing detection:

**URL Detection** (Phishing Websites)
- Random Forest, XGBoost (feature-based)
- CNN (character-level URL analysis)
- CNN-BiLSTM Hybrid (URL + HTML content)

**Email/SMS Detection** (Phishing Messages)
- DistilBERT (transformer-based text classification)

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/phishing-detection-system.git
cd phishing-detection-system
```

### 2. Install Dependencies
```bash
# Using Conda (Recommended)
conda env create -f environment.yml
conda activate phishing-detection-ml

# OR using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Datasets

#### Dataset 1: PhreshPhish (URL Detection)
This dataset is automatically downloaded from Hugging Face when running the notebooks.

```python
from datasets import load_dataset
dataset = load_dataset("phreshphish/phreshphish", cache_dir='path/to/cache')
```

**Dataset Info:**
- Source: `phreshphish/phreshphish` (Hugging Face)
- Size: ~408K URLs with HTML content
- Features: url, label (benign/phish), html, date, language, target
- Used in: `ml_training/notebooks/url-detection/`

#### Dataset 2: Seven Emails Phishing Dataset (Email/SMS Detection)
**Download:** [Seven Phishing Email Datasets on Figshare](https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108)

Download and extract to `data/unprocessed/email-detection/`.

**Dataset Info:**
- Source: Seven combined email phishing datasets
- Size: ~203K emails (after cleaning)
- Features: sender, receiver, subject, body, label, date
- Split: 70% train, 15% validation, 15% test
- Used in: `ml_training/notebooks/email-sms-detection/`

## Model Performance

### URL Detection Models (Test Set)

| Model | Threshold | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|----------|-----------|--------|----------|
| **XGBoost** | 0.40 | 95.85% | 88.47% | 88.94% | 88.71% |
| **CNN** | 0.60 | 96.03% | 88.29% | 90.28% | 89.27% |
| **CNN-BiLSTM** | 0.31 | 96.85% | 92.53% | 90.09% | 91.29% |

### Email/SMS Detection Model (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **DistilBERT** | 99.17% | 99.23% | 98.97% | 99.10% |

*See `results/` directory for detailed confusion matrices and ROC curves.*

## License

MIT License


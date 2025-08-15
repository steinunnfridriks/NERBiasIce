# UNDER CONSTRUCTION

# NERBiasIce - Named Entity Recognition for Bias Detection in Icelandic

This project implements a Named Entity Recognition (NER) system for detecting and classifying bias-related language in Icelandic text. The system uses transformer-based models (IceBERT/ScandiBERT) fine-tuned on annotated Icelandic text data.

## Project Overview

The system identifies and classifies potentially biased language into 14 categories:
- ADDICTION - Addiction-related bias
- DISABILITY - Disability-related bias  
- GENERAL - General bias terms
- LGBTQIA - LGBTQIA+ related bias
- LOOKS - Appearance-based bias
- ORIGIN - Ethnicity/nationality bias
- PERSONAL - Personality trait bias
- PROFANITY - Profane language
- RELIGION - Religious bias
- SEXUAL - Sexual content bias
- SOCIAL_STATUS - Social class bias
- STUPIDITY - Intelligence-based bias
- VULGAR - Vulgar language
- WOMEN - Gender-based bias

## Quick Start

1. **Set up environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Evaluate the pre-trained model:**
   ```bash
   python train_model.py --eval-only --model-dir icebert
   ```

3. **Generate performance visualization:**
   ```bash
   python plot.py
   ```

## Scripts Documentation

### Core Training/Evaluation Scripts

- **train_model.py** - Advanced training script with class weighting for imbalanced data (ScandiBERT-based)
- **train_model.py** - Basic training script using IceBERT
- **test_model.py** - Interactive testing script for single text samples

### Data Processing Scripts

- **balance_classes.py** - Balances training data by reducing overrepresented bias categories
- **split_data.py** - Splits annotated data into train/dev/test sets (80/10/10)
- **collect_non_o.py** - Filters sentences containing at least one bias label

### Labeling Scripts

- **biolabeler.py** - Automatically labels text with BIO tags based on bias vocabulary
- **rmh_extractor.py** - Extracts and processes text from RMH corpus XML files

### Visualization

- **plot.py** - Generates performance heatmaps with confidence intervals

## Data Format

The system uses CoNLL-style format with tab-separated values:
```
word1	TAG1
word2	TAG2

word1	TAG1
```

Tags follow BIO notation:
- B-CATEGORY: Beginning of bias entity
- I-CATEGORY: Inside bias entity
- O: Outside any bias entity

## Model Performance

The pre-trained IceBERT model achieves:
- Macro F1: 85.1%
- Micro F1: 87.5%
- Per-category F1 ranges from 66.7% (ORIGIN) to 91.0% (ADDICTION/PROFANITY)

## Directory Structure

```
.
├── data/               # Training/evaluation data (gitignored)
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
├── icebert/           # Pre-trained model files (gitignored)
│   ├── model.safetensors
│   ├── config.json
│   └── ...
└── *.py              # Processing scripts
```

## Citation

If you use this work, please cite:
```
[Your citation information here]
```

## License

[Your license information here]

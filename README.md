# Intent Classification: Baseline vs. Fine-Tuned Transformer

Comparing a TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT on the [Banking77](https://huggingface.co/datasets/PolyAI/banking77) intent classification dataset (77 intents, ~13K examples).

## Problem

Given a customer banking query (e.g., *"I need to top up my account"*), classify it into one of 77 intent categories. This project asks: **how much does fine-tuning a small transformer improve over a simple, strong baseline — and where does each fail?**

## Dataset

**Banking77** — 10,003 training + 3,080 test customer-support queries across 77 banking intents.

- Training set is split 85/15 into train/validation (stratified).
- Test set is held out and evaluated only once at the end.

## Methods

| Model | Description |
|-------|-------------|
| **Baseline** | TF-IDF (word 1-2-grams, sublinear TF) → Logistic Regression. Hyperparameters (C, n-gram range) tuned on validation. |
| **DistilBERT** | `distilbert-base-uncased` fine-tuned with AdamW + linear warmup. Hand-written PyTorch training loop. Evaluated with 3 random seeds. |

## Results

*Fill in after running the experiments.*

| Model | Test Accuracy | Test Macro-F1 |
|-------|:---:|:---:|
| TF-IDF + LogReg | [X%] | [Y%] |
| DistilBERT (mean ± std, 3 seeds) | [X ± σ%] | [Y ± σ%] |

### Learning Curve

*Plot saved to `results/learning_curve.png` after running `python src/analyze.py`.*

Both models are trained on 10%, 25%, 50%, and 100% of the training data to show when the transformer's advantage appears.

### Error Analysis

*Summary saved to `results/error_analysis.json` after running `python src/analyze.py`.*

Key questions answered:
- Which intents are hardest for each model?
- Which intent pairs get confused most often, and why?
- Does the transformer fix the baseline's mistakes, or make different ones?

## Reproducibility

- **Baseline seed:** 42 (for splits and logistic regression).
- **Transformer seeds:** 42, 123, 456 (3 runs, mean ± std reported).
- All library versions pinned in `requirements.txt`.

## How to Run

### Prerequisites

- Python 3.9+
- GPU recommended for fine-tuning (free Google Colab or Kaggle works). CPU is fine for the baseline.

### Setup

```bash
git clone <repo-url>
cd intent-classification
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step-by-step

```bash
# 1. Explore the dataset — prints class counts, query lengths, sample examples
python src/data.py

# 2. Run the baseline — sweeps hyperparameters, evaluates on test
python src/baseline.py

# 3. Fine-tune DistilBERT — trains 3 seeds, evaluates on test
python src/finetune.py

# 4. Error analysis + learning curve — needs both models' predictions
python src/analyze.py
```

All results (metrics, plots, CSVs) are saved to the `results/` directory.

### Custom options

```bash
# Fine-tune with different settings
python src/finetune.py --epochs 3 --lr 2e-5 --batch_size 16

# Use different seeds
python src/finetune.py --seeds 1 2 3
```

## Project Structure

```
intent-classification/
├── README.md              # This file
├── requirements.txt       # Pinned dependencies
├── .gitignore             # Excludes checkpoints, caches, secrets
├── src/
│   ├── data.py            # Load Banking77, create train/val/test splits
│   ├── baseline.py        # TF-IDF + Logistic Regression baseline
│   ├── finetune.py        # DistilBERT fine-tuning (hand-written PyTorch loop)
│   └── analyze.py         # Error analysis + learning-curve experiment
├── notebooks/             # Exploration notebooks (optional)
└── results/               # Metrics JSON/CSV, plots (auto-generated)
```

## Limitations

- Single dataset (Banking77) — results may not generalize to other intent taxonomies.
- One small model (DistilBERT) — larger models would likely perform better.
- Limited hyperparameter search — only learning rate and epochs tuned for the transformer.
- English only.
- No out-of-scope detection (Banking77 has no OOS class).

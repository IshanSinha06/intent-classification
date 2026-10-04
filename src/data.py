"""
data.py — Load Banking77 dataset, explore it, and create train/validation/test splits.

This module is the data foundation for the entire project. It is called by:
  - src/baseline.py   : to get train/val/test texts and labels for the TF-IDF baseline.
  - src/finetune.py   : to get train/val/test texts and labels for DistilBERT fine-tuning.
  - src/analyze.py    : to get label names and test data for error analysis and learning curves.

When run directly (`python src/data.py`), it prints dataset statistics and sample examples
so you can visually inspect the data before training.
"""

import os
import json
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Random seed for reproducibility across all splits.
# Used here and importable by other modules for consistency.
RANDOM_SEED = 42

# Fraction of the original training set held out for validation.
VALIDATION_SIZE = 0.15

# Directory where results (metrics, plots, error tables) are saved.
# Used by baseline.py, finetune.py, and analyze.py when writing output.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_banking77():
    """
    Load the Banking77 dataset from the original GitHub CSVs.

    The PolyAI/banking77 HuggingFace repo uses a legacy loading script that
    datasets 3.x+ no longer supports, so we load the CSVs directly from the
    PolyAI GitHub repository instead.

    The CSV columns are 'text' (query string) and 'category' (intent name string).
    We convert the category strings to integer label ids using a sorted label list.

    Returns:
        dict with keys:
            "train_texts"  : List[str]  — raw training query strings.
            "train_labels" : List[int]  — integer label ids for training queries.
            "test_texts"   : List[str]  — raw test query strings.
            "test_labels"  : List[int]  — integer label ids for test queries.
            "label_names"  : List[str]  — 77 intent names, sorted alphabetically.

    Called by:
      - get_splits() in this file to build train/val/test data.
      - Can be called standalone for quick exploration.
    """
    _TRAIN_URL = (
        "https://raw.githubusercontent.com/PolyAI-LDN/"
        "task-specific-datasets/master/banking_data/train.csv"
    )
    _TEST_URL = (
        "https://raw.githubusercontent.com/PolyAI-LDN/"
        "task-specific-datasets/master/banking_data/test.csv"
    )

    dataset = load_dataset("csv", data_files={"train": _TRAIN_URL, "test": _TEST_URL})

    # Build a sorted label vocabulary and map category strings to integer ids.
    all_categories = sorted(set(list(dataset["train"]["category"]) + list(dataset["test"]["category"])))
    cat_to_id = {cat: i for i, cat in enumerate(all_categories)}

    return {
        "train_texts": list(dataset["train"]["text"]),
        "train_labels": [cat_to_id[c] for c in dataset["train"]["category"]],
        "test_texts": list(dataset["test"]["text"]),
        "test_labels": [cat_to_id[c] for c in dataset["test"]["category"]],
        "label_names": all_categories,
    }


def get_splits(seed=RANDOM_SEED):
    """
    Load Banking77 and split into train, validation, and test sets.

    The original training set is split 85/15 into train/validation (stratified).
    The original test set is kept untouched for final evaluation only.

    Args:
        seed (int): Random seed for the train/val split. Default is RANDOM_SEED (42).

    Returns:
        dict with keys:
            "train_texts"  : List[str]  — training queries
            "train_labels" : List[int]  — training intent label ids
            "val_texts"    : List[str]  — validation queries
            "val_labels"   : List[int]  — validation intent label ids
            "test_texts"   : List[str]  — test queries (untouched until final eval)
            "test_labels"  : List[int]  — test intent label ids
            "label_names"  : List[str]  — 77 human-readable intent names

    Called by:
      - src/baseline.py   : to train and evaluate the TF-IDF + LogReg baseline.
      - src/finetune.py   : to train and evaluate DistilBERT.
      - src/analyze.py    : to run error analysis and the learning-curve experiment.
    """
    raw = load_banking77()
    label_names = raw["label_names"]

    all_train_texts = raw["train_texts"]
    all_train_labels = raw["train_labels"]
    test_texts = raw["test_texts"]
    test_labels = raw["test_labels"]

    # Stratified split: keeps class proportions consistent in train and val.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_train_texts,
        all_train_labels,
        test_size=VALIDATION_SIZE,
        stratify=all_train_labels,
        random_state=seed,
    )

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "label_names": label_names,
    }


def get_partial_splits(fraction, seed=RANDOM_SEED):
    """
    Get splits with only a fraction of the training data (for learning-curve experiments).

    Uses stratified sampling to take `fraction` of the training set while keeping
    class proportions intact. Validation and test sets remain full-sized.

    Args:
        fraction (float): Proportion of training data to keep, e.g. 0.1 for 10%.
        seed (int): Random seed for reproducibility.

    Returns:
        Same dict format as get_splits(), but with a smaller training set.

    Called by:
      - src/analyze.py : in the learning_curve_experiment() function, to train
                         both baseline and transformer on 10%, 25%, 50%, 100% of data.
    """
    splits = get_splits(seed=seed)

    if fraction >= 1.0:
        return splits

    # Take a stratified subsample of the training data.
    train_texts, _, train_labels, _ = train_test_split(
        splits["train_texts"],
        splits["train_labels"],
        train_size=fraction,
        stratify=splits["train_labels"],
        random_state=seed,
    )

    splits["train_texts"] = train_texts
    splits["train_labels"] = train_labels
    return splits


def print_dataset_stats(splits):
    """
    Print summary statistics about the dataset splits.

    Shows split sizes, class count, query-length distribution, class balance,
    and sample examples. Useful for initial data exploration (Step 1 of project).

    Args:
        splits: The dict returned by get_splits().

    Called by:
      - __main__ block at the bottom of this file (when run directly).
    """
    label_names = splits["label_names"]
    num_classes = len(label_names)

    print("=" * 60)
    print("Banking77 Dataset Summary")
    print("=" * 60)

    # --- Split sizes ---
    print(f"\nNumber of intents: {num_classes}")
    print(f"Train size:        {len(splits['train_texts'])}")
    print(f"Validation size:   {len(splits['val_texts'])}")
    print(f"Test size:         {len(splits['test_texts'])}")

    # --- Query length distribution ---
    all_texts = splits["train_texts"] + splits["val_texts"]
    lengths = [len(t.split()) for t in all_texts]
    print(f"\nQuery length (words): min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")

    # --- Class balance ---
    train_labels = np.array(splits["train_labels"])
    counts = np.bincount(train_labels, minlength=num_classes)
    print(f"\nSamples per class: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, std={counts.std():.1f}")

    # --- Show the 5 smallest classes (potential trouble spots) ---
    smallest_idx = np.argsort(counts)[:5]
    print("\n5 smallest classes:")
    for idx in smallest_idx:
        print(f"  [{idx:2d}] {label_names[idx]:40s} — {counts[idx]} samples")

    # --- Sample examples from a few classes ---
    print("\n" + "-" * 60)
    print("Sample examples (3 per class, first 5 classes):")
    print("-" * 60)
    sample_classes = sorted(set(splits["train_labels"]))[:5]
    for cls_id in sample_classes:
        print(f"\n  Intent: {label_names[cls_id]}")
        examples = [t for t, l in zip(splits["train_texts"], splits["train_labels"])
                     if l == cls_id][:3]
        for ex in examples:
            print(f"    • {ex}")


def save_split_info(splits):
    """
    Save split sizes to a JSON file in the results directory for later reference.

    Args:
        splits: The dict returned by get_splits().

    Called by:
      - __main__ block at the bottom of this file (when run directly).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    info = {
        "num_classes": len(splits["label_names"]),
        "train_size": len(splits["train_texts"]),
        "val_size": len(splits["val_texts"]),
        "test_size": len(splits["test_texts"]),
        "val_fraction": VALIDATION_SIZE,
        "random_seed": RANDOM_SEED,
    }
    path = os.path.join(RESULTS_DIR, "split_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSplit info saved to {path}")


# ---------------------------------------------------------------------------
# Main — run `python src/data.py` to explore the dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading Banking77 dataset...")
    splits = get_splits()
    print_dataset_stats(splits)
    save_split_info(splits)

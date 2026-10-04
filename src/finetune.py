"""
finetune.py — Fine-tune DistilBERT on Banking77 for intent classification.

Implements a hand-written PyTorch training loop (no HF Trainer) with:
  - AdamW optimizer with linear warmup schedule.
  - Per-epoch validation with early checkpoint saving.
  - Support for multiple random seeds (default: 3 seeds for final evaluation).

Data source:
  - All data is loaded from src/data.py via get_splits().

Results are saved to:
  - results/finetune_log.json         — per-epoch metrics for every seed.
  - results/finetune_results.json     — final test-set metrics (mean ± std across seeds).
  - results/finetune_predictions.json — per-example test predictions from the last seed
                                        (used by src/analyze.py for error analysis).

Called by / used from:
  - Run directly: `python src/finetune.py`
  - src/analyze.py : imports train_and_evaluate() to retrain on partial data
                     for the learning-curve experiment.
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data import get_splits, RESULTS_DIR, RANDOM_SEED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DistilBERT model identifier on Hugging Face Hub.
MODEL_NAME = "distilbert-base-uncased"

# Max token length for tokenization. Banking77 queries are short (~10 words),
# so 64 tokens is plenty and keeps training fast.
MAX_LENGTH = 64

# Default training hyperparameters (tunable via CLI args).
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_EPOCHS = 5
DEFAULT_WARMUP_RATIO = 0.1

# Seeds for the 3 final runs, as required by the project plan.
EVAL_SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    """
    PyTorch Dataset wrapper for tokenized intent-classification examples.

    Takes raw texts and labels, tokenizes them with the DistilBERT tokenizer,
    and returns tensors suitable for the model's forward pass.

    Args:
        texts (List[str]): Raw query strings.
        labels (List[int]): Integer intent label ids.
        tokenizer: A Hugging Face tokenizer instance.
        max_length (int): Maximum token sequence length.

    Used by:
      - create_dataloaders() in this file, to build train/val/test DataLoaders.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        # Tokenize all texts at once for efficiency.
        # Returns a BatchEncoding with 'input_ids' and 'attention_mask'.
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """Return the number of examples. Called by DataLoader to determine batching."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return a single tokenized example as a dict of tensors.

        Called by:
          - PyTorch DataLoader during iteration to fetch individual examples
            before collating them into batches.
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def create_dataloaders(splits, tokenizer, batch_size=DEFAULT_BATCH_SIZE):
    """
    Build PyTorch DataLoaders for train, validation, and test sets.

    Args:
        splits (dict): Data dict from src/data.py get_splits().
        tokenizer: Hugging Face tokenizer for DistilBERT.
        batch_size (int): Batch size for all loaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader) — PyTorch DataLoaders.

    Called by:
      - train_and_evaluate() in this file, to get data loaders before training.
    """
    train_dataset = IntentDataset(splits["train_texts"], splits["train_labels"], tokenizer)
    val_dataset = IntentDataset(splits["val_texts"], splits["val_labels"], tokenizer)
    test_dataset = IntentDataset(splits["test_texts"], splits["test_labels"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def get_device():
    """
    Detect the best available device (CUDA GPU, Apple MPS, or CPU).

    Returns:
        torch.device: The device to run training on.

    Called by:
      - train_and_evaluate() in this file, at the start of each training run.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    """
    Run one training epoch: forward pass, loss computation, backpropagation.

    Args:
        model: The DistilBERT model (AutoModelForSequenceClassification).
        train_loader: DataLoader yielding batches of tokenized training examples.
        optimizer: AdamW optimizer.
        scheduler: Linear warmup learning rate scheduler.
        device: torch.device (cuda/mps/cpu).

    Returns:
        float: Average training loss for this epoch.

    Called by:
      - train_and_evaluate() in this file, once per epoch in the training loop.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="  Training", leave=False):
        # Move batch tensors to the training device (GPU/MPS/CPU).
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass — model returns loss and logits when labels are provided.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass — compute gradients and update weights.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on a dataset (validation or test) without gradient computation.

    Args:
        model: The trained DistilBERT model.
        data_loader: DataLoader for the evaluation set.
        device: torch.device.

    Returns:
        dict: {"accuracy": float, "macro_f1": float, "loss": float,
               "predictions": List[int], "true_labels": List[int]}

    Called by:
      - train_and_evaluate() in this file, after each epoch (on val) and at the end (on test).
      - src/analyze.py : could be imported for custom evaluation if needed.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="  Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

            # Get predicted class: argmax over the 77-class logit vector.
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "loss": total_loss / num_batches,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def train_and_evaluate(
    splits,
    seed=RANDOM_SEED,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    evaluate_test=True,
):
    """
    Full training pipeline for one seed: init model → train → validate → test.

    This is the core function. It:
      1. Sets the random seed for reproducibility.
      2. Loads the DistilBERT tokenizer and model.
      3. Tokenizes data and creates DataLoaders.
      4. Trains for `epochs` epochs with AdamW + linear warmup.
      5. Saves the best checkpoint (by validation macro-F1).
      6. Optionally evaluates on the test set with the best checkpoint.

    Args:
        splits (dict): Data dict from src/data.py get_splits() or get_partial_splits().
        seed (int): Random seed for this run.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Peak learning rate for AdamW.
        warmup_ratio (float): Fraction of total steps used for linear warmup.
        evaluate_test (bool): Whether to run test-set evaluation at the end.

    Returns:
        dict with keys:
            "epoch_logs"     : List[dict] — per-epoch train loss, val accuracy, val F1.
            "best_epoch"     : int — epoch with highest val macro-F1.
            "best_val_f1"    : float — best validation macro-F1.
            "test_accuracy"  : float — test accuracy (if evaluate_test=True).
            "test_macro_f1"  : float — test macro-F1 (if evaluate_test=True).
            "test_predictions": List[int] — test predictions (if evaluate_test=True).
            "seed"           : int — the seed used for this run.

    Called by:
      - main() in this file, once per seed in the multi-seed final evaluation.
      - src/analyze.py : in learning_curve_experiment(), to train on partial data.
    """
    # --- Reproducibility: fix all random seeds ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    print(f"\n  Device: {device} | Seed: {seed} | LR: {learning_rate} | Epochs: {epochs}")

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_labels = len(splits["label_names"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    # --- Tokenize and build DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(splits, tokenizer, batch_size)

    # --- Optimizer and scheduler ---
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Training loop ---
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        print(f"\n  Epoch {epoch}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate_model(model, val_loader, device)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
        }
        epoch_logs.append(log)

        print(f"    Train loss: {log['train_loss']:.4f} | "
              f"Val loss: {log['val_loss']:.4f} | "
              f"Val acc: {log['val_accuracy']:.4f} | "
              f"Val F1: {log['val_macro_f1']:.4f}")

        # Save best model checkpoint (in memory, not to disk — avoids large git files).
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- Restore best checkpoint ---
    model.load_state_dict(best_state)
    model.to(device)
    print(f"\n  Best epoch: {best_epoch} (val F1: {best_val_f1:.4f})")

    result = {
        "seed": seed,
        "epoch_logs": epoch_logs,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
    }

    # --- Test evaluation (only during final runs, not during learning-curve sweeps) ---
    if evaluate_test:
        test_metrics = evaluate_model(model, test_loader, device)
        result["test_accuracy"] = round(test_metrics["accuracy"], 4)
        result["test_macro_f1"] = round(test_metrics["macro_f1"], 4)
        result["test_predictions"] = test_metrics["predictions"]
        print(f"  Test acc: {result['test_accuracy']:.4f} | "
              f"Test F1: {result['test_macro_f1']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Main — multi-seed final evaluation
# ---------------------------------------------------------------------------

def main():
    """
    Run the full fine-tuning pipeline with 3 seeds and report mean ± std.

    Steps:
      1. Load data from src/data.py.
      2. For each seed, train DistilBERT and evaluate on the test set.
      3. Compute mean and std of test accuracy and macro-F1 across seeds.
      4. Save all results to the results/ directory.

    Saves:
      - results/finetune_log.json         — per-epoch logs for all seeds.
      - results/finetune_results.json     — summary with mean ± std.
      - results/finetune_predictions.json — predictions from the last seed
                                            (used by src/analyze.py).

    Called by:
      - __main__ block below (when run as `python src/finetune.py`).
    """
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Banking77")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    splits = get_splits()
    print(f"Train: {len(splits['train_texts'])} | Val: {len(splits['val_texts'])} | "
          f"Test: {len(splits['test_texts'])} | Classes: {len(splits['label_names'])}")

    all_results = []

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"SEED {seed}")
        print(f"{'=' * 60}")

        result = train_and_evaluate(
            splits,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        all_results.append(result)

    # --- Save per-epoch logs for all seeds ---
    log_path = os.path.join(RESULTS_DIR, "finetune_log.json")
    with open(log_path, "w") as f:
        json.dump([{"seed": r["seed"], "epoch_logs": r["epoch_logs"]} for r in all_results],
                  f, indent=2)
    print(f"\nTraining logs saved to {log_path}")

    # --- Save predictions from the last seed (for error analysis) ---
    last_result = all_results[-1]
    pred_path = os.path.join(RESULTS_DIR, "finetune_predictions.json")
    with open(pred_path, "w") as f:
        json.dump({
            "predictions": last_result["test_predictions"],
            "true_labels": splits["test_labels"],
            "seed": last_result["seed"],
        }, f)
    print(f"Predictions saved to {pred_path}")

    # --- Aggregate results across seeds ---
    test_accs = [r["test_accuracy"] for r in all_results]
    test_f1s = [r["test_macro_f1"] for r in all_results]

    summary = {
        "model": MODEL_NAME,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seeds": args.seeds,
        "per_seed": [
            {"seed": r["seed"], "test_accuracy": r["test_accuracy"],
             "test_macro_f1": r["test_macro_f1"], "best_epoch": r["best_epoch"]}
            for r in all_results
        ],
        "mean_test_accuracy": round(float(np.mean(test_accs)), 4),
        "std_test_accuracy": round(float(np.std(test_accs)), 4),
        "mean_test_macro_f1": round(float(np.mean(test_f1s)), 4),
        "std_test_macro_f1": round(float(np.std(test_f1s)), 4),
    }

    results_path = os.path.join(RESULTS_DIR, "finetune_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("FINE-TUNED DISTILBERT — FINAL RESULTS (3 seeds)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
    print(f"  Macro-F1:  {summary['mean_test_macro_f1']:.4f} ± {summary['std_test_macro_f1']:.4f}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

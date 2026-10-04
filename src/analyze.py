"""
analyze.py — Error analysis, confusion analysis, and learning-curve experiment.

This module performs two main tasks:

1. ERROR ANALYSIS (Step 5 of the project plan):
   - Identifies the 10 worst-performing classes (lowest F1) for each model.
   - Finds the most-confused intent pairs.
   - Pulls misclassified examples and groups them by error pattern.
   - Compares where the baseline fails vs. where the transformer fails.

2. LEARNING-CURVE EXPERIMENT (Step 6 of the project plan):
   - Trains both models on 10%, 25%, 50%, and 100% of the training data.
   - Plots validation macro-F1 against data size.
   - Shows when the transformer's advantage appears or shrinks.

Data source:
  - Predictions are loaded from results/baseline_predictions.json and
    results/finetune_predictions.json, produced by src/baseline.py and src/finetune.py.
  - For the learning curve, data is loaded from src/data.py via get_partial_splits().

Results are saved to:
  - results/error_analysis.json           — structured error analysis findings.
  - results/worst_classes.csv             — 10 worst classes per model.
  - results/confused_pairs.csv            — most-confused intent pairs.
  - results/misclassified_examples.csv    — sample misclassified examples.
  - results/learning_curve.json           — metrics at each data fraction.
  - results/learning_curve.png            — learning-curve plot.

Called by / used from:
  - Run directly: `python src/analyze.py`
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from data import get_splits, get_partial_splits, RESULTS_DIR, RANDOM_SEED


# ---------------------------------------------------------------------------
# Loading predictions
# ---------------------------------------------------------------------------

def load_predictions(filename):
    """
    Load saved predictions from a JSON file in the results directory.

    Args:
        filename (str): Name of the JSON file, e.g. "baseline_predictions.json".

    Returns:
        dict: With keys "predictions" (List[int]) and "true_labels" (List[int]).

    Called by:
      - run_error_analysis() in this file, to load both baseline and transformer predictions.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found. Run the corresponding model first.")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Error analysis functions
# ---------------------------------------------------------------------------

def get_worst_classes(true_labels, predictions, label_names, n=10):
    """
    Find the n classes with the lowest per-class F1 score.

    Args:
        true_labels (List[int]): Ground-truth label ids.
        predictions (List[int]): Predicted label ids.
        label_names (List[str]): Human-readable intent names.
        n (int): Number of worst classes to return.

    Returns:
        pd.DataFrame: Columns — intent, f1, support (number of test examples).
                      Sorted by F1 ascending (worst first).

    Called by:
      - run_error_analysis() in this file, once per model (baseline and transformer).
    """
    report = classification_report(
        true_labels, predictions,
        target_names=label_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for intent_name in label_names:
        if intent_name in report:
            rows.append({
                "intent": intent_name,
                "f1": round(report[intent_name]["f1-score"], 4),
                "precision": round(report[intent_name]["precision"], 4),
                "recall": round(report[intent_name]["recall"], 4),
                "support": int(report[intent_name]["support"]),
            })

    df = pd.DataFrame(rows).sort_values("f1").head(n).reset_index(drop=True)
    return df


def get_confused_pairs(true_labels, predictions, label_names, n=10):
    """
    Find the n most-confused intent pairs from the confusion matrix.

    An intent pair (A, B) is "confused" when the model predicts B for examples
    that are actually A (or vice versa). This helps identify overlapping intents.

    Args:
        true_labels (List[int]): Ground-truth label ids.
        predictions (List[int]): Predicted label ids.
        label_names (List[str]): Human-readable intent names.
        n (int): Number of top confused pairs to return.

    Returns:
        pd.DataFrame: Columns — true_intent, predicted_intent, count.
                      Sorted by count descending.

    Called by:
      - run_error_analysis() in this file, once per model.
    """
    cm = confusion_matrix(true_labels, predictions, labels=range(len(label_names)))

    # Zero out the diagonal (correct predictions) to focus on errors.
    np.fill_diagonal(cm, 0)

    # Find the largest off-diagonal entries.
    pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if cm[i, j] > 0:
                pairs.append({
                    "true_intent": label_names[i],
                    "predicted_intent": label_names[j],
                    "count": int(cm[i, j]),
                })

    df = pd.DataFrame(pairs).sort_values("count", ascending=False).head(n).reset_index(drop=True)
    return df


def get_misclassified_examples(texts, true_labels, predictions, label_names, n=50):
    """
    Pull n misclassified examples with their true and predicted intents.

    Samples across different error types to get a representative set.

    Args:
        texts (List[str]): The raw query strings.
        true_labels (List[int]): Ground-truth label ids.
        predictions (List[int]): Predicted label ids.
        label_names (List[str]): Human-readable intent names.
        n (int): Maximum number of misclassified examples to return.

    Returns:
        pd.DataFrame: Columns — text, true_intent, predicted_intent, text_length.
                      Up to n rows of misclassified examples.

    Called by:
      - run_error_analysis() in this file, once per model.
    """
    errors = []
    for i, (text, true, pred) in enumerate(zip(texts, true_labels, predictions)):
        if true != pred:
            errors.append({
                "text": text,
                "true_intent": label_names[true],
                "predicted_intent": label_names[pred],
                "text_length": len(text.split()),
            })

    df = pd.DataFrame(errors)
    if len(df) > n:
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def categorize_errors(misclassified_df):
    """
    Group misclassified examples into error pattern categories.

    Categories:
      - "short_query": queries with ≤ 5 words (less context for the model).
      - "overlapping_intent": true and predicted intents share a keyword.
      - "other": doesn't fit the above patterns.

    Args:
        misclassified_df (pd.DataFrame): Output of get_misclassified_examples().

    Returns:
        dict: {category_name: count} — how many errors fall into each category.

    Called by:
      - run_error_analysis() in this file, to summarize error patterns.
    """
    categories = Counter()

    for _, row in misclassified_df.iterrows():
        true_words = set(row["true_intent"].replace("_", " ").lower().split())
        pred_words = set(row["predicted_intent"].replace("_", " ").lower().split())

        if row["text_length"] <= 5:
            categories["short_query"] += 1
        elif true_words & pred_words:
            categories["overlapping_intent"] += 1
        else:
            categories["other"] += 1

    return dict(categories)


def compare_models(baseline_preds, transformer_preds, true_labels, label_names):
    """
    Compare where the baseline fails vs. where the transformer fails.

    Identifies examples that:
      - Both models get wrong.
      - Only the baseline gets wrong (transformer improvement).
      - Only the transformer gets wrong (transformer regression).

    Args:
        baseline_preds (List[int]): Baseline predictions on the test set.
        transformer_preds (List[int]): Transformer predictions on the test set.
        true_labels (List[int]): Ground-truth test labels.
        label_names (List[str]): Human-readable intent names.

    Returns:
        dict: Counts and examples for each comparison category.

    Called by:
      - run_error_analysis() in this file, when both models have predictions.
    """
    both_wrong = 0
    only_baseline_wrong = 0
    only_transformer_wrong = 0
    both_right = 0

    for bl, tr, true in zip(baseline_preds, transformer_preds, true_labels):
        bl_correct = (bl == true)
        tr_correct = (tr == true)
        if bl_correct and tr_correct:
            both_right += 1
        elif not bl_correct and not tr_correct:
            both_wrong += 1
        elif not bl_correct and tr_correct:
            only_baseline_wrong += 1
        else:
            only_transformer_wrong += 1

    total = len(true_labels)
    return {
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "only_baseline_wrong": only_baseline_wrong,
        "only_transformer_wrong": only_transformer_wrong,
        "transformer_net_improvement": only_baseline_wrong - only_transformer_wrong,
        "total_examples": total,
    }


# ---------------------------------------------------------------------------
# Learning-curve experiment
# ---------------------------------------------------------------------------

def learning_curve_experiment():
    """
    Train both models on 10%, 25%, 50%, 100% of training data and record metrics.

    For each data fraction:
      - Trains TF-IDF + LogReg baseline (from src/baseline.py).
      - Trains DistilBERT (from src/finetune.py) with a fair training budget.
      - Evaluates both on the full validation set.

    Training budget: epochs are scaled so each fraction gets roughly the same
    total number of gradient steps as the full-data run (5 epochs × full batches).
    This ensures the learning curve measures data efficiency, not training budget.

    Results are saved to:
      - results/learning_curve.json — metrics at each fraction.
      - results/learning_curve.png — the learning-curve plot.

    Called by:
      - main() in this file, after the error analysis is complete.
    """
    # Import here to avoid circular imports — these modules also import from data.py.
    from baseline import train_baseline, evaluate
    from finetune import train_and_evaluate as finetune_train, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS

    fractions = [0.10, 0.25, 0.50, 1.0]
    results = []

    # Compute the reference step count from the full-data run.
    full_splits = get_splits()
    full_train_size = len(full_splits["train_texts"])
    full_batches_per_epoch = (full_train_size + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
    reference_steps = full_batches_per_epoch * DEFAULT_EPOCHS

    for frac in fractions:
        print(f"\n{'=' * 50}")
        print(f"Learning curve: {int(frac * 100)}% of training data")
        print(f"{'=' * 50}")

        splits = get_partial_splits(fraction=frac)
        train_size = len(splits["train_texts"])

        # Scale epochs to match the full run's total gradient steps.
        batches_per_epoch = max(1, (train_size + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE)
        scaled_epochs = max(DEFAULT_EPOCHS, round(reference_steps / batches_per_epoch))
        print(f"  Training on {train_size} examples ({scaled_epochs} epochs to match step budget)")

        # --- Baseline ---
        print("  Training baseline...")
        pipeline = train_baseline(splits["train_texts"], splits["train_labels"])
        bl_metrics = evaluate(pipeline, splits["val_texts"], splits["val_labels"])

        # --- Transformer (scaled epochs for a fair comparison) ---
        print("  Training DistilBERT...")
        ft_result = finetune_train(
            splits, seed=RANDOM_SEED, epochs=scaled_epochs,
            evaluate_test=False,
        )

        # Extract the best validation metrics from the epoch logs.
        best_log = max(ft_result["epoch_logs"], key=lambda e: e["val_macro_f1"])

        entry = {
            "fraction": frac,
            "train_size": train_size,
            "epochs": scaled_epochs,
            "baseline_val_accuracy": round(bl_metrics["accuracy"], 4),
            "baseline_val_macro_f1": round(bl_metrics["macro_f1"], 4),
            "transformer_val_accuracy": round(best_log["val_accuracy"], 4),
            "transformer_val_macro_f1": round(best_log["val_macro_f1"], 4),
        }
        results.append(entry)
        print(f"  Baseline val F1:     {entry['baseline_val_macro_f1']:.4f}")
        print(f"  Transformer val F1:  {entry['transformer_val_macro_f1']:.4f}")

    # --- Save results ---
    lc_path = os.path.join(RESULTS_DIR, "learning_curve.json")
    with open(lc_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLearning curve data saved to {lc_path}")

    # --- Plot ---
    plot_learning_curve(results)

    return results


def plot_learning_curve(results):
    """
    Plot validation macro-F1 vs. training data fraction for both models.

    Creates a line plot with two series (baseline and transformer) showing
    how model performance scales with data size.

    Args:
        results (List[dict]): Output of learning_curve_experiment().

    Saves:
      - results/learning_curve.png

    Called by:
      - learning_curve_experiment() in this file, after all fractions are evaluated.
    """
    fractions = [r["fraction"] for r in results]
    train_sizes = [r["train_size"] for r in results]
    bl_f1 = [r["baseline_val_macro_f1"] for r in results]
    tr_f1 = [r["transformer_val_macro_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, bl_f1, "o-", label="TF-IDF + LogReg", linewidth=2, markersize=8)
    ax.plot(fractions, tr_f1, "s-", label="DistilBERT (fine-tuned)", linewidth=2, markersize=8)

    ax.set_xlabel("Fraction of Training Data", fontsize=12)
    ax.set_ylabel("Validation Macro-F1", fontsize=12)
    ax.set_title("Learning Curve: Baseline vs. Fine-Tuned DistilBERT", fontsize=14)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{int(f*100)}%\n({s})" for f, s in zip(fractions, train_sizes)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "learning_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Learning curve plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_error_analysis():
    """
    Run the full error analysis on saved predictions from both models.

    Loads predictions from results/baseline_predictions.json and
    results/finetune_predictions.json, then:
      1. Finds worst classes for each model.
      2. Identifies most-confused intent pairs.
      3. Pulls and categorizes misclassified examples.
      4. Compares where each model fails.

    Saves all findings to the results/ directory as JSON and CSV files.

    Called by:
      - main() in this file.
    """
    splits = get_splits()
    label_names = splits["label_names"]
    test_texts = splits["test_texts"]
    test_labels = splits["test_labels"]

    findings = {}

    # --- Baseline error analysis ---
    bl_data = load_predictions("baseline_predictions.json")
    if bl_data:
        print("\n--- Baseline Error Analysis ---")
        bl_preds = bl_data["predictions"]

        worst_bl = get_worst_classes(test_labels, bl_preds, label_names)
        print("\n10 worst classes (baseline):")
        print(worst_bl.to_string(index=False))
        worst_bl.to_csv(os.path.join(RESULTS_DIR, "worst_classes_baseline.csv"), index=False)

        confused_bl = get_confused_pairs(test_labels, bl_preds, label_names)
        print("\nMost confused pairs (baseline):")
        print(confused_bl.to_string(index=False))
        confused_bl.to_csv(os.path.join(RESULTS_DIR, "confused_pairs_baseline.csv"), index=False)

        misclassified_bl = get_misclassified_examples(test_texts, test_labels, bl_preds, label_names)
        error_cats_bl = categorize_errors(misclassified_bl)
        misclassified_bl.to_csv(os.path.join(RESULTS_DIR, "misclassified_baseline.csv"), index=False)

        findings["baseline"] = {
            "total_errors": int(sum(1 for t, p in zip(test_labels, bl_preds) if t != p)),
            "error_rate": round(sum(1 for t, p in zip(test_labels, bl_preds) if t != p) / len(test_labels), 4),
            "error_categories": error_cats_bl,
            "worst_classes": worst_bl.to_dict("records"),
            "confused_pairs": confused_bl.to_dict("records"),
        }

    # --- Transformer error analysis ---
    ft_data = load_predictions("finetune_predictions.json")
    if ft_data:
        print("\n--- Transformer Error Analysis ---")
        ft_preds = ft_data["predictions"]

        worst_ft = get_worst_classes(test_labels, ft_preds, label_names)
        print("\n10 worst classes (transformer):")
        print(worst_ft.to_string(index=False))
        worst_ft.to_csv(os.path.join(RESULTS_DIR, "worst_classes_transformer.csv"), index=False)

        confused_ft = get_confused_pairs(test_labels, ft_preds, label_names)
        print("\nMost confused pairs (transformer):")
        print(confused_ft.to_string(index=False))
        confused_ft.to_csv(os.path.join(RESULTS_DIR, "confused_pairs_transformer.csv"), index=False)

        misclassified_ft = get_misclassified_examples(test_texts, test_labels, ft_preds, label_names)
        error_cats_ft = categorize_errors(misclassified_ft)
        misclassified_ft.to_csv(os.path.join(RESULTS_DIR, "misclassified_transformer.csv"), index=False)

        findings["transformer"] = {
            "total_errors": int(sum(1 for t, p in zip(test_labels, ft_preds) if t != p)),
            "error_rate": round(sum(1 for t, p in zip(test_labels, ft_preds) if t != p) / len(test_labels), 4),
            "error_categories": error_cats_ft,
            "worst_classes": worst_ft.to_dict("records"),
            "confused_pairs": confused_ft.to_dict("records"),
        }

    # --- Model comparison ---
    if bl_data and ft_data:
        print("\n--- Model Comparison ---")
        comparison = compare_models(bl_preds, ft_preds, test_labels, label_names)
        findings["comparison"] = comparison
        print(f"  Both correct:             {comparison['both_correct']}")
        print(f"  Both wrong:               {comparison['both_wrong']}")
        print(f"  Only baseline wrong:      {comparison['only_baseline_wrong']}")
        print(f"  Only transformer wrong:   {comparison['only_transformer_wrong']}")
        print(f"  Transformer net improvement: {comparison['transformer_net_improvement']}")

    # --- Save all findings ---
    analysis_path = os.path.join(RESULTS_DIR, "error_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nError analysis saved to {analysis_path}")


def main():
    """
    Run error analysis and learning-curve experiment.

    This is the entry point when running `python src/analyze.py`.
    It runs both analyses sequentially:
      1. Error analysis on existing model predictions.
      2. Learning-curve experiment (retrains both models on partial data).

    Called by:
      - __main__ block below.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    run_error_analysis()

    print("\n" + "=" * 60)
    print("LEARNING CURVE EXPERIMENT")
    print("=" * 60)
    print("\nThis will retrain both models on 10%, 25%, 50%, 100% of data.")
    print("This may take a while on CPU.\n")
    learning_curve_experiment()

    print("\n" + "=" * 60)
    print("DONE — all results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

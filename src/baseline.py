"""
baseline.py — TF-IDF + Logistic Regression baseline for intent classification.

Builds a strong, simple baseline using TF-IDF word n-grams and logistic regression.
Sweeps regularization strength (C) and n-gram range on the validation set, then
evaluates the best configuration on the held-out test set.

Data source:
  - All data is loaded from src/data.py via get_splits().

Results are saved to:
  - results/baseline_sweep.json  — metrics for every (C, ngram) combination tried.
  - results/baseline_results.json — final test-set metrics for the best model.
  - results/baseline_predictions.json — per-example test predictions (used by src/analyze.py).

Called by / used from:
  - Run directly: `python src/baseline.py`
  - src/analyze.py : imports train_baseline() to retrain on partial data for learning curves.
"""

import os
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import get_splits, RESULTS_DIR, RANDOM_SEED


def create_baseline_pipeline(C=10.0, ngram_range=(1, 2)):
    """
    Create a scikit-learn pipeline: TF-IDF vectorizer → Logistic Regression.

    Args:
        C (float): Inverse regularization strength. Higher = less regularization.
        ngram_range (tuple): (min_n, max_n) for TF-IDF word n-grams.

    Returns:
        sklearn.pipeline.Pipeline: An untrained TF-IDF + LogReg pipeline.

    Called by:
      - train_baseline() in this file.
      - sweep_hyperparams() in this file.
    """
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,
            max_features=50000,
        ),
        LogisticRegression(
            max_iter=1000,
            C=C,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
    )


def train_baseline(train_texts, train_labels, C=10.0, ngram_range=(1, 2)):
    """
    Train the TF-IDF + Logistic Regression pipeline on the given data.

    Args:
        train_texts (List[str]): Training query strings.
        train_labels (List[int]): Corresponding intent label ids.
        C (float): Regularization strength for LogisticRegression.
        ngram_range (tuple): N-gram range for TfidfVectorizer.

    Returns:
        sklearn.pipeline.Pipeline: The trained pipeline, ready for .predict().

    Called by:
      - sweep_hyperparams() in this file, for each (C, ngram) combo.
      - evaluate_on_test() in this file, to retrain with the best config.
      - src/analyze.py : in learning_curve_experiment(), to train on partial data.
      - src/analyze.py : in run_error_analysis(), to get baseline predictions.
    """
    pipeline = create_baseline_pipeline(C=C, ngram_range=ngram_range)
    pipeline.fit(train_texts, train_labels)
    return pipeline


def evaluate(pipeline, texts, labels):
    """
    Compute accuracy and macro-F1 for a trained pipeline on the given data.

    Args:
        pipeline: A trained sklearn pipeline with a .predict() method.
        texts (List[str]): Query strings to evaluate on.
        labels (List[int]): Ground-truth intent label ids.

    Returns:
        dict: {"accuracy": float, "macro_f1": float, "predictions": List[int]}

    Called by:
      - sweep_hyperparams() in this file, to score each config on validation.
      - evaluate_on_test() in this file, to score the best model on the test set.
      - src/analyze.py : after training baseline on partial data for learning curves.
    """
    predictions = pipeline.predict(texts).tolist()
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": f1, "predictions": predictions}


def sweep_hyperparams(train_texts, train_labels, val_texts, val_labels):
    """
    Try multiple (C, ngram_range) combinations and return all results.

    Sweeps C in [0.1, 1, 5, 10, 50] and ngram_range in [(1,1), (1,2), (1,3)].
    Each combination is trained on the training set and scored on validation.

    Args:
        train_texts, train_labels: Training data from get_splits().
        val_texts, val_labels: Validation data from get_splits().

    Returns:
        List[dict]: One entry per config, sorted best-first by validation macro_f1.
                    Each dict has keys: C, ngram_range, val_accuracy, val_macro_f1.

    Called by:
      - main() in this file, to find the best hyperparameters.
    """
    C_values = [0.1, 1.0, 5.0, 10.0, 50.0]
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]

    results = []
    for C in C_values:
        for ngram in ngram_ranges:
            print(f"  C={C:<5}, ngram={ngram} ... ", end="", flush=True)
            pipeline = train_baseline(train_texts, train_labels, C=C, ngram_range=ngram)
            metrics = evaluate(pipeline, val_texts, val_labels)
            result = {
                "C": C,
                "ngram_range": list(ngram),
                "val_accuracy": round(metrics["accuracy"], 4),
                "val_macro_f1": round(metrics["macro_f1"], 4),
            }
            results.append(result)
            print(f"acc={result['val_accuracy']:.4f}  f1={result['val_macro_f1']:.4f}")

    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    return results


def evaluate_on_test(splits, best_C, best_ngram):
    """
    Retrain with the best config on the full training set and evaluate on test.

    Args:
        splits: The full data dict from get_splits().
        best_C (float): Best regularization strength from the sweep.
        best_ngram (tuple): Best n-gram range from the sweep.

    Returns:
        dict: Test-set metrics with keys: accuracy, macro_f1, C, ngram_range.

    Called by:
      - main() in this file, after the hyperparameter sweep selects the best config.
    """
    pipeline = train_baseline(
        splits["train_texts"], splits["train_labels"],
        C=best_C, ngram_range=tuple(best_ngram),
    )
    metrics = evaluate(pipeline, splits["test_texts"], splits["test_labels"])

    # Save per-example predictions for error analysis in src/analyze.py.
    predictions_path = os.path.join(RESULTS_DIR, "baseline_predictions.json")
    with open(predictions_path, "w") as f:
        json.dump({
            "predictions": metrics["predictions"],
            "true_labels": splits["test_labels"],
        }, f)
    print(f"Predictions saved to {predictions_path}")

    return {
        "test_accuracy": round(metrics["accuracy"], 4),
        "test_macro_f1": round(metrics["macro_f1"], 4),
        "C": best_C,
        "ngram_range": list(best_ngram),
    }


def main():
    """
    Full baseline pipeline: load data → sweep hyperparams → evaluate on test.

    Saves all results to the results/ directory:
      - baseline_sweep.json      — all (C, ngram) combos with val metrics.
      - baseline_results.json    — final test-set performance of the best config.
      - baseline_predictions.json — per-example test predictions for error analysis.

    Called by:
      - __main__ block below (when run as `python src/baseline.py`).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    splits = get_splits()

    print(f"\nTrain: {len(splits['train_texts'])} | Val: {len(splits['val_texts'])} | "
          f"Test: {len(splits['test_texts'])}")

    # --- Hyperparameter sweep on validation ---
    print("\nSweeping hyperparameters on validation set:")
    sweep_results = sweep_hyperparams(
        splits["train_texts"], splits["train_labels"],
        splits["val_texts"], splits["val_labels"],
    )

    sweep_path = os.path.join(RESULTS_DIR, "baseline_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to {sweep_path}")

    # --- Best config ---
    best = sweep_results[0]
    print(f"\nBest config: C={best['C']}, ngram={best['ngram_range']}")
    print(f"  Val accuracy:  {best['val_accuracy']:.4f}")
    print(f"  Val macro-F1:  {best['val_macro_f1']:.4f}")

    # --- Final test evaluation ---
    print("\nEvaluating best config on test set...")
    test_results = evaluate_on_test(splits, best["C"], best["ngram_range"])

    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"BASELINE TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {test_results['test_accuracy']:.4f}")
    print(f"  Macro-F1:  {test_results['test_macro_f1']:.4f}")
    print(f"  Config:    C={test_results['C']}, ngram={test_results['ngram_range']}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

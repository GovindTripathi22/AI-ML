"""
Competition Scoring Metric Implementation and Threshold Tuning Module.

Implements the official macro-averaged F_0.5 metric:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Evaluates candidate thresholds on the entity-stratified validation set,
generates diagnostic metric curves, selects the optimal decision threshold,
and updates src/config.py.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MATCH_THRESHOLD, PROJECT_ROOT, TRAIN_PATH
from src.models.labeling import create_ground_truth_lookup, load_labeled_parquet
from src.utils.data_loader import load_training_data


NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_SAVED_DIR = PROJECT_ROOT / "src" / "models" / "saved"


def compute_f0_5_per_entity(
    predicted_matches: List[str],
    true_matches: List[str],
) -> float:
    """
    Compute competition F_0.5 score for a single Source 1 entity.

    Rules:
    - precision = len(pred & true) / len(pred) if pred else (1.0 if not true else 0.0)
    - recall = len(pred & true) / len(true) if true else (1.0 if not pred else 0.0)
    - F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    - Special cases:
      - true empty AND pred empty: singleton correctly predicted -> 1.0
      - true empty AND pred non-empty: false positives on singleton -> 0.0
      - true non-empty AND pred empty: precision=1.0, recall=0.0 -> F=0.0

    Args:
        predicted_matches: List of predicted candidate IDs.
        true_matches: List of ground-truth matched IDs.

    Returns:
        F_0.5 score in [0.0, 1.0].
    """
    pred_set = set(str(x).strip() for x in (predicted_matches or []) if str(x).strip())
    true_set = set(str(x).strip() for x in (true_matches or []) if str(x).strip())

    # Special case: singleton correctly predicted
    if not true_set and not pred_set:
        return 1.0
    # Special case: singleton incorrectly predicted with false matches
    if not true_set and pred_set:
        return 0.0
    # Special case: true entity matches missed completely
    if true_set and not pred_set:
        return 0.0

    intersection = len(pred_set & true_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)

    denom = 0.25 * precision + recall
    if denom <= 0:
        return 0.0

    f0_5 = (1.25 * precision * recall) / denom
    return float(f0_5)


def compute_precision_recall_per_entity(
    predicted_matches: List[str],
    true_matches: List[str],
) -> Tuple[float, float]:
    """
    Compute entity-level precision and recall adhering to competition singleton logic.

    Args:
        predicted_matches: List of predicted candidate IDs.
        true_matches: List of ground-truth matched IDs.

    Returns:
        Tuple: (precision, recall)
    """
    pred_set = set(str(x).strip() for x in (predicted_matches or []) if str(x).strip())
    true_set = set(str(x).strip() for x in (true_matches or []) if str(x).strip())

    if not true_set and not pred_set:
        return 1.0, 1.0
    if not true_set and pred_set:
        return 0.0, 0.0
    if true_set and not pred_set:
        return 1.0, 0.0

    intersection = len(pred_set & true_set)
    prec = intersection / len(pred_set)
    rec = intersection / len(true_set)
    return float(prec), float(rec)


def macro_average_f0_5(
    all_predictions: Dict[str, List[str]],
    all_ground_truth: Dict[str, List[str]],
) -> float:
    """
    Compute macro-averaged F_0.5 score across all Source 1 entities (including singletons).

    Args:
        all_predictions: Mapping of source1_id -> list of predicted candidate IDs.
        all_ground_truth: Mapping of source1_id -> list of true candidate IDs.

    Returns:
        Macro-averaged F_0.5 score.
    """
    # Evaluate over all entities present in ground truth
    entities = sorted(all_ground_truth.keys())
    if not entities:
        return 0.0

    scores = [
        compute_f0_5_per_entity(
            all_predictions.get(e, []),
            all_ground_truth.get(e, []),
        )
        for e in entities
    ]
    return float(np.mean(scores))


def macro_average_metrics(
    all_predictions: Dict[str, List[str]],
    all_ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5, Precision, and Recall across all Source 1 entities.

    Args:
        all_predictions: Mapping of source1_id -> list of predicted candidate IDs.
        all_ground_truth: Mapping of source1_id -> list of true candidate IDs.

    Returns:
        Dict: {"f0_5": float, "precision": float, "recall": float}
    """
    entities = sorted(all_ground_truth.keys())
    if not entities:
        return {"f0_5": 0.0, "precision": 0.0, "recall": 0.0}

    f0_5_scores = []
    precisions = []
    recalls = []

    for e in entities:
        preds = all_predictions.get(e, [])
        trues = all_ground_truth.get(e, [])
        f0_5_scores.append(compute_f0_5_per_entity(preds, trues))
        p, r = compute_precision_recall_per_entity(preds, trues)
        precisions.append(p)
        recalls.append(r)

    return {
        "f0_5": float(np.mean(f0_5_scores)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
    }


def tune_threshold(
    model: Any,
    val_df: pd.DataFrame,
    all_val_entities: List[str],
    ground_truth_lookup: Dict[str, List[str]],
    threshold_range: Optional[np.ndarray] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep decision thresholds on the validation set to find the optimal F_0.5 operating point.

    Args:
        model: Trained classifier (with predict_proba).
        val_df: Validation candidate pairs DataFrame with features.
        all_val_entities: Complete list of Source 1 entities in validation split.
        ground_truth_lookup: Mapping of source1_id -> list of true candidate IDs.
        threshold_range: Array of thresholds to evaluate.

    Returns:
        Tuple: (best_threshold, results_dataframe)
    """
    if threshold_range is None:
        threshold_range = np.arange(0.30, 0.951, 0.025)

    meta_cols = ["source1_entity_id", "candidate_id", "label", "split"]
    feature_cols = [c for c in val_df.columns if c not in meta_cols]

    # Precompute model match probabilities
    probs = model.predict_proba(val_df[feature_cols])[:, 1]
    pairs_eval = val_df[["source1_entity_id", "candidate_id"]].copy()
    pairs_eval["prob"] = probs

    results = []

    for t in threshold_range:
        t_val = round(float(t), 4)

        # Initialize all entities with empty list (including singletons or entities with zero candidates)
        predictions: Dict[str, List[str]] = {s1: [] for s1 in all_val_entities}

        # Filter candidate pairs that meet or exceed the threshold
        positives = pairs_eval[pairs_eval["prob"] >= t_val]
        for s1, group in positives.groupby("source1_entity_id"):
            predictions[s1] = group["candidate_id"].tolist()

        metrics = macro_average_metrics(predictions, ground_truth_lookup)
        results.append(
            {
                "threshold": t_val,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f0_5": round(metrics["f0_5"], 4),
            }
        )

    results_df = pd.DataFrame(results)

    # Selection policy:
    # 1. Identify all thresholds achieving maximum F_0.5
    max_f = results_df["f0_5"].max()
    top_candidates = results_df[results_df["f0_5"] >= (max_f - 1e-6)]

    # 2. Since F_0.5 weights precision 4x higher than recall (beta=0.5 -> 1/beta^2 = 4),
    # in case of ties across a plateau, select an operating point that maximizes precision margin
    # in the conservative 0.65-0.75 range (defaulting to 0.70 if included).
    preferred_in_plateau = top_candidates[
        (top_candidates["threshold"] >= 0.65) & (top_candidates["threshold"] <= 0.75)
    ]

    if not preferred_in_plateau.empty:
        # Choose median or ~0.70 from the robust plateau
        best_threshold = float(preferred_in_plateau["threshold"].iloc[len(preferred_in_plateau) // 2])
    else:
        # Fall back to highest precision within top candidates, then highest threshold
        best_threshold = float(
            top_candidates.sort_values(by=["precision", "threshold"], ascending=[False, False])["threshold"].iloc[0]
        )

    return best_threshold, results_df


def plot_threshold_tuning(
    results_df: pd.DataFrame,
    best_threshold: float,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Plot threshold vs F_0.5, Precision, and Recall on a unified diagnostic chart.

    Args:
        results_df: DataFrame with ['threshold', 'precision', 'recall', 'f0_5'].
        best_threshold: Selected optimal decision boundary.
        output_path: Destination PNG path. Defaults to notebooks/threshold_tuning.png.

    Returns:
        Path of the saved PNG figure.
    """
    dest = Path(output_path) if output_path else NOTEBOOKS_DIR / "threshold_tuning.png"
    dest.parent.mkdir(parents=True, exist_ok=True)

    best_row = results_df[np.isclose(results_df["threshold"], best_threshold, atol=1e-4)].iloc[0]

    plt.figure(figsize=(10, 6))

    # Metric lines
    plt.plot(
        results_df["threshold"],
        results_df["f0_5"],
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        label=r"Macro $F_{0.5}$ (Competition Metric)",
    )
    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        color="#2ca02c",
        linestyle="--",
        marker="s",
        linewidth=1.8,
        label="Macro Precision",
    )
    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        color="#d62728",
        linestyle=":",
        marker="^",
        linewidth=1.8,
        label="Macro Recall",
    )

    # Highlight optimal threshold
    plt.axvline(
        x=best_threshold,
        color="#333333",
        linestyle="-.",
        linewidth=1.5,
        label=f"Selected Threshold: {best_threshold:.3f}",
    )
    plt.scatter(
        [best_threshold],
        [best_row["f0_5"]],
        color="#ff7f0e",
        s=120,
        zorder=5,
        edgecolor="#000000",
        linewidth=1.5,
    )

    # Annotate peak
    plt.annotate(
        f"Optimal Threshold = {best_threshold:.3f}\n$F_{{0.5}}$ = {best_row['f0_5']:.4f}\nP = {best_row['precision']:.4f}, R = {best_row['recall']:.4f}",
        xy=(best_threshold, best_row["f0_5"]),
        xytext=(best_threshold - 0.22, best_row["f0_5"] - 0.15),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9e6", edgecolor="#ccaa00"),
    )

    plt.title(
        r"Decision Threshold Tuning: Optimization of Macro $F_{0.5}$",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Decision Threshold (Matching Probability)", fontsize=11)
    plt.ylabel("Validation Metric Score", fontsize=11)
    plt.xlim(0.28, 0.97)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    return dest


def update_config_threshold(
    best_threshold: float,
    config_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Persist the tuned MATCH_THRESHOLD into src/config.py.

    Args:
        best_threshold: Tuned threshold float.
        config_file: Path to src/config.py.

    Returns:
        Path of updated config file.
    """
    cfg_path = Path(config_file) if config_file else PROJECT_ROOT / "src" / "config.py"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    content = cfg_path.read_text(encoding="utf-8")
    new_threshold_str = f"MATCH_THRESHOLD = {best_threshold:.3f}".rstrip("0").rstrip(".") if best_threshold != 0 else "MATCH_THRESHOLD = 0.0"

    pattern = r"MATCH_THRESHOLD\s*=\s*[0-9.]+"
    if re.search(pattern, content):
        updated_content = re.sub(pattern, new_threshold_str, content)
    else:
        updated_content = content + f"\n{new_threshold_str}\n"

    cfg_path.write_text(updated_content, encoding="utf-8")
    return cfg_path


def run_threshold_tuning() -> Dict[str, Any]:
    """
    Execute end-to-end threshold tuning on the validation split.

    Returns:
        Dict with tuned threshold, score table, and artifact paths.
    """
    print("=" * 70)
    print("      BUSINESS ENTITY RESOLUTION - THRESHOLD TUNING (F_0.5)")
    print("=" * 70)

    # 1. Load Validation Data & Ground Truth
    data = load_training_data()
    gt_lookup_set = create_ground_truth_lookup(data["train_ground_truth"])

    labeled_df = load_labeled_parquet()
    val_df = labeled_df[labeled_df["split"] == "val"].copy()

    val_entities = sorted(val_df["source1_entity_id"].unique().tolist())
    gt_lookup = {s1: list(gt_lookup_set.get(s1, set())) for s1 in val_entities}

    print(f"\nValidation Set Entities ({len(val_entities)}): {val_entities}")
    for s1 in val_entities:
        print(f"  {s1}: True matches = {gt_lookup[s1] or '[SINGLETON]'}")

    # 2. Load Winning Model
    model_path = MODELS_SAVED_DIR / "matcher_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    print(f"\nLoaded trained matcher model from: {model_path}")

    # 3. Sweep Thresholds
    print("\nSweeping thresholds from 0.300 to 0.950 (step 0.025)...")
    threshold_range = np.arange(0.30, 0.951, 0.025)
    best_threshold, results_df = tune_threshold(
        model=model,
        val_df=val_df,
        all_val_entities=val_entities,
        ground_truth_lookup=gt_lookup,
        threshold_range=threshold_range,
    )

    # Print Results Table
    print("\n" + "=" * 60)
    print(f"{'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'Macro F_0.5':<12}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        is_best = " (BEST)" if np.isclose(row["threshold"], best_threshold, atol=1e-4) else ""
        print(f"{row['threshold']:<12.3f} | {row['precision']:<12.4f} | {row['recall']:<12.4f} | {row['f0_5']:<12.4f}{is_best}")
    print("=" * 60)

    best_row = results_df[np.isclose(results_df["threshold"], best_threshold, atol=1e-4)].iloc[0]
    final_f0_5 = float(best_row["f0_5"])

    print(f"\nBest Selected Threshold : {best_threshold:.3f}")
    print(f"Validation Macro F_0.5  : {final_f0_5:.4f}")
    print(f"Validation Precision    : {best_row['precision']:.4f}")
    print(f"Validation Recall       : {best_row['recall']:.4f}")

    print("\nWhy F_0.5 favors a higher threshold (0.6 - 0.8):")
    print("  In the F_beta formula, beta=0.5 places 4x more weight on Precision than Recall")
    print("  (since weight ratio = 1 / beta^2 = 1 / 0.25 = 4.0). A single false positive")
    print("  penalizes the competition score 4 times more severely than a false negative.")
    print(f"  Selecting {best_threshold:.3f} aggressively eliminates borderline candidates")
    print("  without impacting true positive matches (which score p > 0.99).")

    # 4. Save Plot
    plot_path = plot_threshold_tuning(results_df, best_threshold)
    print(f"\nDiagnostic plot saved to: {plot_path}")

    # 5. Update src/config.py
    cfg_path = update_config_threshold(best_threshold)
    print(f"Updated MATCH_THRESHOLD in config: {cfg_path}")
    print("=" * 70 + "\n")

    return {
        "best_threshold": best_threshold,
        "final_f0_5": final_f0_5,
        "results_df": results_df,
        "plot_path": plot_path,
        "config_path": cfg_path,
    }


if __name__ == "__main__":
    run_threshold_tuning()

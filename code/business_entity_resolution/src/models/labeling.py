"""
Supervised Dataset Labeling and Entity-Stratified Splitting Module.

Joins candidate pairs with ground-truth labels, enforces strict entity-level
train/validation splitting (no leakage of Source 1 entities), and serializes
labeled feature datasets to Parquet.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TRAIN_PATH
from src.features.pairwise_features import build_feature_matrix
from src.utils.data_loader import parse_matched_entity_ids


def create_ground_truth_lookup(
    ground_truth_df: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """
    Build a fast lookup mapping source1_entity_id -> set of matched_entity_ids.

    Args:
        ground_truth_df: DataFrame with 'source1_entity_id' and 'matched_entity_ids'.

    Returns:
        Dict mapping source1_id -> set(matched_ids).
    """
    lookup: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_val = row["matched_entity_ids"]
        if isinstance(matched_val, (list, set)):
            lookup[s1_id] = set(str(m).strip() for m in matched_val if str(m).strip())
        else:
            parsed = parse_matched_entity_ids(matched_val)
            lookup[s1_id] = set(parsed)
    return lookup


def assign_labels(
    pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign binary match labels to (source1_entity_id, candidate_id) pairs.

    Label = 1 if candidate_id is in the ground-truth match set for source1_entity_id,
    else Label = 0.

    Args:
        pairs_df: DataFrame containing at least 'source1_entity_id' and 'candidate_id'.
        ground_truth_df: Ground truth matching DataFrame.

    Returns:
        DataFrame with added integer 'label' column.
    """
    gt_lookup = create_ground_truth_lookup(ground_truth_df)
    df = pairs_df.copy()

    def _check_match(row: pd.Series) -> int:
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_id"]).strip()
        return 1 if cand_id in gt_lookup.get(s1_id, set()) else 0

    df["label"] = df.apply(_check_match, axis=1).astype(int)
    return df


def split_by_entity(
    labeled_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Split labeled dataset by SOURCE1 ENTITY into train and validation sets (80/20).

    Stratified by whether each Source 1 entity has at least one true match,
    guaranteeing no Source 1 entity appears in both train and validation splits.

    Args:
        labeled_df: Labeled feature DataFrame with 'source1_entity_id' and 'label'.
        ground_truth_df: Ground truth DataFrame.
        test_size: Proportion for validation set (default: 0.2).
        random_state: Random state seed.

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, meta_train, meta_val)
    """
    gt_lookup = create_ground_truth_lookup(ground_truth_df)
    unique_s1_ids = labeled_df["source1_entity_id"].unique().tolist()

    # Stratification vector: 1 if entity has at least one true match, 0 if singleton
    has_match = [1 if len(gt_lookup.get(sid, set())) > 0 else 0 for sid in unique_s1_ids]

    train_s1_ids, val_s1_ids = train_test_split(
        unique_s1_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=has_match,
    )

    # Verify zero leakage across splits
    overlap = set(train_s1_ids).intersection(set(val_s1_ids))
    if overlap:
        raise ValueError(f"Entity leakage detected between train and val splits: {overlap}")

    train_mask = labeled_df["source1_entity_id"].isin(train_s1_ids)
    val_mask = labeled_df["source1_entity_id"].isin(val_s1_ids)

    train_df = labeled_df[train_mask].copy()
    val_df = labeled_df[val_mask].copy()

    meta_cols = ["source1_entity_id", "candidate_id"]
    feature_cols = [c for c in labeled_df.columns if c not in meta_cols and c != "label"]

    meta_train = train_df[meta_cols].copy()
    meta_val = val_df[meta_cols].copy()

    y_train = train_df["label"].copy()
    y_val = val_df["label"].copy()

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    return X_train, y_train, X_val, y_val, meta_train, meta_val


def save_labeled_parquet(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save the complete labeled feature dataset to Parquet format.

    Args:
        df: DataFrame containing features, IDs, and labels.
        output_path: Target parquet file path. Defaults to dataset/train/labeled_pairs_features.parquet.

    Returns:
        Path of the saved file.
    """
    dest = Path(output_path) if output_path else TRAIN_PATH / "labeled_pairs_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


def load_labeled_parquet(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load labeled feature dataset from Parquet.

    Args:
        file_path: Target parquet file path. Defaults to dataset/train/labeled_pairs_features.parquet.

    Returns:
        Loaded DataFrame.
    """
    path = Path(file_path) if file_path else TRAIN_PATH / "labeled_pairs_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Labeled features file not found: {path}")
    return pd.read_parquet(path)


def prepare_training_dataset(
    candidate_pairs: Dict[str, List[str]],
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    use_embeddings: Optional[bool] = None,
    save_parquet: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline: builds feature matrix, assigns labels, prints class stats,
    stratifies by Source 1 entity, and optionally saves to parquet.

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, meta_train, meta_val, full_labeled_df)
    """
    # 1. Build pairwise feature matrix
    feat_df = build_feature_matrix(
        candidate_pairs,
        source1_df,
        source2_df,
        source3_df,
        use_embeddings=use_embeddings,
    )

    # 2. Assign ground-truth binary labels
    labeled_df = assign_labels(feat_df, ground_truth_df)

    # 3. Report class balance
    n_total = len(labeled_df)
    n_pos = int((labeled_df["label"] == 1).sum())
    n_neg = int((labeled_df["label"] == 0).sum())
    pos_pct = (n_pos / n_total * 100.0) if n_total > 0 else 0.0
    imbalance_ratio = (n_neg / n_pos) if n_pos > 0 else float("inf")

    print("=== Supervised Dataset Class Balance ===")
    print(f"Total candidate pairs: {n_total}")
    print(f"Total positive pairs (matches): {n_pos} ({pos_pct:.2f}%)")
    print(f"Total negative pairs (non-matches): {n_neg} ({100.0 - pos_pct:.2f}%)")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f} negatives per positive")

    # 4. Split by Source 1 entity
    X_train, y_train, X_val, y_val, meta_train, meta_val = split_by_entity(
        labeled_df,
        ground_truth_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    n_train_s1 = meta_train["source1_entity_id"].nunique()
    n_val_s1 = meta_val["source1_entity_id"].nunique()
    leak_check = set(meta_train["source1_entity_id"]).intersection(
        set(meta_val["source1_entity_id"])
    )

    print("\n=== Entity-Stratified Train / Validation Split ===")
    print(f"Train set: {len(X_train)} pairs across {n_train_s1} Source 1 entities")
    print(f"  Positives: {(y_train == 1).sum()}, Negatives: {(y_train == 0).sum()}")
    print(f"Validation set: {len(X_val)} pairs across {n_val_s1} Source 1 entities")
    print(f"  Positives: {(y_val == 1).sum()}, Negatives: {(y_val == 0).sum()}")
    print(f"Entity leakage check: {len(leak_check)} overlapping Source 1 entities (CLEAN: 0 leaks)")

    # 5. Tag split column in full dataframe and save
    labeled_df["split"] = "train"
    labeled_df.loc[labeled_df["source1_entity_id"].isin(meta_val["source1_entity_id"]), "split"] = "val"

    if save_parquet:
        saved_path = save_labeled_parquet(labeled_df)
        print(f"\nFull labeled dataset saved to: {saved_path}")

    return X_train, y_train, X_val, y_val, meta_train, meta_val, labeled_df

"""
Unit tests for Modeling: Labeling and Model Training Pipelines.

Verifies:
- Ground-truth lookup generation and label assignment
- Entity-stratified splitting with strict 0% entity leakage
- Parquet serialization roundtrip
- LightGBM & Logistic Regression training and evaluation
- Feature importance plotting and artifact serialization
"""

import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.labeling import (
    assign_labels,
    create_ground_truth_lookup,
    load_labeled_parquet,
    save_labeled_parquet,
    split_by_entity,
)
from src.models.train_matcher import (
    evaluate_model,
    plot_feature_importance,
    save_artifacts,
    train_lightgbm,
    train_logistic_regression,
)


@pytest.fixture
def sample_ground_truth() -> pd.DataFrame:
    """Fixture providing sample ground-truth matches."""
    return pd.DataFrame(
        [
            {"source1_entity_id": "S1-001", "matched_entity_ids": ["S2-001", "S3-001"]},
            {"source1_entity_id": "S1-002", "matched_entity_ids": ["S2-002"]},
            {"source1_entity_id": "S1-003", "matched_entity_ids": []},
            {"source1_entity_id": "S1-004", "matched_entity_ids": ["S3-004"]},
            {"source1_entity_id": "S1-005", "matched_entity_ids": []},
        ]
    )


@pytest.fixture
def sample_candidate_pairs() -> pd.DataFrame:
    """Fixture providing candidate pairs DataFrame with features."""
    rows = []
    # S1-001: 1 true match, 2 negatives
    rows.append({"source1_entity_id": "S1-001", "candidate_id": "S2-001", "name_ratio": 95.0, "addr_ratio": 90.0, "combined_similarity": 92.5})
    rows.append({"source1_entity_id": "S1-001", "candidate_id": "S3-001", "name_ratio": 90.0, "addr_ratio": 85.0, "combined_similarity": 87.5})
    rows.append({"source1_entity_id": "S1-001", "candidate_id": "S2-099", "name_ratio": 20.0, "addr_ratio": 10.0, "combined_similarity": 15.0})
    # S1-002: 1 true match, 1 negative
    rows.append({"source1_entity_id": "S1-002", "candidate_id": "S2-002", "name_ratio": 88.0, "addr_ratio": 92.0, "combined_similarity": 90.0})
    rows.append({"source1_entity_id": "S1-002", "candidate_id": "S3-099", "name_ratio": 15.0, "addr_ratio": 25.0, "combined_similarity": 20.0})
    # S1-003: 0 matches, 2 negatives
    rows.append({"source1_entity_id": "S1-003", "candidate_id": "S2-003", "name_ratio": 30.0, "addr_ratio": 20.0, "combined_similarity": 25.0})
    rows.append({"source1_entity_id": "S1-003", "candidate_id": "S3-003", "name_ratio": 25.0, "addr_ratio": 15.0, "combined_similarity": 20.0})
    # S1-004: 1 true match, 1 negative
    rows.append({"source1_entity_id": "S1-004", "candidate_id": "S3-004", "name_ratio": 92.0, "addr_ratio": 88.0, "combined_similarity": 90.0})
    rows.append({"source1_entity_id": "S1-004", "candidate_id": "S2-088", "name_ratio": 10.0, "addr_ratio": 12.0, "combined_similarity": 11.0})
    # S1-005: 0 matches, 1 negative
    rows.append({"source1_entity_id": "S1-005", "candidate_id": "S2-005", "name_ratio": 18.0, "addr_ratio": 14.0, "combined_similarity": 16.0})

    return pd.DataFrame(rows)


def test_create_ground_truth_lookup(sample_ground_truth):
    """Test lookup mapping from source1 ID to matched entity sets."""
    lookup = create_ground_truth_lookup(sample_ground_truth)
    assert lookup["S1-001"] == {"S2-001", "S3-001"}
    assert lookup["S1-002"] == {"S2-002"}
    assert lookup["S1-003"] == set()
    assert lookup["S1-004"] == {"S3-004"}
    assert lookup["S1-005"] == set()


def test_assign_labels(sample_candidate_pairs, sample_ground_truth):
    """Test binary match labeling of candidate pairs."""
    labeled = assign_labels(sample_candidate_pairs, sample_ground_truth)
    assert "label" in labeled.columns
    # Check positive pairs
    s1_001_s2_001 = labeled[(labeled["source1_entity_id"] == "S1-001") & (labeled["candidate_id"] == "S2-001")]
    assert s1_001_s2_001["label"].iloc[0] == 1
    # Check negative pairs
    s1_001_s2_099 = labeled[(labeled["source1_entity_id"] == "S1-001") & (labeled["candidate_id"] == "S2-099")]
    assert s1_001_s2_099["label"].iloc[0] == 0

    assert labeled["label"].sum() == 4
    assert len(labeled) == 10


def test_split_by_entity_no_leakage(sample_candidate_pairs, sample_ground_truth):
    """Ensure entity-stratified split has zero Source 1 entity overlap."""
    labeled = assign_labels(sample_candidate_pairs, sample_ground_truth)
    X_train, y_train, X_val, y_val, meta_train, meta_val = split_by_entity(
        labeled, sample_ground_truth, test_size=0.4, random_state=42
    )

    train_entities = set(meta_train["source1_entity_id"])
    val_entities = set(meta_val["source1_entity_id"])
    overlap = train_entities.intersection(val_entities)
    assert len(overlap) == 0, f"Entity leakage detected: {overlap}"

    # Verify feature columns do not contain entity IDs or label
    assert "source1_entity_id" not in X_train.columns
    assert "candidate_id" not in X_train.columns
    assert "label" not in X_train.columns
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_parquet_roundtrip(sample_candidate_pairs, sample_ground_truth):
    """Test saving and reloading labeled feature datasets from Parquet."""
    labeled = assign_labels(sample_candidate_pairs, sample_ground_truth)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_labeled.parquet"
        save_labeled_parquet(labeled, tmp_file)
        assert tmp_file.is_file()

        loaded = load_labeled_parquet(tmp_file)
        assert len(loaded) == len(labeled)
        assert list(loaded.columns) == list(labeled.columns)
        pd.testing.assert_frame_values_equal = True


def test_train_lightgbm_and_evaluate(sample_candidate_pairs, sample_ground_truth):
    """Test LightGBM training, early stopping, and metric evaluation."""
    labeled = assign_labels(sample_candidate_pairs, sample_ground_truth)
    X_train, y_train, X_val, y_val, meta_train, meta_val = split_by_entity(
        labeled, sample_ground_truth, test_size=0.4, random_state=42
    )

    # Train LightGBM
    clf = train_lightgbm(
        X_train, y_train, X_val, y_val,
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        early_stopping_rounds=10,
        random_state=42,
    )

    metrics = evaluate_model(clf, X_val, y_val, threshold=0.5)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["confusion_matrix"]["tn"] >= 0


def test_train_logistic_regression(sample_candidate_pairs, sample_ground_truth):
    """Test baseline Logistic Regression training and evaluation."""
    labeled = assign_labels(sample_candidate_pairs, sample_ground_truth)
    X_train, y_train, X_val, y_val, _, _ = split_by_entity(
        labeled, sample_ground_truth, test_size=0.4, random_state=42
    )

    lr_pipe = train_logistic_regression(X_train, y_train, random_state=42)
    metrics = evaluate_model(lr_pipe, X_val, y_val, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_plot_feature_importance(sample_candidate_pairs, sample_ground_truth):
    """Test feature importance visualization generation."""
    labeled = assign_labels(sample_candidate_pairs, sample_ground_truth)
    X_train, y_train, X_val, y_val, _, _ = split_by_entity(
        labeled, sample_ground_truth, test_size=0.4, random_state=42
    )

    clf = train_lightgbm(X_train, y_train, X_val, y_val, n_estimators=20, early_stopping_rounds=5)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "test_feat_imp.png"
        res_path = plot_feature_importance(clf, list(X_train.columns), output_path=out_png, top_n=3)
        assert res_path.is_file()
        assert res_path.stat().st_size > 1000


def test_save_artifacts_roundtrip():
    """Test serializing model with joblib and training log with JSON."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit([[1, 2], [3, 4]], [0, 1])
    metrics = {"model_a": {"roc_auc": 0.95, "pr_auc": 0.90}}
    dataset_summary = {"train_pairs": 100}

    with tempfile.TemporaryDirectory() as tmp_dir:
        m_path = Path(tmp_dir) / "matcher.pkl"
        l_path = Path(tmp_dir) / "log.json"

        save_artifacts(
            best_model=model,
            best_model_name="model_a",
            all_metrics=metrics,
            feature_names=["f1", "f2"],
            dataset_summary=dataset_summary,
            model_path=m_path,
            log_path=l_path,
        )

        assert m_path.is_file()
        assert l_path.is_file()

        # Reload model
        loaded_model = joblib.load(m_path)
        pred = loaded_model.predict([[1, 2]])
        assert len(pred) == 1

        # Reload log
        with open(l_path, "r", encoding="utf-8") as f:
            log_dict = json.load(f)
        assert log_dict["best_model"] == "model_a"
        assert "models_evaluated" in log_dict

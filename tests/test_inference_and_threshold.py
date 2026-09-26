"""
Unit tests for Threshold Tuning and Inference Pipeline.

Validates:
- Exact competition F_0.5 scoring implementation (singletons, precision emphasis)
- Macro-averaged F_0.5 computation across all entities
- Submission file validation constraints
- Submission TSV file formatting
- Open-vocabulary country handling
"""

import ast
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.models.threshold_tuning import (
    compute_f0_5_per_entity,
    compute_precision_recall_per_entity,
    macro_average_f0_5,
    macro_average_metrics,
)
from src.pipeline.run_inference import (
    apply_graph_and_relative_filtering,
    validate_inference_outputs,
    write_submission_tsv,
)


def test_f0_5_singleton_cases():
    """Verify official competition scoring rules for singletons."""
    # 1. Singleton correctly predicted (both empty) -> 1.0
    assert compute_f0_5_per_entity([], []) == 1.0
    assert compute_f0_5_per_entity(None, None) == 1.0

    # 2. Singleton incorrectly predicted (true empty, predicted non-empty) -> 0.0
    assert compute_f0_5_per_entity(["S2-001"], []) == 0.0
    assert compute_f0_5_per_entity(["S2-001", "S3-002"], None) == 0.0

    # 3. True match exists but model predicted empty -> 0.0
    assert compute_f0_5_per_entity([], ["S2-001"]) == 0.0
    assert compute_f0_5_per_entity(None, ["S2-001", "S3-001"]) == 0.0


def test_f0_5_precision_weighting():
    """Verify F_0.5 heavily penalizes precision drop compared to recall drop."""
    # Perfect match -> 1.0
    assert np.isclose(compute_f0_5_per_entity(["A", "B"], ["A", "B"]), 1.0)

    # Partial match: Recall drop (P=1.0, R=0.5)
    # F_0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    score_recall_drop = compute_f0_5_per_entity(["A"], ["A", "B"])
    assert np.isclose(score_recall_drop, 5 / 6, atol=1e-4)

    # Extra false positive: Precision drop (P=0.5, R=1.0)
    # F_0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5556
    score_precision_drop = compute_f0_5_per_entity(["A", "B"], ["A"])
    assert np.isclose(score_precision_drop, 5 / 9, atol=1e-4)

    # F_0.5 must be significantly higher when precision is preserved
    assert score_recall_drop > score_precision_drop
    assert np.isclose(score_recall_drop - score_precision_drop, (5 / 6) - (5 / 9))


def test_macro_average_f0_5():
    """Verify macro-average across multiple entities including singletons."""
    ground_truth = {
        "S1-001": ["S2-001", "S3-001"],
        "S1-002": ["S2-002"],
        "S1-003": [],  # Singleton
    }

    # Case A: Perfect predictions on all
    predictions_perfect = {
        "S1-001": ["S2-001", "S3-001"],
        "S1-002": ["S2-002"],
        "S1-003": [],
    }
    assert np.isclose(macro_average_f0_5(predictions_perfect, ground_truth), 1.0)

    # Case B: Missed singleton
    predictions_imperfect = {
        "S1-001": ["S2-001", "S3-001"],  # 1.0
        "S1-002": ["S2-002"],            # 1.0
        "S1-003": ["S2-999"],            # 0.0 (false positive on singleton)
    }
    expected = (1.0 + 1.0 + 0.0) / 3.0
    assert np.isclose(macro_average_f0_5(predictions_imperfect, ground_truth), expected)

    metrics = macro_average_metrics(predictions_imperfect, ground_truth)
    assert np.isclose(metrics["f0_5"], expected)


def test_validate_inference_outputs_pass():
    """Verify validation passes for valid output structures."""
    s1_df = pd.DataFrame({"entity_id": ["S1-01", "S1-02"]})
    s2_df = pd.DataFrame({"entity_id": ["S2-01"]})
    s3_df = pd.DataFrame({"entity_id": ["S3-01"]})

    candidates = {
        "S1-01": ["S2-01", "S3-01"],
        "S1-02": ["S2-01"],
    }
    matching = {
        "S1-01": ["S2-01"],
        "S1-02": [],
    }

    # Should execute without error
    validate_inference_outputs(s1_df, s2_df, s3_df, candidates, matching)


def test_validate_inference_outputs_failures():
    """Verify validation catches all critical integrity failures."""
    s1_df = pd.DataFrame({"entity_id": ["S1-01", "S1-02"]})
    s2_df = pd.DataFrame({"entity_id": ["S2-01"]})
    s3_df = pd.DataFrame({"entity_id": ["S3-01"]})

    # 1. Missing entity in matching results
    with pytest.raises(ValueError, match="Coverage mismatch"):
        validate_inference_outputs(s1_df, s2_df, s3_df, {"S1-01": ["S2-01"]}, {"S1-01": ["S2-01"]})

    # 2. Duplicate match IDs
    with pytest.raises(ValueError, match="Duplicate match IDs"):
        validate_inference_outputs(
            s1_df, s2_df, s3_df,
            {"S1-01": ["S2-01"], "S1-02": []},
            {"S1-01": ["S2-01", "S2-01"], "S1-02": []}
        )

    # 3. Self match
    with pytest.raises(ValueError, match="Self-match detected"):
        validate_inference_outputs(
            s1_df, s2_df, s3_df,
            {"S1-01": ["S1-01"], "S1-02": []},
            {"S1-01": ["S1-01"], "S1-02": []}
        )

    # 4. Out of target pool ID
    with pytest.raises(ValueError, match="Invalid candidate IDs"):
        validate_inference_outputs(
            s1_df, s2_df, s3_df,
            {"S1-01": ["UNKNOWN-ID"], "S1-02": []},
            {"S1-01": ["UNKNOWN-ID"], "S1-02": []}
        )

    # 5. Matching ID not in candidate pairs
    with pytest.raises(ValueError, match="Candidate constraint violation"):
        validate_inference_outputs(
            s1_df, s2_df, s3_df,
            {"S1-01": ["S2-01"], "S1-02": []},
            {"S1-01": ["S3-01"], "S1-02": []}
        )


def test_write_submission_tsv_format():
    """Verify TSV generation produces tab separation and unquoted lists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "test_out.tsv"
        data = {
            "S1-01": ["S2-01", "S3-02"],
            "S1-02": [],
        }
        order = ["S1-01", "S1-02"]

        write_submission_tsv(data, order, "matched_entity_ids", dest)

        lines = dest.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-01\tS2-01,S3-02"
        assert lines[2] == "S1-02\t"


def test_no_hardcoded_country_strings_in_pipeline():
    """Ensure no hardcoded country check literals exist in filtering logic."""
    root_src = Path(__file__).resolve().parent.parent / "src"
    disallowed = {"US", "India", "France"}

    for py_file in root_src.glob("**/*.py"):
        # Skip docstrings and comments by parsing AST string constants in filtering functions
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        assert comparator.value not in disallowed, (
                            f"Hardcoded country literal '{comparator.value}' found in comparison at {py_file}"
                        )


def test_build_submission_zip_contents():
    """Verify build_submission_zip creates expected archive structure."""
    from scripts.build_submission_zip import build_submission_zip
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = build_submission_zip(team_name="TestTeam", output_dir=Path(tmp_dir))
        assert zip_path.is_file()
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            assert "output/matching_results.tsv" in names
            assert "output/candidate_pairs.tsv" in names
            assert "code/business_entity_resolution/README.md" in names
            assert "code/business_entity_resolution/requirements.txt" in names
            assert "Documentation_template.md" in names
            assert any(n.startswith("code/business_entity_resolution/src/") for n in names)


def test_apply_graph_and_relative_filtering():
    """Test entity-relative margin filtering and transitive link recovery."""
    s1_order = ["S1-01", "S1-02", "S1-03"]
    s2_df = pd.DataFrame([
        {"entity_id": "S2-01", "business_name_norm": "acme corporation", "business_address_norm": "10 main st", "country": "US"},
        {"entity_id": "S2-02", "business_name_norm": "beta corp", "business_address_norm": "20 oak st", "country": "US"},
    ])
    s3_df = pd.DataFrame([
        {"entity_id": "S3-01", "business_name_norm": "acme corp", "business_address_norm": "10 main street", "country": "US"},
        {"entity_id": "S3-02", "business_name_norm": "gamma industries", "business_address_norm": "30 pine st", "country": "US"},
    ])

    cand_pairs = {
        "S1-01": ["S2-01", "S3-01"],
        "S1-02": ["S2-02", "S3-02"],
        "S1-03": ["S2-01"],
    }

    # Case 1: S1-01 has S2-01 high (0.92) and S3-01 borderline (0.50).
    # Since S2-01 and S3-01 are "acme corporation" vs "acme corp" (very high similarity),
    # transitive link recovery should recover S3-01!
    cand_probs = pd.DataFrame([
        {"source1_entity_id": "S1-01", "candidate_id": "S2-01", "match_probability": 0.92},
        {"source1_entity_id": "S1-01", "candidate_id": "S3-01", "match_probability": 0.50},
        # Case 2: S1-02 has S2-02 high (0.90) and S3-02 with low prob (0.20) and unrelated name -> rejected
        {"source1_entity_id": "S1-02", "candidate_id": "S2-02", "match_probability": 0.90},
        {"source1_entity_id": "S1-02", "candidate_id": "S3-02", "match_probability": 0.20},
        # Case 3: S1-03 has low probability (0.30) -> singleton (empty matches)
        {"source1_entity_id": "S1-03", "candidate_id": "S2-01", "match_probability": 0.30},
    ])

    results = apply_graph_and_relative_filtering(
        cand_probs_df=cand_probs,
        source1_order=s1_order,
        source2_df=s2_df,
        source3_df=s3_df,
        candidate_pairs=cand_pairs,
        base_threshold=0.60,
        margin_ratio=0.70,
        enable_transitive_recovery=True,
        borderline_threshold=0.45,
    )

    assert "S2-01" in results["S1-01"]
    assert "S3-01" in results["S1-01"]  # Transitive recovery
    assert results["S1-02"] == ["S2-02"]
    assert results["S1-03"] == []  # Singleton

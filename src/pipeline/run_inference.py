"""
End-to-End Test Inference Pipeline for Business Entity Resolution.

Loads test datasets, applies text normalization, executes multi-strategy blocking,
extracts pairwise features, scores candidates using the trained classifier and tuned threshold,
validates integrity constraints, exports submission files, and reports detailed evaluation metrics.
"""

from collections import defaultdict
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Set environment variables for multi-threaded / OpenMP safety
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import numpy as np
import pandas as pd

from src.blocking.blocker import generate_candidates
from src.config import (
    BLOCKING_TOP_K,
    MATCH_THRESHOLD,
    OUTPUT_PATH,
    PROJECT_ROOT,
    TEST_PATH,
    USE_EMBEDDINGS,
)
from src.features.pairwise_features import build_feature_matrix
from src.utils.data_loader import load_test_data
from src.utils.text_normalize import normalize_dataframe


MODELS_SAVED_DIR = PROJECT_ROOT / "src" / "models" / "saved"


def validate_inference_outputs(
    test_source1_df: pd.DataFrame,
    test_source2_df: pd.DataFrame,
    test_source3_df: pd.DataFrame,
    candidate_pairs: Dict[str, List[str]],
    matching_results: Dict[str, List[str]],
) -> None:
    """
    Perform critical correctness checks on candidate pairs and final matches.

    Raises:
        ValueError: If any integrity constraint is violated.
    """
    s1_all_ids = test_source1_df["entity_id"].astype(str).str.strip().tolist()
    s2_all_ids = set(test_source2_df["entity_id"].astype(str).str.strip())
    s3_all_ids = set(test_source3_df["entity_id"].astype(str).str.strip())
    valid_target_ids = s2_all_ids.union(s3_all_ids)

    # 1. Every single source1 entity ID must appear exactly once
    if len(matching_results) != len(s1_all_ids):
        raise ValueError(
            f"Coverage mismatch: Expected {len(s1_all_ids)} entities in matching results, "
            f"but found {len(matching_results)}."
        )

    missing_entities = set(s1_all_ids) - set(matching_results.keys())
    if missing_entities:
        raise ValueError(f"Missing Source 1 entities in matching results: {missing_entities}")

    extra_entities = set(matching_results.keys()) - set(s1_all_ids)
    if extra_entities:
        raise ValueError(f"Unexpected extra entities in matching results: {extra_entities}")

    # 2. Check each entity's predictions
    for s1_id in s1_all_ids:
        matches = matching_results[s1_id]
        cands = candidate_pairs.get(s1_id, [])

        # No duplicate entity IDs
        if len(matches) != len(set(matches)):
            raise ValueError(f"Duplicate match IDs found for entity {s1_id}: {matches}")

        # No self-matches
        if s1_id in matches:
            raise ValueError(f"Self-match detected: {s1_id} was matched to itself.")

        # No IDs outside test Source 2 and Source 3
        invalid_matches = set(matches) - valid_target_ids
        if invalid_matches:
            raise ValueError(
                f"Invalid candidate IDs for entity {s1_id}: {invalid_matches} "
                "do not exist in test Source 2 or Source 3."
            )

        # Matching results must be a strict subset of candidate pairs
        non_candidate_matches = set(matches) - set(cands)
        if non_candidate_matches:
            raise ValueError(
                f"Candidate constraint violation for {s1_id}: Matched IDs {non_candidate_matches} "
                "were not present in the candidate pairs fed to the model."
            )


def write_submission_tsv(
    data_dict: Dict[str, List[str]],
    entity_order: List[str],
    id_col_name: str,
    output_path: Union[str, Path],
) -> Path:
    """
    Write entity predictions to clean TSV format without quoting:
    source1_entity_id \\t <id_col_name> (comma-separated).

    Args:
        data_dict: Mapping of source1_id -> list of matched/candidate IDs.
        entity_order: Ordered list of source1 IDs.
        id_col_name: Name of second column ('matched_entity_ids' or 'candidate_entity_ids').
        output_path: Destination TSV path.

    Returns:
        Path of written TSV file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1_id in entity_order:
            ids_str = ",".join(data_dict.get(s1_id, []))
            f.write(f"{s1_id}\t{ids_str}\n")

    return dest


def print_inference_summary(
    test_source1_df: pd.DataFrame,
    matching_results: Dict[str, List[str]],
    candidate_pairs: Dict[str, List[str]],
) -> None:
    """
    Print comprehensive statistical breakdown of inference results across countries.
    """
    total_s1 = len(test_source1_df)
    matched_count = sum(1 for s1_id in test_source1_df["entity_id"] if len(matching_results.get(s1_id, [])) > 0)
    singleton_count = total_s1 - matched_count
    total_matches = sum(len(matching_results.get(s1_id, [])) for s1_id in test_source1_df["entity_id"])
    avg_matches = total_matches / total_s1 if total_s1 > 0 else 0.0

    total_candidates = sum(len(candidate_pairs.get(s1_id, [])) for s1_id in test_source1_df["entity_id"])
    avg_candidates = total_candidates / total_s1 if total_s1 > 0 else 0.0

    print("\n" + "=" * 70)
    print("                    INFERENCE SUMMARY & STATISTICS")
    print("=" * 70)
    print(f"Total Test Source 1 Entities : {total_s1}")
    print(f"Entities with >= 1 Match    : {matched_count} ({matched_count / total_s1 * 100:.1f}%)")
    print(f"Singletons (Zero Matches)   : {singleton_count} ({singleton_count / total_s1 * 100:.1f}%)")
    print(f"Total Matches Predicted     : {total_matches}")
    print(f"Average Matches per Entity  : {avg_matches:.2f}")
    print(f"Average Candidates per Entity: {avg_candidates:.2f}")

    # Breakdown by country (open-vocabulary, no hardcoded countries)
    print("\n" + "-" * 70)
    print(f"{'Country':<15} | {'Entities':<10} | {'Matched':<10} | {'Singletons':<12} | {'Match Rate':<12} | {'Avg Matches':<12}")
    print("-" * 70)

    for country, group in test_source1_df.groupby("country"):
        c_entities = group["entity_id"].tolist()
        c_total = len(c_entities)
        c_matched = sum(1 for eid in c_entities if len(matching_results.get(eid, [])) > 0)
        c_singletons = c_total - c_matched
        c_rate = (c_matched / c_total * 100.0) if c_total > 0 else 0.0
        c_matches = sum(len(matching_results.get(eid, [])) for eid in c_entities)
        c_avg = c_matches / c_total if c_total > 0 else 0.0
        print(f"{str(country):<15} | {c_total:<10} | {c_matched:<10} | {c_singletons:<12} | {c_rate:<11.1f}% | {c_avg:<12.2f}")

    print("=" * 70 + "\n")


def run_test_inference(
    test_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    use_embeddings: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Execute full inference pipeline on the test dataset.

    Args:
        test_dir: Directory with test TSV files. Defaults to TEST_PATH.
        output_dir: Directory for outputs. Defaults to OUTPUT_PATH.
        threshold: Matching threshold. Defaults to MATCH_THRESHOLD from config.
        use_embeddings: Whether to compute transformer embeddings. Defaults to USE_EMBEDDINGS.

    Returns:
        Dict containing output paths, counts, and predictions.
    """
    t_start = time.time()
    print("=" * 70)
    print("      BUSINESS ENTITY RESOLUTION - TEST INFERENCE PIPELINE")
    print("=" * 70)

    # Resolve parameters
    t_dir = Path(test_dir) if test_dir else TEST_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUT_PATH
    t_thresh = threshold if threshold is not None else MATCH_THRESHOLD
    embeddings_flag = use_embeddings if use_embeddings is not None else USE_EMBEDDINGS

    print(f"Test Data Directory : {t_dir}")
    print(f"Output Directory    : {out_dir}")
    print(f"Decision Threshold  : {t_thresh:.3f}")
    print(f"Use Embeddings      : {embeddings_flag}")

    # 1. Load Test TSV Files
    print("\n[1/6] Loading test datasets...")
    test_data = load_test_data(t_dir)
    raw_s1 = test_data["test_source1"]
    raw_s2 = test_data["test_source2"]
    raw_s3 = test_data["test_source3"]

    s1_order = raw_s1["entity_id"].astype(str).str.strip().tolist()
    print(f"  Loaded {len(raw_s1)} S1 entities, {len(raw_s2)} S2 entities, {len(raw_s3)} S3 entities.")

    # 2. Text Normalization Pipeline
    print("\n[2/6] Normalizing business names, addresses, and countries...")
    norm_s1 = normalize_dataframe(raw_s1)
    norm_s2 = normalize_dataframe(raw_s2)
    norm_s3 = normalize_dataframe(raw_s3)
    print("  Normalization complete.")

    # 3. Multi-Strategy Blocking Pipeline
    print("\n[3/6] Generating candidate pairs via multi-strategy blocking...")
    candidate_pairs = generate_candidates(
        source1_df=norm_s1,
        source2_df=norm_s2,
        source3_df=norm_s3,
        top_k=BLOCKING_TOP_K,
    )
    # Ensure every S1 entity is represented
    for s1_id in s1_order:
        if s1_id not in candidate_pairs:
            candidate_pairs[s1_id] = []

    total_cands = sum(len(c) for c in candidate_pairs.values())
    print(f"  Generated {total_cands} total candidate pairs across {len(candidate_pairs)} S1 entities.")

    # 4. Pairwise Feature Engineering
    print("\n[4/6] Extracting pairwise feature matrix...")
    if total_cands > 0:
        feat_df = build_feature_matrix(
            candidate_pairs_dict=candidate_pairs,
            source1_df=norm_s1,
            source2_df=norm_s2,
            source3_df=norm_s3,
            use_embeddings=embeddings_flag,
        )
    else:
        feat_df = pd.DataFrame(columns=["source1_entity_id", "candidate_id"])
    print(f"  Constructed feature matrix: {feat_df.shape}")

    # 5. Load Model & Score Candidates
    print("\n[5/6] Loading trained model and scoring candidates...")
    model_path = MODELS_SAVED_DIR / "matcher_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    matching_results: Dict[str, List[str]] = {s1: [] for s1 in s1_order}

    if not feat_df.empty:
        feature_cols = [c for c in feat_df.columns if c not in ["source1_entity_id", "candidate_id"]]
        probs = model.predict_proba(feat_df[feature_cols])[:, 1]
        feat_df = feat_df.copy()
        feat_df["match_probability"] = probs

        # Filter by threshold
        accepted = feat_df[feat_df["match_probability"] >= t_thresh]
        for s1, group in accepted.groupby("source1_entity_id"):
            matching_results[s1] = group["candidate_id"].tolist()

    # 6. Correctness Checks
    print("\n[6/6] Verifying submission integrity constraints...")
    validate_inference_outputs(
        test_source1_df=raw_s1,
        test_source2_df=raw_s2,
        test_source3_df=raw_s3,
        candidate_pairs=candidate_pairs,
        matching_results=matching_results,
    )
    print("  All validation constraints passed successfully (CLEAN).")

    # Write Output Files
    matching_dest = out_dir / "matching_results.tsv"
    candidate_dest = out_dir / "candidate_pairs.tsv"

    write_submission_tsv(
        data_dict=matching_results,
        entity_order=s1_order,
        id_col_name="matched_entity_ids",
        output_path=matching_dest,
    )
    print(f"\nFinal matching results saved to : {matching_dest}")

    write_submission_tsv(
        data_dict=candidate_pairs,
        entity_order=s1_order,
        id_col_name="candidate_entity_ids",
        output_path=candidate_dest,
    )
    print(f"Candidate pairs saved to         : {candidate_dest}")

    # Print Detailed Statistics
    print_inference_summary(
        test_source1_df=raw_s1,
        matching_results=matching_results,
        candidate_pairs=candidate_pairs,
    )

    elapsed = time.time() - t_start
    print(f"Inference pipeline completed in {elapsed:.2f} seconds.")

    return {
        "matching_results": matching_results,
        "candidate_pairs": candidate_pairs,
        "matching_tsv": matching_dest,
        "candidate_tsv": candidate_dest,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    run_test_inference()

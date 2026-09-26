"""
Multi-Strategy Blocking Pipeline for Business Entity Resolution.

Implements three complementary candidate generation strategies:
1. TF-IDF character n-gram nearest neighbors (sparse batch matrix ops)
2. Sorted-neighborhood / prefix blocking (sliding window by country)
3. Inverted index token-overlap blocking (shared keyword matching)

The results are unioned and filtered by open-set country equality to maximize
recall while minimizing candidate pairs (high reduction ratio).
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BLOCKING_TOP_K, OUTPUT_PATH
from src.utils.text_normalize import extract_tokens, normalize_dataframe


def _ensure_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that required normalized columns exist in the DataFrame."""
    required = ["business_name_norm", "business_address_norm", "address_pincode", "has_landmark"]
    if all(col in df.columns for col in required):
        return df
    return normalize_dataframe(df)


def _country_equal(c1: Any, c2: Any) -> bool:
    """
    Open-vocabulary country equality check.
    Works for any country value without hardcoding.
    """
    if c1 is None or c2 is None:
        return False
    s1 = str(c1).strip().lower()
    s2 = str(c2).strip().lower()
    return bool(s1 and s1 == s2)


def blocking_tfidf_neighbors(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
    batch_size: int = 500,
) -> Dict[str, Set[str]]:
    """
    Strategy A: TF-IDF character n-gram nearest neighbors.
    
    Fits TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)) on combined
    business_name_norm + business_address_norm of all candidate records (S2 + S3).
    Computes top-k nearest neighbors via efficient sparse cosine dot products in batches.

    Args:
        s1_df: Source 1 DataFrame.
        cand_df: Combined Source 2 + Source 3 DataFrame.
        top_k: Maximum candidates to retrieve per Source 1 entity.
        batch_size: Query batch size for sparse matrix multiplication.

    Returns:
        Dict mapping source1_entity_id -> set of candidate entity_ids.
    """
    results: Dict[str, Set[str]] = defaultdict(set)
    if s1_df.empty or cand_df.empty:
        return results

    # Construct combined text for TF-IDF representation
    cand_texts = (
        cand_df["business_name_norm"].fillna("")
        + " "
        + cand_df["business_address_norm"].fillna("")
    ).tolist()
    s1_texts = (
        s1_df["business_name_norm"].fillna("")
        + " "
        + s1_df["business_address_norm"].fillna("")
    ).tolist()

    # Fit TF-IDF on candidate pool
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
    )
    X_cand = vectorizer.fit_transform(cand_texts)
    X_s1 = vectorizer.transform(s1_texts)

    s1_ids = s1_df["entity_id"].tolist()
    s1_countries = s1_df["country"].tolist()
    cand_ids = cand_df["entity_id"].values
    cand_countries = cand_df["country"].values

    n_s1 = len(s1_ids)
    n_cand = len(cand_ids)
    effective_k = min(top_k, n_cand)

    # Process in batches to maintain sparse memory efficiency
    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)
        X_s1_batch = X_s1[start_idx:end_idx]

        # Sparse dot product equals cosine similarity since TF-IDF rows are L2-normalized
        sim_batch = X_s1_batch.dot(X_cand.T).toarray()

        for i in range(end_idx - start_idx):
            global_s1_idx = start_idx + i
            s1_id = s1_ids[global_s1_idx]
            s1_country = s1_countries[global_s1_idx]
            sim_scores = sim_batch[i]

            # Filter candidates by country equality before ranking
            country_mask = np.array(
                [_country_equal(s1_country, cc) for cc in cand_countries],
                dtype=bool,
            )
            filtered_indices = np.where(country_mask)[0]

            if len(filtered_indices) == 0:
                continue

            filtered_scores = sim_scores[filtered_indices]
            k_for_query = min(effective_k, len(filtered_indices))

            # Retrieve top-k indices with highest similarity
            if len(filtered_scores) > k_for_query:
                top_part = np.argpartition(filtered_scores, -k_for_query)[-k_for_query:]
                sorted_top = top_part[np.argsort(-filtered_scores[top_part])]
                best_indices = filtered_indices[sorted_top]
            else:
                sorted_all = np.argsort(-filtered_scores)
                best_indices = filtered_indices[sorted_all]

            for idx in best_indices:
                results[s1_id].add(cand_ids[idx])

    return results


def blocking_sorted_neighborhood(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    prefix_len: int = 4,
    window_size: int = 10,
) -> Dict[str, Set[str]]:
    """
    Strategy B: Sorted-neighborhood / prefix blocking.

    Sorts records by the first N characters of business_name_norm and applies a sliding
    window within each country partition to form candidate pairs between S1 and (S2/S3).

    Args:
        s1_df: Source 1 DataFrame.
        cand_df: Combined Source 2 + Source 3 DataFrame.
        prefix_len: Number of characters to extract for prefix sorting.
        window_size: Sliding window size.

    Returns:
        Dict mapping source1_entity_id -> set of candidate entity_ids.
    """
    results: Dict[str, Set[str]] = defaultdict(set)
    if s1_df.empty or cand_df.empty:
        return results

    # Tag records with source type ('S1' vs 'CAND')
    df1 = s1_df[["entity_id", "business_name_norm", "country"]].copy()
    df1["is_s1"] = True

    df_cand = cand_df[["entity_id", "business_name_norm", "country"]].copy()
    df_cand["is_s1"] = False

    combined = pd.concat([df1, df_cand], ignore_index=True)
    combined["sort_key"] = (
        combined["business_name_norm"]
        .fillna("")
        .str.slice(0, prefix_len)
        .str.lower()
    )
    combined["country_clean"] = combined["country"].fillna("").str.strip().str.lower()

    # Group strictly by clean country string
    for _, country_group in combined.groupby("country_clean"):
        if country_group.empty:
            continue
        # Sort by prefix key followed by full normalized name
        sorted_records = country_group.sort_values(
            by=["sort_key", "business_name_norm"]
        ).to_dict("records")
        n_records = len(sorted_records)

        # Sliding window
        for i in range(n_records):
            rec_i = sorted_records[i]
            for j in range(i + 1, min(i + window_size, n_records)):
                rec_j = sorted_records[j]

                # Pair must be strictly between Source 1 and a secondary record
                if rec_i["is_s1"] and not rec_j["is_s1"]:
                    results[rec_i["entity_id"]].add(rec_j["entity_id"])
                elif rec_j["is_s1"] and not rec_i["is_s1"]:
                    results[rec_j["entity_id"]].add(rec_i["entity_id"])

    return results


def blocking_token_overlap(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    min_shared_tokens: int = 2,
) -> Dict[str, Set[str]]:
    """
    Strategy C: Token-overlap inverted index blocking.

    Builds an inverted index mapping tokens to candidate entity IDs from business_name_norm.
    For each Source 1 record, identifies candidate entities sharing at least min_shared_tokens.

    Args:
        s1_df: Source 1 DataFrame.
        cand_df: Combined Source 2 + Source 3 DataFrame.
        min_shared_tokens: Minimum required token overlap.

    Returns:
        Dict mapping source1_entity_id -> set of candidate entity_ids.
    """
    results: Dict[str, Set[str]] = defaultdict(set)
    if s1_df.empty or cand_df.empty:
        return results

    # Build inverted index: token -> list of (entity_id, country)
    inverted_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, row in cand_df.iterrows():
        cand_id = row["entity_id"]
        cand_country = row["country"]
        tokens = extract_tokens(row["business_name_norm"])
        for token in tokens:
            inverted_index[token].append((cand_id, cand_country))

    # Query inverted index for each S1 record
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        s1_country = row["country"]
        s1_tokens = extract_tokens(row["business_name_norm"])

        # Count token co-occurrences per candidate
        candidate_counts: Counter = Counter()
        candidate_countries: Dict[str, str] = {}

        for token in s1_tokens:
            for cand_id, cand_country in inverted_index.get(token, ()):
                candidate_counts[cand_id] += 1
                candidate_countries[cand_id] = cand_country

        # Keep candidates meeting the minimum token threshold with matching country
        for cand_id, count in candidate_counts.items():
            if count >= min_shared_tokens:
                if _country_equal(s1_country, candidate_countries[cand_id]):
                    results[s1_id].add(cand_id)

    return results


def generate_candidates(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    top_k: int = BLOCKING_TOP_K,
    window_size: int = 10,
    min_shared_tokens: int = 2,
) -> Dict[str, List[str]]:
    """
    Generate candidate pairs by unioning all three blocking strategies:
    - TF-IDF character n-gram nearest neighbors
    - Sorted-neighborhood prefix window
    - Token overlap inverted index

    Args:
        source1_df: Source 1 DataFrame.
        source2_df: Source 2 DataFrame.
        source3_df: Source 3 DataFrame.
        top_k: Top-K neighbors for Strategy A.
        window_size: Window size for Strategy B.
        min_shared_tokens: Overlap threshold for Strategy C.

    Returns:
        Dict mapping source1_entity_id -> sorted list of unique candidate_ids.
    """
    s1_norm = _ensure_normalized_columns(source1_df)
    s2_norm = _ensure_normalized_columns(source2_df)
    s3_norm = _ensure_normalized_columns(source3_df)

    cand_combined = pd.concat([s2_norm, s3_norm], ignore_index=True)

    # 1. Strategy A: TF-IDF character n-gram nearest neighbors
    cands_tfidf = blocking_tfidf_neighbors(s1_norm, cand_combined, top_k=top_k)

    # 2. Strategy B: Sorted-neighborhood / prefix window
    cands_sn = blocking_sorted_neighborhood(
        s1_norm, cand_combined, prefix_len=4, window_size=window_size
    )

    # 3. Strategy C: Token-overlap inverted index
    cands_tokens = blocking_token_overlap(
        s1_norm, cand_combined, min_shared_tokens=min_shared_tokens
    )

    # 4. Union & deduplicate per Source 1 entity
    all_s1_ids = s1_norm["entity_id"].tolist()
    final_candidates: Dict[str, List[str]] = {}

    for s1_id in all_s1_ids:
        union_set = (
            cands_tfidf.get(s1_id, set())
            | cands_sn.get(s1_id, set())
            | cands_tokens.get(s1_id, set())
        )
        final_candidates[s1_id] = sorted(list(union_set))

    return final_candidates


def evaluate_blocking(
    candidate_pairs: Dict[str, List[str]],
    ground_truth_df: pd.DataFrame,
    source2_count: int,
    source3_count: int,
) -> Dict[str, float]:
    """
    Compute Blocking Recall, Reduction Ratio, and Average Candidate Counts.

    Args:
        candidate_pairs: Mapping of source1_id -> list of candidate_ids.
        ground_truth_df: Ground truth DataFrame with 'source1_entity_id' and 'matched_entity_ids'.
        source2_count: Total records in Source 2.
        source3_count: Total records in Source 3.

    Returns:
        Dict containing:
        - 'recall': Percentage of ground truth matches captured
        - 'reduction_ratio': 1.0 - (total_candidate_pairs / total_possible_pairs)
        - 'avg_candidates_per_s1': Average number of candidates per S1 entity
        - 'total_candidates': Total candidate pairs generated
        - 'total_true_matches': Total true matches in ground truth
        - 'captured_matches': True matches present in candidate pairs
    """
    total_true_matches = 0
    captured_matches = 0

    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        true_matches = row["matched_entity_ids"]
        if isinstance(true_matches, str):
            true_matches = [m.strip() for m in true_matches.split(",") if m.strip()]

        candidates = set(candidate_pairs.get(s1_id, []))
        for match_id in true_matches:
            total_true_matches += 1
            if match_id in candidates:
                captured_matches += 1

    recall = (captured_matches / total_true_matches * 100.0) if total_true_matches > 0 else 100.0

    n_s1 = len(candidate_pairs)
    total_candidates = sum(len(cands) for cands in candidate_pairs.values())
    total_possible_pairs = n_s1 * (source2_count + source3_count)

    reduction_ratio = (
        (1.0 - (total_candidates / total_possible_pairs))
        if total_possible_pairs > 0
        else 0.0
    )
    avg_candidates = total_candidates / n_s1 if n_s1 > 0 else 0.0

    return {
        "recall": recall,
        "reduction_ratio": reduction_ratio,
        "avg_candidates_per_s1": avg_candidates,
        "total_candidates": total_candidates,
        "total_true_matches": total_true_matches,
        "captured_matches": captured_matches,
    }


def tune_blocking_parameters(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    target_recall: float = 98.0,
) -> Tuple[Dict[str, List[str]], Dict[str, float], Dict[str, Any]]:
    """
    Iteratively tune blocking hyperparameters until target recall is achieved.

    Args:
        source1_df: Source 1 records.
        source2_df: Source 2 records.
        source3_df: Source 3 records.
        ground_truth_df: Ground truth matching records.
        target_recall: Desired recall threshold (default: 98.0%).

    Returns:
        Tuple of (best_candidate_pairs, evaluation_metrics, best_params).
    """
    param_grid = [
        {"top_k": 15, "window_size": 10, "min_shared_tokens": 2},
        {"top_k": 20, "window_size": 12, "min_shared_tokens": 2},
        {"top_k": 25, "window_size": 15, "min_shared_tokens": 1},
        {"top_k": 30, "window_size": 20, "min_shared_tokens": 1},
    ]

    s2_count = len(source2_df)
    s3_count = len(source3_df)

    best_candidates: Dict[str, List[str]] = {}
    best_metrics: Dict[str, float] = {}
    best_params: Dict[str, Any] = param_grid[0]

    for params in param_grid:
        candidates = generate_candidates(
            source1_df,
            source2_df,
            source3_df,
            top_k=params["top_k"],
            window_size=params["window_size"],
            min_shared_tokens=params["min_shared_tokens"],
        )
        metrics = evaluate_blocking(candidates, ground_truth_df, s2_count, s3_count)

        best_candidates = candidates
        best_metrics = metrics
        best_params = params

        if metrics["recall"] >= target_recall:
            break

    return best_candidates, best_metrics, best_params


def save_candidate_pairs(
    candidate_pairs: Dict[str, List[str]],
    output_path: Union[str, Path],
) -> None:
    """
    Save candidate pairs to a TSV file matching the challenge submission specification:
    source1_entity_id \\t candidate_entity_ids (comma-separated).

    Args:
        candidate_pairs: Mapping of source1_entity_id -> list of candidate_ids.
        output_path: Destination TSV file path.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, cands in candidate_pairs.items():
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands),
        })

    df = pd.DataFrame(rows)
    df.to_csv(dest, sep="\t", index=False)

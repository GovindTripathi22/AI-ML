"""
Pairwise Feature Engineering Module for Business Entity Resolution.

Computes multi-attribute similarity features between candidate pairs (Source 1 <-> Source 2/3):
- String distance metrics (Levenshtein, token sort, token set, partial ratio via RapidFuzz)
- Token overlap Jaccard similarities
- Postal/PIN code consistency and landmark co-occurrence
- Open-vocabulary country matching
- Weighted composite similarity
- Optional semantic dense embeddings (via SentenceTransformer 'all-MiniLM-L6-v2')
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from src.config import USE_EMBEDDINGS
from src.utils.text_normalize import extract_tokens, normalize_dataframe

STOPWORDS_ACRONYM = {
    "inc", "ltd", "pvt", "limited", "corp", "corporation", "co", "llc",
    "and", "the", "of", "for", "in", "to", "a", "&", "services", "technologies"
}


def compute_acronym_match(name1: str, name2: str) -> float:
    """
    Check if one business name is an acronym / initialism of the other.
    E.g. TCS <-> Tata Consultancy Services, SBI <-> State Bank of India.
    """
    def _extract_acronym(text: str) -> str:
        words = [w for w in re.findall(r"[a-zA-Z]+", text) if w.lower() not in STOPWORDS_ACRONYM]
        return "".join(w[0].upper() for w in words) if words else ""

    w1 = [w.upper() for w in re.findall(r"[a-zA-Z0-9]+", name1) if w.lower() not in STOPWORDS_ACRONYM]
    w2 = [w.upper() for w in re.findall(r"[a-zA-Z0-9]+", name2) if w.lower() not in STOPWORDS_ACRONYM]
    acr1 = _extract_acronym(name1)
    acr2 = _extract_acronym(name2)
    joined1 = "".join(w1)
    joined2 = "".join(w2)

    if acr2 and len(acr2) >= 2 and (any(tok == acr2 for tok in w1 if len(tok) >= 2) or joined1 == acr2):
        return 1.0
    if acr1 and len(acr1) >= 2 and (any(tok == acr1 for tok in w2 if len(tok) >= 2) or joined2 == acr1):
        return 1.0
    return 0.0


def compute_addr_numeric_similarity(addr1: str, addr2: str) -> float:
    """
    Compute Jaccard similarity of street / building numbers (excluding 5 or 6 digit pincodes).
    Returns 0.5 (neutral) if either address lacks numbers.
    """
    def _get_numbers(addr: str) -> Set[str]:
        return {num for num in re.findall(r"\b\d+\b", addr) if len(num) not in (5, 6)}

    nums1 = _get_numbers(addr1)
    nums2 = _get_numbers(addr2)
    if not nums1 and not nums2:
        return 0.5
    if not nums1 or not nums2:
        return 0.5
    intersection = nums1 & nums2
    union = nums1 | nums2
    return float(len(intersection) / len(union)) if union else 0.5


def compute_phonetic_similarity(name1: str, name2: str) -> float:
    """
    Compute Levenshtein similarity on Metaphone phonetic representations.
    Handles transliterations and alternate spelling variations.
    """
    toks1 = [jellyfish.metaphone(w) for w in re.findall(r"[a-zA-Z]+", name1) if w.lower() not in STOPWORDS_ACRONYM]
    toks2 = [jellyfish.metaphone(w) for w in re.findall(r"[a-zA-Z]+", name2) if w.lower() not in STOPWORDS_ACRONYM]
    if not toks1 or not toks2:
        return 0.0
    s1 = " ".join(toks1)
    s2 = " ".join(toks2)
    return float(fuzz.ratio(s1, s2))


def _safe_str(val: Any) -> str:
    """Convert None / NaN to empty string."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def compute_pair_features(
    row1: Dict[str, Any],
    row2: Dict[str, Any],
    embed_sim: Optional[float] = None,
    tokens1: Optional[Tuple[Set[str], Set[str]]] = None,
    tokens2: Optional[Tuple[Set[str], Set[str]]] = None,
) -> Dict[str, Any]:
    """
    Compute similarity features for a single (source1_record, candidate_record) pair.

    Args:
        row1: Dictionary containing Source 1 record attributes.
        row2: Dictionary containing Candidate record attributes.
        embed_sim: Precomputed cosine similarity between semantic embeddings (optional).
        tokens1: Precomputed tuple of (name_tokens, addr_tokens) for row1 (optional).
        tokens2: Precomputed tuple of (name_tokens, addr_tokens) for row2 (optional).

    Returns:
        Dictionary of computed feature names and numerical values.
    """
    name1 = _safe_str(row1.get("business_name_norm", row1.get("business_name", "")))
    name2 = _safe_str(row2.get("business_name_norm", row2.get("business_name", "")))
    addr1 = _safe_str(row1.get("business_address_norm", row1.get("business_address", "")))
    addr2 = _safe_str(row2.get("business_address_norm", row2.get("business_address", "")))

    # Use precomputed token sets or extract on the fly
    if tokens1 is not None:
        t_n1, t_a1 = tokens1
    else:
        t_n1 = extract_tokens(name1)
        t_a1 = extract_tokens(addr1)

    if tokens2 is not None:
        t_n2, t_a2 = tokens2
    else:
        t_n2 = extract_tokens(name2)
        t_a2 = extract_tokens(addr2)

    # -------------------------------------------------------------
    # 1. Name Similarity Features (via RapidFuzz)
    # -------------------------------------------------------------
    name_ratio = float(fuzz.ratio(name1, name2))
    name_token_sort_ratio = float(fuzz.token_sort_ratio(name1, name2))
    name_token_set_ratio = float(fuzz.token_set_ratio(name1, name2))
    name_partial_ratio = float(fuzz.partial_ratio(name1, name2))

    # Token-set Jaccard similarity
    union_n = t_n1 | t_n2
    name_jaccard = float(len(t_n1 & t_n2) / len(union_n)) if union_n else 0.0

    # First word exact match
    words1 = name1.split()
    words2 = name2.split()
    w1 = words1[0] if words1 else ""
    w2 = words2[0] if words2 else ""
    name_first_word_match = int(bool(w1 and w1 == w2))

    # Length difference and common token counts
    name_length_diff = abs(len(name1) - len(name2))
    name_num_common_tokens = len(t_n1 & t_n2)

    # -------------------------------------------------------------
    # 2. Address Similarity Features
    # -------------------------------------------------------------
    addr_ratio = float(fuzz.ratio(addr1, addr2))
    addr_token_sort_ratio = float(fuzz.token_sort_ratio(addr1, addr2))
    addr_partial_ratio = float(fuzz.partial_ratio(addr1, addr2))

    union_a = t_a1 | t_a2
    addr_jaccard = float(len(t_a1 & t_a2) / len(union_a)) if union_a else 0.0

    # PIN code match: 1 if both non-empty and equal, 0 if mismatch, -1 if either missing
    pin1 = _safe_str(row1.get("address_pincode", ""))
    pin2 = _safe_str(row2.get("address_pincode", ""))
    if pin1 and pin2:
        pincode_match = 1 if pin1 == pin2 else 0
    else:
        pincode_match = -1

    # Landmark co-occurrence flag
    lm1 = bool(row1.get("has_landmark", False))
    lm2 = bool(row2.get("has_landmark", False))
    has_landmark_both = int(lm1 and lm2)

    # -------------------------------------------------------------
    # 3. Cross Features
    # -------------------------------------------------------------
    c1 = _safe_str(row1.get("country", "")).lower()
    c2 = _safe_str(row2.get("country", "")).lower()
    same_country = int(bool(c1 and c1 == c2))

    # Composite weighted similarity (0-100 scale)
    combined_similarity = 0.6 * name_token_set_ratio + 0.4 * addr_token_sort_ratio

    # -------------------------------------------------------------
    # 4. Targeted Features (Phonetic, Numeric Address, Acronyms)
    # -------------------------------------------------------------
    name_phonetic_sim = compute_phonetic_similarity(name1, name2)
    addr_numeric_similarity = compute_addr_numeric_similarity(addr1, addr2)
    name_acronym_match = compute_acronym_match(name1, name2)

    features: Dict[str, Any] = {
        "name_ratio": name_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_jaccard": name_jaccard,
        "name_first_word_match": name_first_word_match,
        "name_length_diff": name_length_diff,
        "name_num_common_tokens": name_num_common_tokens,
        "addr_ratio": addr_ratio,
        "addr_token_sort_ratio": addr_token_sort_ratio,
        "addr_partial_ratio": addr_partial_ratio,
        "addr_jaccard": addr_jaccard,
        "pincode_match": pincode_match,
        "has_landmark_both": has_landmark_both,
        "same_country": same_country,
        "combined_similarity": combined_similarity,
        "name_phonetic_sim": name_phonetic_sim,
        "addr_numeric_similarity": addr_numeric_similarity,
        "name_acronym_match": name_acronym_match,
    }

    # Optional semantic embedding cosine similarity
    if embed_sim is not None:
        features["embedding_cosine_sim"] = float(embed_sim)

    return features


def compute_embeddings_cache(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Precompute and cache L2-normalized dense embeddings per unique entity_id.

    Args:
        df: DataFrame with 'entity_id', 'business_name_norm', and 'business_address_norm'.
        model_name: HuggingFace sentence-transformers model checkpoint.
        batch_size: Batch size for model inference.

    Returns:
        Dict mapping entity_id -> 1D normalized numpy array embedding.
    """
    try:
        import torch  # Ensure torch DLLs load first on Windows
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = (
            df["business_name_norm"].fillna("")
            + " "
            + df["business_address_norm"].fillna("")
        ).tolist()
        entity_ids = df["entity_id"].tolist()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return {eid: emb for eid, emb in zip(entity_ids, embeddings)}
    except Exception as exc:
        print(f"Warning: Semantic embedding extraction unavailable ({exc}). Continuing without embeddings.")
        return {}



def build_feature_matrix(
    candidate_pairs_dict: Dict[str, List[str]],
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    use_embeddings: Optional[bool] = None,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a pairwise feature matrix for all (source1_id, candidate_id) pairs.

    Args:
        candidate_pairs_dict: Mapping of source1_id -> list of candidate_ids.
        source1_df: Source 1 DataFrame.
        source2_df: Source 2 DataFrame.
        source3_df: Source 3 DataFrame.
        use_embeddings: Whether to compute dense embedding similarities.
            Defaults to src.config.USE_EMBEDDINGS if None.
        max_pairs: Optional maximum number of pairs to compute (e.g. for sampling/testing).

    Returns:
        DataFrame containing 'source1_entity_id', 'candidate_id', and all computed features.
    """
    should_embed = USE_EMBEDDINGS if use_embeddings is None else use_embeddings

    # Ensure required normalized columns exist
    from src.blocking.blocker import _ensure_normalized_columns

    s1_norm = _ensure_normalized_columns(source1_df)
    s2_norm = _ensure_normalized_columns(source2_df)
    s3_norm = _ensure_normalized_columns(source3_df)

    # Fast indexed record lookups by entity_id
    combined_cand = pd.concat([s2_norm, s3_norm], ignore_index=True)
    s1_records: Dict[str, Dict[str, Any]] = s1_norm.set_index("entity_id").to_dict("index")
    cand_records: Dict[str, Dict[str, Any]] = combined_cand.set_index("entity_id").to_dict("index")

    # Pre-extract token sets once per entity
    s1_tokens: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for eid, rec in s1_records.items():
        s1_tokens[eid] = (
            extract_tokens(rec.get("business_name_norm", "")),
            extract_tokens(rec.get("business_address_norm", "")),
        )

    cand_tokens: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for eid, rec in cand_records.items():
        cand_tokens[eid] = (
            extract_tokens(rec.get("business_name_norm", "")),
            extract_tokens(rec.get("business_address_norm", "")),
        )

    # Optional embedding cache
    embed_cache: Dict[str, np.ndarray] = {}
    if should_embed:
        print("Computing dense embeddings for unique entities...")
        all_unique_df = pd.concat([s1_norm, combined_cand], ignore_index=True).drop_duplicates(
            subset=["entity_id"]
        )
        embed_cache = compute_embeddings_cache(all_unique_df)

    # Assemble pairs list
    pair_list: List[Tuple[str, str]] = []
    for s1_id, cands in candidate_pairs_dict.items():
        if s1_id not in s1_records:
            continue
        for cid in cands:
            if cid in cand_records:
                pair_list.append((s1_id, cid))
                if max_pairs and len(pair_list) >= max_pairs:
                    break
        if max_pairs and len(pair_list) >= max_pairs:
            break

    # Compute features with progress bar
    rows: List[Dict[str, Any]] = []
    for s1_id, cid in tqdm(pair_list, desc="Extracting pairwise features"):
        r1 = s1_records[s1_id]
        r2 = cand_records[cid]
        tok1 = s1_tokens[s1_id]
        tok2 = cand_tokens[cid]

        embed_sim = None
        if should_embed and s1_id in embed_cache and cid in embed_cache:
            embed_sim = float(np.dot(embed_cache[s1_id], embed_cache[cid]))

        feat = compute_pair_features(
            r1,
            r2,
            embed_sim=embed_sim,
            tokens1=tok1,
            tokens2=tok2,
        )
        feat["source1_entity_id"] = s1_id
        feat["candidate_id"] = cid
        rows.append(feat)

    feature_df = pd.DataFrame(rows)

    # Reorder columns so entity IDs appear first
    if not feature_df.empty:
        id_cols = ["source1_entity_id", "candidate_id"]
        other_cols = [c for c in feature_df.columns if c not in id_cols]
        feature_df = feature_df[id_cols + other_cols]

    return feature_df

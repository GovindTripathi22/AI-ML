"""
Unit tests for blocking pipeline (src/blocking/blocker.py).
"""

import pandas as pd
import pytest
from src.blocking.blocker import (
    _country_equal,
    blocking_sorted_neighborhood,
    blocking_tfidf_neighbors,
    blocking_token_overlap,
    evaluate_blocking,
    generate_candidates,
)


@pytest.fixture
def sample_dfs():
    s1 = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "Tata Motors Ltd",
            "business_name_norm": "tata motors limited",
            "business_address": "Bombay House, Mumbai 400001",
            "business_address_norm": "bombay house mumbai",
            "country": "India",
        },
        {
            "entity_id": "S1-002",
            "business_name": "Apple Inc.",
            "business_name_norm": "apple incorporated",
            "business_address": "1 Infinite Loop, Cupertino 95014",
            "business_address_norm": "1 infinite loop cupertino",
            "country": "US",
        },
    ])

    s2 = pd.DataFrame([
        {
            "entity_id": "S2-001",
            "business_name": "Tata Motors",
            "business_name_norm": "tata motors",
            "business_address": "Bombay House, Mumbai",
            "business_address_norm": "bombay house mumbai",
            "country": "India",
        },
        {
            "entity_id": "S2-002",
            "business_name": "Apple Computer Co",
            "business_name_norm": "apple computer company",
            "business_address": "1 Infinite Loop, Cupertino",
            "business_address_norm": "1 infinite loop cupertino",
            "country": "US",
        },
    ])

    s3 = pd.DataFrame([
        {
            "entity_id": "S3-001",
            "business_name": "Tata Motors Commercial",
            "business_name_norm": "tata motors commercial",
            "business_address": "Mumbai 400001",
            "business_address_norm": "mumbai",
            "country": "India",
        },
        {
            "entity_id": "S3-002",
            "business_name": "Renault SAS",
            "business_name_norm": "renault sas",
            "business_address": "Boulogne-Billancourt",
            "business_address_norm": "boulogne billancourt",
            "country": "France",
        },
    ])

    return s1, s2, s3


def test_country_equal_open_vocabulary():
    assert _country_equal("US", "us") is True
    assert _country_equal("India", " INDIA ") is True
    assert _country_equal("France", "france") is True
    assert _country_equal("Germany", "germany") is True
    assert _country_equal("US", "India") is False
    assert _country_equal("France", "US") is False
    assert _country_equal(None, "US") is False


def test_blocking_tfidf_and_country_filtering(sample_dfs):
    s1, s2, s3 = sample_dfs
    cand_df = pd.concat([s2, s3], ignore_index=True)
    cands = blocking_tfidf_neighbors(s1, cand_df, top_k=5)

    # S1-001 is India -> candidates must only be from India
    for cid in cands["S1-001"]:
        cand_country = cand_df.loc[cand_df["entity_id"] == cid, "country"].iloc[0]
        assert cand_country == "India"
    assert "S2-001" in cands["S1-001"]

    # S1-002 is US -> candidates must only be from US
    for cid in cands["S1-002"]:
        cand_country = cand_df.loc[cand_df["entity_id"] == cid, "country"].iloc[0]
        assert cand_country == "US"
    assert "S2-002" in cands["S1-002"]


def test_blocking_sorted_neighborhood(sample_dfs):
    s1, s2, s3 = sample_dfs
    cand_df = pd.concat([s2, s3], ignore_index=True)
    cands = blocking_sorted_neighborhood(s1, cand_df, prefix_len=4, window_size=5)

    # Tata Motors S1 and S2 share prefix 'tata'
    assert "S2-001" in cands["S1-001"]
    # Apple S1 and S2 share prefix 'appl'
    assert "S2-002" in cands["S1-002"]


def test_blocking_token_overlap(sample_dfs):
    s1, s2, s3 = sample_dfs
    cand_df = pd.concat([s2, s3], ignore_index=True)
    cands = blocking_token_overlap(s1, cand_df, min_shared_tokens=2)

    # 'tata' and 'motors' match
    assert "S2-001" in cands["S1-001"]
    # French record S3-002 should never appear in US or India candidates
    assert "S3-002" not in cands.get("S1-001", set())
    assert "S3-002" not in cands.get("S1-002", set())


def test_generate_candidates_union(sample_dfs):
    s1, s2, s3 = sample_dfs
    all_cands = generate_candidates(s1, s2, s3, top_k=5)

    assert "S1-001" in all_cands
    assert "S1-002" in all_cands
    assert "S2-001" in all_cands["S1-001"]
    assert "S2-002" in all_cands["S1-002"]


def test_evaluate_blocking():
    cands = {
        "S1-001": ["S2-001", "S3-001"],
        "S1-002": ["S2-002"],
    }
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": ["S2-001"]},
        {"source1_entity_id": "S1-002", "matched_entity_ids": ["S2-002", "S3-002"]},
    ])

    metrics = evaluate_blocking(cands, gt, source2_count=2, source3_count=2)
    # Ground truth has 3 matches: S1-001 -> S2-001 (captured), S1-002 -> S2-002 (captured), S1-002 -> S3-002 (missed)
    assert metrics["captured_matches"] == 2
    assert metrics["total_true_matches"] == 3
    assert abs(metrics["recall"] - (2 / 3 * 100)) < 1e-4
    assert metrics["total_candidates"] == 3

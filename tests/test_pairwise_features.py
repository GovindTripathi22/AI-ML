"""
Unit tests for feature engineering module (src/features/pairwise_features.py).
"""

import pandas as pd
import pytest
from src.features.pairwise_features import (
    build_feature_matrix,
    compute_pair_features,
)


@pytest.fixture
def sample_pair():
    r1 = {
        "entity_id": "S1-001",
        "business_name": "Tata Motors Ltd",
        "business_name_norm": "tata motors limited",
        "business_address": "Bombay House, Mumbai 400001",
        "business_address_norm": "bombay house mumbai",
        "address_pincode": "400001",
        "has_landmark": False,
        "country": "India",
    }
    r2 = {
        "entity_id": "S2-001",
        "business_name": "Tata Motors",
        "business_name_norm": "tata motors",
        "business_address": "Bombay House, Near CST, Mumbai 400001",
        "business_address_norm": "bombay house mumbai",
        "address_pincode": "400001",
        "has_landmark": True,
        "country": "India",
    }
    return r1, r2


def test_compute_pair_features_values(sample_pair):
    r1, r2 = sample_pair
    feats = compute_pair_features(r1, r2)

    # Name features
    assert feats["name_token_set_ratio"] == 100.0
    assert feats["name_first_word_match"] == 1
    assert feats["name_num_common_tokens"] == 2
    assert feats["name_jaccard"] > 0.0

    # Address features
    assert feats["addr_ratio"] == 100.0
    assert feats["pincode_match"] == 1
    assert feats["has_landmark_both"] == 0  # r1 does not have landmark

    # Cross features
    assert feats["same_country"] == 1
    assert feats["combined_similarity"] > 80.0


def test_pincode_match_conditions():
    # Both have matching pincodes -> 1
    f1 = compute_pair_features({"address_pincode": "10001"}, {"address_pincode": "10001"})
    assert f1["pincode_match"] == 1

    # Both have differing pincodes -> 0
    f2 = compute_pair_features({"address_pincode": "10001"}, {"address_pincode": "90210"})
    assert f2["pincode_match"] == 0

    # One or both missing -> -1
    f3 = compute_pair_features({"address_pincode": "10001"}, {"address_pincode": ""})
    assert f3["pincode_match"] == -1

    f4 = compute_pair_features({"address_pincode": None}, {"address_pincode": None})
    assert f4["pincode_match"] == -1


def test_build_feature_matrix():
    s1 = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "Acme Corp",
            "business_name_norm": "acme corporation",
            "business_address": "123 Main St",
            "business_address_norm": "123 main street",
            "address_pincode": "10001",
            "has_landmark": False,
            "country": "US",
        }
    ])
    s2 = pd.DataFrame([
        {
            "entity_id": "S2-001",
            "business_name": "Acme Inc",
            "business_name_norm": "acme incorporated",
            "business_address": "123 Main Street",
            "business_address_norm": "123 main street",
            "address_pincode": "10001",
            "has_landmark": False,
            "country": "US",
        }
    ])
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    cands = {"S1-001": ["S2-001"]}
    feat_df = build_feature_matrix(cands, s1, s2, s3, use_embeddings=False)

    assert len(feat_df) == 1
    assert feat_df.iloc[0]["source1_entity_id"] == "S1-001"
    assert feat_df.iloc[0]["candidate_id"] == "S2-001"
    assert feat_df.iloc[0]["same_country"] == 1
    assert feat_df.iloc[0]["pincode_match"] == 1
    assert "combined_similarity" in feat_df.columns

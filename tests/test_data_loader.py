"""
Unit tests for data loader module (src/utils/data_loader.py).
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.utils.data_loader import (
    load_ground_truth_file,
    load_source_file,
    load_training_data,
    parse_matched_entity_ids,
    validate_ground_truth_columns,
    validate_source_columns,
)


class TestDataLoader:
    """Test data loader and validation functions."""

    def test_parse_matched_entity_ids(self):
        assert parse_matched_entity_ids("") == []
        assert parse_matched_entity_ids("   ") == []
        assert parse_matched_entity_ids(None) == []
        assert parse_matched_entity_ids(float("nan")) == []
        assert parse_matched_entity_ids("S2-00001") == ["S2-00001"]
        assert parse_matched_entity_ids("S2-00001, S3-00002") == ["S2-00001", "S3-00002"]
        assert parse_matched_entity_ids("S2-00001,S3-00002,S2-00003") == ["S2-00001", "S3-00002", "S2-00003"]

    def test_validate_source_columns(self):
        valid_df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        validate_source_columns(valid_df, "test.tsv")  # Should not raise

        invalid_df = pd.DataFrame(columns=["id", "name", "address"])
        with pytest.raises(ValueError, match="Invalid columns in bad.tsv"):
            validate_source_columns(invalid_df, "bad.tsv")

    def test_validate_ground_truth_columns(self):
        valid_df = pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
        validate_ground_truth_columns(valid_df, "gt.tsv")  # Should not raise

        invalid_df = pd.DataFrame(columns=["source1", "matches"])
        with pytest.raises(ValueError, match="Invalid columns in ground truth file bad_gt.tsv"):
            validate_ground_truth_columns(invalid_df, "bad_gt.tsv")

    def test_load_source_and_gt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create sample source TSV
            src_file = tmp_path / "source1.tsv"
            src_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-001\tAcme Corp\t123 Main St\tUS\n")

            df_src = load_source_file(src_file)
            assert len(df_src) == 1
            assert df_src.iloc[0]["entity_id"] == "S1-001"

            # Create sample ground truth TSV
            gt_file = tmp_path / "train_ground_truth.tsv"
            gt_file.write_text("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001,S3-001\nS1-002\t\n")

            df_gt = load_ground_truth_file(gt_file)
            assert len(df_gt) == 2
            assert df_gt.iloc[0]["matched_entity_ids"] == ["S2-001", "S3-001"]
            assert df_gt.iloc[1]["matched_entity_ids"] == []

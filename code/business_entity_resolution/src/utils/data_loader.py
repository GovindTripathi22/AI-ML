"""
Data Loader Utility for Business Entity Resolution Pipeline.

Loads, validates, and parses tabular TSV datasets for training and test sources.
All TSV files are loaded explicitly using sep="\\t".
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.config import TEST_PATH, TRAIN_PATH

# -----------------------------------------------------------------------------
# Column Specifications
# -----------------------------------------------------------------------------
EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def parse_matched_entity_ids(val: Any) -> List[str]:
    """
    Parse a comma-separated matched_entity_ids string into a Python list of IDs.
    
    Handles empty strings, whitespace, None, and NaN values by returning an empty list.

    Args:
        val: Raw value from the matched_entity_ids column.

    Returns:
        List of parsed entity ID strings.
    """
    if val is None or pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    val_str = str(val).strip()
    if not val_str:
        return []
    return [item.strip() for item in val_str.split(",") if item.strip()]


def validate_source_columns(df: pd.DataFrame, filename: str) -> None:
    """
    Validate that a source DataFrame has exactly the required columns.

    Args:
        df: Loaded DataFrame.
        filename: Name of the file for error reporting.

    Raises:
        ValueError: If columns do not match EXPECTED_SOURCE_COLUMNS.
    """
    actual_cols = list(df.columns)
    if actual_cols != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            f"Invalid columns in {filename}. "
            f"Expected exactly: {EXPECTED_SOURCE_COLUMNS}, but got: {actual_cols}"
        )


def validate_ground_truth_columns(df: pd.DataFrame, filename: str) -> None:
    """
    Validate that a ground truth DataFrame has the required columns.

    Args:
        df: Loaded ground truth DataFrame.
        filename: Name of the file for error reporting.

    Raises:
        ValueError: If columns do not match EXPECTED_GROUND_TRUTH_COLUMNS.
    """
    actual_cols = list(df.columns)
    if actual_cols != EXPECTED_GROUND_TRUTH_COLUMNS:
        raise ValueError(
            f"Invalid columns in ground truth file {filename}. "
            f"Expected exactly: {EXPECTED_GROUND_TRUTH_COLUMNS}, but got: {actual_cols}"
        )


def load_source_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single entity source TSV file using sep="\\t" and validate columns.

    Args:
        file_path: Path to the TSV file.

    Returns:
        Validated pandas DataFrame.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    # Explicitly use sep="\t", never default
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    validate_source_columns(df, path.name)
    return df


def load_ground_truth_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a ground truth TSV file using sep="\\t", validate columns, and parse matched_entity_ids.

    Args:
        file_path: Path to the ground truth TSV file.

    Returns:
        Validated pandas DataFrame with matched_entity_ids as List[str].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    # Explicitly use sep="\t", never default
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    validate_ground_truth_columns(df, path.name)

    # Parse comma-separated IDs into a Python list
    df["matched_entity_ids"] = df["matched_entity_ids"].apply(parse_matched_entity_ids)
    return df


def load_training_data(
    train_dir: Optional[Union[str, Path]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all training TSV files from the training directory.

    Expected files:
    - train_source1.tsv
    - train_source2.tsv
    - train_source3.tsv
    - train_ground_truth.tsv (or ground_truth.tsv)

    Args:
        train_dir: Directory containing training TSV files. Defaults to TRAIN_PATH.

    Returns:
        Dictionary mapping file keys to validated DataFrames:
        {
            'train_source1': DataFrame,
            'train_source2': DataFrame,
            'train_source3': DataFrame,
            'train_ground_truth': DataFrame
        }
    """
    base_dir = Path(train_dir) if train_dir else TRAIN_PATH
    data: Dict[str, pd.DataFrame] = {}

    for i in (1, 2, 3):
        key = f"train_source{i}"
        fpath = base_dir / f"{key}.tsv"
        data[key] = load_source_file(fpath)

    # Check for train_ground_truth.tsv or ground_truth.tsv
    gt_file = base_dir / "train_ground_truth.tsv"
    if not gt_file.is_file():
        gt_alt = base_dir / "ground_truth.tsv"
        if gt_alt.is_file():
            gt_file = gt_alt
        else:
            raise FileNotFoundError(
                f"Ground truth file not found in {base_dir} (checked train_ground_truth.tsv and ground_truth.tsv)"
            )

    data["train_ground_truth"] = load_ground_truth_file(gt_file)
    return data


def load_test_data(
    test_dir: Optional[Union[str, Path]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all test TSV files from the test directory.

    Expected files:
    - test_source1.tsv
    - test_source2.tsv
    - test_source3.tsv

    Args:
        test_dir: Directory containing test TSV files. Defaults to TEST_PATH.

    Returns:
        Dictionary mapping file keys to validated DataFrames:
        {
            'test_source1': DataFrame,
            'test_source2': DataFrame,
            'test_source3': DataFrame
        }
    """
    base_dir = Path(test_dir) if test_dir else TEST_PATH
    data: Dict[str, pd.DataFrame] = {}

    for i in (1, 2, 3):
        key = f"test_source{i}"
        fpath = base_dir / f"{key}.tsv"
        data[key] = load_source_file(fpath)

    return data


def load_all_data(
    train_dir: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all training and test TSV files.

    Args:
        train_dir: Directory containing training TSVs. Defaults to TRAIN_PATH.
        test_dir: Directory containing test TSVs. Defaults to TEST_PATH.

    Returns:
        Dictionary mapping dataset keys to validated DataFrames.
    """
    all_data: Dict[str, pd.DataFrame] = {}
    all_data.update(load_training_data(train_dir))
    all_data.update(load_test_data(test_dir))
    return all_data

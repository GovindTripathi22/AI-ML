"""
Submission Validator for Business Entity Resolution Challenge.

Validates matching_results.tsv and candidate_pairs.tsv against test data:
- File existence and non-emptiness
- Proper TSV structure (tab-separated, exact required headers)
- Complete 1-to-1 coverage of all test Source 1 entities
- No duplicate entities or duplicate candidate IDs
- No self-matches
- Candidate IDs belong strictly to Source 2 / Source 3 test sets
- Matching results are a strict subset of candidate pairs for every entity
"""

import argparse
from pathlib import Path
import sys
from typing import List, Set
import pandas as pd


def parse_id_list(value: str) -> List[str]:
    """Parse comma-separated IDs, handling empty or whitespace strings."""
    if pd.isna(value) or value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def validate_submission(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> bool:
    """
    Validate submission files. Returns True if PASS, raises ValueError on failure.
    """
    print(f"Validating matching file  : {matching_path}")
    print(f"Validating candidate file : {candidate_path}")
    print(f"Test data directory       : {test_dir}")
    print("-" * 60)

    # 1. Check file existence
    if not matching_path.is_file():
        raise FileNotFoundError(f"Matching results file not found: {matching_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate pairs file not found: {candidate_path}")

    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    for f in [s1_file, s2_file, s3_file]:
        if not f.is_file():
            raise FileNotFoundError(f"Required test file not found: {f}")

    # 2. Load ground test sets
    s1_df = pd.read_csv(s1_file, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(s2_file, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(s3_file, sep="\t", dtype=str, keep_default_na=False)

    s1_expected_ids = s1_df["entity_id"].str.strip().tolist()
    s2_ids: Set[str] = set(s2_df["entity_id"].str.strip())
    s3_ids: Set[str] = set(s3_df["entity_id"].str.strip())
    valid_target_ids = s2_ids | s3_ids

    print(f"Loaded test entities: {len(s1_expected_ids)} S1, {len(s2_ids)} S2, {len(s3_ids)} S3")

    # 3. Validate matching_results.tsv headers & formatting
    with open(matching_path, "r", encoding="utf-8") as f:
        match_header_line = f.readline().rstrip("\r\n")
    match_headers = match_header_line.split("\t")
    if match_headers != ["source1_entity_id", "matched_entity_ids"]:
        raise ValueError(
            f"Invalid headers in {matching_path.name}. "
            f"Expected exactly ['source1_entity_id', 'matched_entity_ids'], but got: {match_headers}"
        )

    match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)

    # 4. Validate candidate_pairs.tsv headers & formatting
    with open(candidate_path, "r", encoding="utf-8") as f:
        cand_header_line = f.readline().rstrip("\r\n")
    cand_headers = cand_header_line.split("\t")
    if cand_headers != ["source1_entity_id", "candidate_entity_ids"]:
        raise ValueError(
            f"Invalid headers in {candidate_path.name}. "
            f"Expected exactly ['source1_entity_id', 'candidate_entity_ids'], but got: {cand_headers}"
        )

    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)

    # 5. Row count and 1-to-1 entity coverage checks
    match_s1_ids = match_df["source1_entity_id"].str.strip().tolist()
    cand_s1_ids = cand_df["source1_entity_id"].str.strip().tolist()

    if len(match_s1_ids) != len(s1_expected_ids):
        raise ValueError(
            f"Row count mismatch in {matching_path.name}: "
            f"Expected {len(s1_expected_ids)} rows, but got {len(match_s1_ids)}"
        )
    if len(cand_s1_ids) != len(s1_expected_ids):
        raise ValueError(
            f"Row count mismatch in {candidate_path.name}: "
            f"Expected {len(s1_expected_ids)} rows, but got {len(cand_s1_ids)}"
        )

    if set(match_s1_ids) != set(s1_expected_ids):
        missing = set(s1_expected_ids) - set(match_s1_ids)
        extra = set(match_s1_ids) - set(s1_expected_ids)
        raise ValueError(f"Matching file entity mismatch. Missing: {missing}, Extra: {extra}")

    if set(cand_s1_ids) != set(s1_expected_ids):
        missing = set(s1_expected_ids) - set(cand_s1_ids)
        extra = set(cand_s1_ids) - set(s1_expected_ids)
        raise ValueError(f"Candidate file entity mismatch. Missing: {missing}, Extra: {extra}")

    if len(match_s1_ids) != len(set(match_s1_ids)):
        raise ValueError(f"Duplicate source1_entity_id rows found in {matching_path.name}")
    if len(cand_s1_ids) != len(set(cand_s1_ids)):
        raise ValueError(f"Duplicate source1_entity_id rows found in {candidate_path.name}")

    # Build entity lookups
    match_dict = {
        row["source1_entity_id"].strip(): parse_id_list(row["matched_entity_ids"])
        for _, row in match_df.iterrows()
    }
    cand_dict = {
        row["source1_entity_id"].strip(): parse_id_list(row["candidate_entity_ids"])
        for _, row in cand_df.iterrows()
    }

    # 6. Detailed entity-level consistency checks
    for s1_id in s1_expected_ids:
        matches = match_dict[s1_id]
        cands = cand_dict[s1_id]

        # Check duplicate IDs within entity
        if len(matches) != len(set(matches)):
            raise ValueError(f"Duplicate IDs in matched list for {s1_id}: {matches}")
        if len(cands) != len(set(cands)):
            raise ValueError(f"Duplicate IDs in candidate list for {s1_id}: {cands}")

        # Check self-matches
        if s1_id in matches:
            raise ValueError(f"Self-match detected in matched list for {s1_id}")
        if s1_id in cands:
            raise ValueError(f"Self-match detected in candidate list for {s1_id}")

        # Check valid candidate target IDs (must be in Source 2 or Source 3)
        invalid_m = set(matches) - valid_target_ids
        if invalid_m:
            raise ValueError(f"Invalid target IDs in matches for {s1_id}: {invalid_m}")

        invalid_c = set(cands) - valid_target_ids
        if invalid_c:
            raise ValueError(f"Invalid target IDs in candidates for {s1_id}: {invalid_c}")

        # Matched IDs must be a subset of candidate IDs
        non_cand_matches = set(matches) - set(cands)
        if non_cand_matches:
            raise ValueError(
                f"Candidate constraint violation for {s1_id}: "
                f"Matches {non_cand_matches} were not included in candidates {cands}"
            )

    print("All structural, integrity, and candidate subset constraints verified.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate challenge submission TSV files.")
    parser.add_argument("--matching", type=str, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test dataset directory")

    args = parser.parse_args()

    try:
        validate_submission(
            matching_path=Path(args.matching),
            candidate_path=Path(args.candidate),
            test_dir=Path(args.test_dir),
        )
        print("-" * 60)
        print("PASS")
        sys.exit(0)
    except Exception as exc:
        print("-" * 60)
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

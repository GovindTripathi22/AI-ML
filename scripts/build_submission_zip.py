"""
Submission Archive Builder and Sanity Verification Script.

Assembles the final challenge submission ZIP archive matching the required layout:
    <team_name>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/src/  (all source code)
    ├── code/business_entity_resolution/README.md
    ├── code/business_entity_resolution/requirements.txt
    └── Documentation_template.md

Extracts the generated zip into a temporary directory and executes the challenge
submission validator to guarantee 100% submission compliance.
"""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.validate_submission import validate_submission


def build_submission_zip(
    team_name: str = "DocGuru",
    output_dir: Path = PROJECT_ROOT,
) -> Path:
    """
    Build the submission ZIP archive from the current project repository.

    Args:
        team_name: Identifier for the team submission filename.
        output_dir: Directory where the zip file will be written.

    Returns:
        Path to the created ZIP file.
    """
    zip_filename = f"{team_name}_submission.zip"
    zip_path = output_dir / zip_filename

    # Source files
    matching_tsv = PROJECT_ROOT / "output" / "matching_results.tsv"
    candidate_tsv = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    src_dir = PROJECT_ROOT / "src"
    readme_file = PROJECT_ROOT / "code" / "business_entity_resolution" / "README.md"
    req_file = PROJECT_ROOT / "code" / "business_entity_resolution" / "requirements.txt"
    doc_file = PROJECT_ROOT / "Documentation_template.md"

    # Pre-flight check
    required_files = [matching_tsv, candidate_tsv, readme_file, req_file, doc_file]
    for rf in required_files:
        if not rf.is_file():
            raise FileNotFoundError(f"Missing required submission file: {rf}")

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source code directory not found: {src_dir}")

    print("=" * 70)
    print(f"       BUILDING SUBMISSION ARCHIVE: {zip_filename}")
    print("=" * 70)

    # Build ZIP archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output files
        print("Adding submission outputs...")
        zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        print("  + output/matching_results.tsv")
        zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")
        print("  + output/candidate_pairs.tsv")

        # 2. Source code (exclude __pycache__ and .pyc)
        print("Adding source code...")
        for file in src_dir.rglob("*"):
            if file.is_file():
                if "__pycache__" in file.parts or file.suffix == ".pyc":
                    continue
                rel_to_src = file.relative_to(src_dir)
                arcname = Path("code/business_entity_resolution/src") / rel_to_src
                zipf.write(file, arcname=str(arcname).replace("\\", "/"))
                print(f"  + {str(arcname).replace('\\', '/')}")

        # 3. Code documentation & requirements
        print("Adding code README & requirements...")
        zipf.write(readme_file, arcname="code/business_entity_resolution/README.md")
        print("  + code/business_entity_resolution/README.md")
        zipf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")
        print("  + code/business_entity_resolution/requirements.txt")

        # 4. Root documentation template
        print("Adding Documentation_template.md...")
        zipf.write(doc_file, arcname="Documentation_template.md")
        print("  + Documentation_template.md")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nZIP file successfully created at: {zip_path}")
    print(f"Total size: {zip_size_mb:.2f} MB ({zip_path.stat().st_size:,} bytes)")
    print("-" * 70)

    # Verification / Sanity Check: Unzip into temp dir and validate
    print("Performing sanity check on packaged archive...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(temp_path)

        unzipped_matching = temp_path / "output" / "matching_results.tsv"
        unzipped_candidate = temp_path / "output" / "candidate_pairs.tsv"
        test_dir = PROJECT_ROOT / "dataset" / "test"

        # Run validator
        validate_submission(
            matching_path=unzipped_matching,
            candidate_path=unzipped_candidate,
            test_dir=test_dir,
        )

    print("=" * 70)
    print(f"SANITY CHECK PASSED: Archive {zip_filename} is 100% compliant!")
    print(f"Final Zip Location : {zip_path.resolve()}")
    print("Status             : PASS (exit code 0)")
    print("=" * 70)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Create challenge submission zip file.")
    parser.add_argument("--team-name", type=str, default="DocGuru", help="Team name for zip file")
    args = parser.parse_args()

    try:
        build_submission_zip(team_name=args.team_name)
        sys.exit(0)
    except Exception as exc:
        print(f"Error creating submission zip: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

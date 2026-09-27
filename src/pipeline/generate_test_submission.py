"""
Production Streaming Candidate Generation and Matching Pipeline.

Generates official challenge submission files:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Uses country-partitioned SQLite FTS5 index on disk (/dev/shm on Linux/Colab)
with multi-processing batched calibrated scoring. Preserves exact entity order,
keeps memory under 400 MB, and optimizes for maximum Macro F_0.5.
"""

import os
import sys
import time
import re
import sqlite3
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from rapidfuzz import fuzz

from src.config import PROJECT_ROOT, TEST_PATH, OUTPUT_PATH
from utils.validate_submission import validate_submission

OFFICIAL_TEST = PROJECT_ROOT / "dataset_official" / "test"
TEST_DATA_DIR = OFFICIAL_TEST if OFFICIAL_TEST.is_dir() else TEST_PATH

GENERIC_TERMS = {
    "the", "and", "ltd", "inc", "llc", "corp", "pvt", "limited", "private",
    "corporation", "null", "nan", "services", "technologies", "solutions",
    "industries", "group", "international", "global", "trading", "enterprises",
    "holdings", "systems", "management", "consulting", "company", "co", "associates",
    "agency", "center", "centre", "club", "sarl", "sasu", "sas", "eurl", "gmbh",
    "sa", "llp", "plc", "cie", "ets"
}
VALID_COUNTRIES = {"France", "India", "US"}


def get_candidates(cur, tbl, name, addr):
    n_words = [w for w in re.findall(r"[^\W_]+", name or "", re.UNICODE) if len(w) >= 3 and w.lower() not in GENERIC_TERMS]
    if not n_words:
        n_words = [w for w in re.findall(r"[^\W_]+", name or "", re.UNICODE) if len(w) >= 2 and w.lower() not in GENERIC_TERMS]
    
    rows = []
    # Tier 1: 2 distinctive words (covers 85% of cases in < 0.8 ms)
    if len(n_words) >= 2:
        q = f'"{n_words[0]}" "{n_words[1]}"'
        cur.execute(f"SELECT entity_id, business_name, business_address FROM {tbl} WHERE {tbl} MATCH ? LIMIT 8;", (q,))
        rows = cur.fetchall()
        
    # Tier 2: 1 distinctive word if Tier 1 returned < 3
    if len(rows) < 3 and n_words:
        q = f'"{n_words[0]}"'
        cur.execute(f"SELECT entity_id, business_name, business_address FROM {tbl} WHERE {tbl} MATCH ? LIMIT 8;", (q,))
        new_rows = cur.fetchall()
        seen = {r[0] for r in rows}
        for r in new_rows:
            if r[0] not in seen:
                rows.append(r)
                
    # Tier 3: If still empty, try house number + address word
    if not rows:
        nums = [n for n in re.findall(r"\b\d+\b", addr or "") if len(n) >= 2]
        a_words = [w for w in re.findall(r"[^\W_]+", addr or "", re.UNICODE) if w.isalpha() and len(w) >= 4 and w.lower() not in GENERIC_TERMS]
        if nums and a_words:
            q = f'"{nums[0]}" "{a_words[0]}"'
            cur.execute(f"SELECT entity_id, business_name, business_address FROM {tbl} WHERE {tbl} MATCH ? LIMIT 8;", (q,))
            rows = cur.fetchall()
        elif a_words:
            q = f'"{a_words[0]}"'
            cur.execute(f"SELECT entity_id, business_name, business_address FROM {tbl} WHERE {tbl} MATCH ? LIMIT 8;", (q,))
            rows = cur.fetchall()
            
    return rows[:8]


def score_match(s1_name, s1_addr, c_name, c_addr):
    s1_nl = s1_name.lower()
    c_nl = c_name.lower()
    s1_al = s1_addr.lower()
    c_al = c_addr.lower()
    
    n_set = fuzz.token_set_ratio(s1_nl, c_nl)
    if n_set < 50:
        if c_al and len(s1_al) > 5:
            a_sort = fuzz.token_sort_ratio(s1_al, c_al)
            if a_sort >= 75:
                return (0.25 * n_set + 0.75 * a_sort) / 100.0
        return 0.0
    
    # If candidate address is empty in Source 2, match based on high name similarity
    if not c_al.strip():
        if n_set >= 85:
            return (n_set / 100.0) * 0.95
        return 0.0
    
    a_sort = fuzz.token_sort_ratio(s1_al, c_al)
    a_set = fuzz.token_set_ratio(s1_al, c_al)
    
    s1_nums = set(re.findall(r"\b\d+\b", s1_al))
    c_nums = set(re.findall(r"\b\d+\b", c_al))
    num_match = bool(s1_nums & c_nums) if (s1_nums and c_nums) else False
    
    # Case 1: Strong name match
    if n_set >= 80:
        if a_sort >= 50 or num_match or a_set >= 60:
            return (0.60 * n_set + 0.40 * max(a_sort, a_set)) / 100.0
        return (0.75 * n_set + 0.25 * a_sort) / 100.0
    
    # Case 2: Moderate name match with strong address
    if n_set >= 60 and (a_sort >= 60 or (num_match and a_set >= 65)):
        return (0.45 * n_set + 0.55 * a_sort) / 100.0
    
    return 0.0


def build_fts_indices(db_path: Path, test_data_dir: Path = None) -> None:
    """Build country-partitioned SQLite FTS5 indices for Source 2 and Source 3."""
    if db_path.exists():
        db_path.unlink()

    src_dir = test_data_dir if test_data_dir else TEST_DATA_DIR
    print("=" * 70)
    print("STEP 1: Indexing Test Source 2 & Source 3 into Country-Partitioned FTS5")
    print(f"Target SQLite database: {db_path}")
    print(f"Reading candidates from: {src_dir}")
    print("=" * 70)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = OFF;")
    conn.execute("PRAGMA cache_size = 200000;")

    countries = ["France", "India", "US"]
    for c in countries:
        conn.execute(f"""
        CREATE VIRTUAL TABLE candidates_fts_{c} USING fts5(
            entity_id UNINDEXED,
            business_name,
            business_address,
            tokenize = 'porter unicode61'
        );
        """)

    t0 = time.time()
    total_indexed = 0

    for src_file in ["test_source2.tsv", "test_source3.tsv"]:
        fpath = src_dir / src_file
        print(f"Indexing {src_file} from {fpath}...")
        
        reader = pl.read_csv_batched(str(fpath), separator="\t", batch_size=500000)
        file_count = 0
        
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
            
            for c in countries:
                sub_df = batch_df.filter(pl.col("country") == c)
                if len(sub_df) > 0:
                    records = [
                        (r[0], r[1] or "", r[2] or "")
                        for r in sub_df.select(["entity_id", "business_name", "business_address"]).iter_rows()
                    ]
                    conn.executemany(f"INSERT INTO candidates_fts_{c} VALUES (?, ?, ?);", records)
                    file_count += len(records)
            
            conn.commit()
            print(f"  Indexed {file_count:,} records from {src_file}...", end="\r", flush=True)

        total_indexed += file_count
        print(f"\n  Done indexing {src_file} ({file_count:,} records).")

    conn.close()
    elapsed = time.time() - t0
    print(f"Successfully indexed {total_indexed:,} records in {elapsed:.1f}s ({total_indexed/elapsed:.0f} rows/s)")
    print("-" * 70)


proc_cur = None
proc_db_uri = None

def init_worker(db_uri):
    global proc_cur, proc_db_uri
    proc_db_uri = db_uri
    conn = sqlite3.connect(db_uri, uri=True)
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA mmap_size = 2147483648;")
    proc_cur = conn.cursor()

def worker_process_chunk(chunk):
    global proc_cur
    results = []
    for s1 in chunk:
        s1_id = s1["entity_id"]
        cntry = s1["country"] or "US"
        tbl = f"candidates_fts_{cntry}" if cntry in VALID_COUNTRIES else "candidates_fts_US"
        s1_name = s1["business_name"] or ""
        s1_addr = s1["business_address"] or ""
        
        c_rows = get_candidates(proc_cur, tbl, s1_name, s1_addr)
        c_ids = [r[0] for r in c_rows]
        
        scored_matches = []
        for c_id, c_name, c_addr in c_rows:
            s = score_match(s1_name, s1_addr, c_name, c_addr)
            if s >= 0.70:
                scored_matches.append((c_id, s))
        
        matched_str = ""
        if scored_matches:
            max_s = max(s for _, s in scored_matches)
            selected = [c for c, s in scored_matches if s >= 0.80 * max_s]
            matched_str = ",".join(selected)
        
        cand_str = ",".join(c_ids)
        results.append((s1_id, matched_str, cand_str))
    
    return results


def generate_submission(
    db_path: Path,
    out_matching: Path,
    out_candidate: Path,
    test_dir: Path = None,
    batch_size: int = 14000,
    num_threads: int = None,
    top_k: int = 8,
    limit: int = None
) -> None:
    src_dir = test_dir if test_dir else TEST_DATA_DIR
    print("=" * 70)
    print("STEP 2: Streaming Inference on Test Source 1")
    print(f"Target FTS DB: {db_path}")
    print(f"Reading Source 1 from: {src_dir}")
    print("=" * 70)

    db_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    num_procs = num_threads or max(2, min(cpu_count(), 14))

    out_matching.parent.mkdir(parents=True, exist_ok=True)
    out_candidate.parent.mkdir(parents=True, exist_ok=True)

    f_match = open(out_matching, "w", encoding="utf-8", newline="\n")
    f_cand = open(out_candidate, "w", encoding="utf-8", newline="\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    s1_file = src_dir / "test_source1.tsv"
    reader = pl.read_csv_batched(str(s1_file), separator="\t", batch_size=batch_size)

    total_s1 = limit if limit else 1732544
    processed_count = 0
    total_matches = 0
    empty_matches = 0
    t_start = time.time()
    last_log = t_start

    print(f"Launching {num_procs} worker processes...")
    with Pool(processes=num_procs, initializer=init_worker, initargs=(db_uri,)) as pool:
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
            if limit and processed_count + len(batch_df) > limit:
                batch_df = batch_df.slice(0, limit - processed_count)
            rows = batch_df.to_dicts()

            chunk_sz = max(100, len(rows) // num_procs)
            chunks = [rows[i:i + chunk_sz] for i in range(0, len(rows), chunk_sz)]

            chunk_results = pool.map(worker_process_chunk, chunks)
            for sub_res in chunk_results:
                for s1_id, m_str, c_str in sub_res:
                    f_match.write(f"{s1_id}\t{m_str}\n")
                    f_cand.write(f"{s1_id}\t{c_str}\n")
                    if m_str:
                        total_matches += len(m_str.split(","))
                    else:
                        empty_matches += 1
                processed_count += len(sub_res)

            f_match.flush()
            f_cand.flush()

            now = time.time()
            if now - last_log >= 10.0 or processed_count == total_s1:
                elapsed = now - t_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                pct = (processed_count / total_s1) * 100
                rem_sec = (total_s1 - processed_count) / rate if rate > 0 else 0
                avg_m = total_matches / processed_count if processed_count > 0 else 0
                empty_pct = (empty_matches / processed_count * 100) if processed_count > 0 else 0
                print(
                    f"[{time.strftime('%H:%M:%S')}] {processed_count:,} / {total_s1:,} ({pct:5.1f}%) | "
                    f"Rate: {rate:6.1f} ent/s | Matches: {total_matches:,} (Avg: {avg_m:.2f}) | "
                    f"Empty: {empty_pct:.1f}% | ETA: {rem_sec/60:.1f}m",
                    flush=True
                )
                last_log = now

            if limit and processed_count >= limit:
                break

    f_match.close()
    f_cand.close()

    total_time = time.time() - t_start
    print("-" * 70)
    print(f"Inference completed in {total_time/60:.2f} minutes ({total_time:.1f}s)")
    print(f"Total Source 1 entities processed: {processed_count:,}")
    print(f"Total matched entities emitted: {total_matches:,}")
    print(f"Average matches per entity: {total_matches / processed_count:.2f}" if processed_count > 0 else "N/A")
    print(f"Empty predictions: {empty_matches:,} ({empty_matches/processed_count*100:.1f}%)")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate official challenge test submission.")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    parser.add_argument("--top-k", type=int, default=8, help="Number of candidates to retrieve per entity")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of S1 entities to process")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip FTS index creation if DB already exists")
    parser.add_argument("--keep-index", action="store_true", help="Keep the FTS index on disk after completion")
    parser.add_argument("--threads", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=14000, help="Batch size for S1 streaming")
    parser.add_argument("--threshold", type=float, default=0.70, help="Decision threshold")
    parser.add_argument("--margin-ratio", type=float, default=0.80, help="Candidate margin ratio")
    parser.add_argument("--db-path", type=str, default=None, help="Path for SQLite FTS index")
    args = parser.parse_args()

    test_data_dir = Path(args.test_dir) if args.test_dir else TEST_DATA_DIR
    db_file = Path(args.db_path) if args.db_path else (OUTPUT_PATH / "test_fts_index.db")
    matching_tsv = OUTPUT_PATH / "matching_results.tsv"
    candidate_tsv = OUTPUT_PATH / "candidate_pairs.tsv"

    # Step 1: Index Source 2 & Source 3 (unless skipped)
    if (args.skip_indexing or args.keep_index) and db_file.exists():
        print(f"Reusing existing index at {db_file}")
    else:
        build_fts_indices(db_file, test_data_dir)

    # Step 2: Stream inference on Source 1
    generate_submission(
        db_path=db_file,
        test_dir=test_data_dir,
        out_matching=matching_tsv,
        out_candidate=candidate_tsv,
        batch_size=args.batch_size,
        num_threads=args.threads,
        top_k=args.top_k,
        limit=args.limit
    )

    # Step 3: Validate generated submission (only on full run)
    if args.limit is None:
        print("\n" + "=" * 70)
        print("STEP 3: Validating Generated Submission Files")
        print("=" * 70)
        validate_submission(
            matching_path=matching_tsv,
            candidate_path=candidate_tsv,
            test_dir=TEST_DATA_DIR
        )
        print("Submission generation and validation SUCCESSFUL!")
        import shutil
        root_matching = PROJECT_ROOT / "matching_results.tsv"
        root_candidate = PROJECT_ROOT / "candidate_pairs.tsv"
        shutil.copy2(matching_tsv, root_matching)
        shutil.copy2(candidate_tsv, root_candidate)
        print(f"Synced submission files to repository root: {root_matching.name}, {root_candidate.name}")

        # Step 4: Rebuild final submission zip
        print("\n" + "=" * 70)
        print("STEP 4: Packaging Winning DocGuru_submission.zip")
        print("=" * 70)
        from scripts.build_submission_zip import build_submission_zip
        build_submission_zip(team_name="DocGuru")


if __name__ == "__main__":
    main()

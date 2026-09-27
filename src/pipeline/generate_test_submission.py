"""
Production Streaming Candidate Generation and Matching Pipeline.

Generates official challenge submission files:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Uses country-partitioned SQLite FTS5 BM25 index on disk with multi-threaded
batched LightGBM ensemble scoring. Preserves exact entity order, keeps memory
under 400 MB, and optimizes for maximum Macro F_0.5 and compact candidate ranking.
"""

import os
import sys
import time
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz

from src.config import PROJECT_ROOT, TEST_PATH, OUTPUT_PATH
from utils.validate_submission import validate_submission

OFFICIAL_TEST = PROJECT_ROOT / "dataset_official" / "test"
TEST_DATA_DIR = OFFICIAL_TEST if OFFICIAL_TEST.is_dir() else TEST_PATH


GENERIC_TERMS = {
    "the", "and", "ltd", "inc", "llc", "corp", "pvt", "limited", "private",
    "corporation", "null", "nan", "street", "st", "ave", "road", "rd", "de", "la",
    "services", "technologies", "solutions", "industries", "group", "international",
    "global", "trading", "enterprises", "holdings", "systems", "management", "consulting",
    "company", "co", "associates", "agency", "center", "centre", "club",
    "sarl", "sasu", "sas", "eurl", "gmbh", "sa", "llp", "plc", "cie", "ets",
    "nagar", "vihar", "colony", "enclave", "bazaar", "bazar", "chowk", "floor",
    "suite", "ste", "room", "building", "bldg", "house", "plot", "flat", "sector",
    "phase", "near", "opp", "opposite", "avenue", "boulevard", "blvd", "lane",
    "drive", "dr", "way", "rue", "place", "route", "chemin", "allee", "zone",
    "zi", "za", "cedex", "paris", "lyon", "bordeaux", "delhi", "mumbai",
    "bangalore", "kolkata", "chennai", "hyderabad", "india", "france"
}
VALID_COUNTRIES = {"France", "India", "US"}


def make_query_dual(name: str, addr: str) -> str:
    """Build dual distinctive name + address FTS query for high-recall BM25 retrieval."""
    n_words = re.findall(r"[^\W_]+", name or "", re.UNICODE)
    dist_name = [w.replace('"', '').replace("'", '') for w in n_words if len(w) >= 3 and w.lower() not in GENERIC_TERMS]
    if not dist_name:
        dist_name = [w.replace('"', '').replace("'", '') for w in n_words if len(w) >= 2 and w.lower() not in GENERIC_TERMS]
    
    # Selective numbers: 3+ digits (street numbers, building numbers, zip codes) to avoid slow 1-2 digit scans
    nums = [n for n in re.findall(r"\b\d+\b", addr or "") if len(n) >= 3]
    a_words = re.findall(r"[^\W_]+", addr or "", re.UNICODE)
    dist_addr = [w.replace('"', '').replace("'", '') for w in a_words if len(w) >= 4 and w.lower() not in GENERIC_TERMS]
    
    terms = []
    seen = set()
    for w in dist_name[:2]:
        tok = f'{w}*'
        if tok.lower() not in seen:
            terms.append(tok)
            seen.add(tok.lower())
    for n in nums[:1]:
        tok = f'"{n}"'
        if tok.lower() not in seen:
            terms.append(tok)
            seen.add(tok.lower())
    for a in dist_addr[:1]:
        tok = f'"{a}"'
        if tok.lower() not in seen:
            terms.append(tok)
            seen.add(tok.lower())
        
    if not terms:
        all_w = [w for w in n_words + a_words if len(w) >= 3 and w.lower() not in GENERIC_TERMS]
        if all_w:
            terms.append(f'{all_w[0]}*')
            
    if not terms:
        return ""
    return " OR ".join(terms[:4])


make_query = make_query_dual


def build_fts_indices(db_path: Path) -> None:
    """
    Build country-partitioned SQLite FTS5 indices for Source 2 and Source 3.
    """
    if db_path.exists():
        db_path.unlink()

    print("=" * 70)
    print("STEP 1: Indexing Test Source 2 & Source 3 into Country-Partitioned FTS5")
    print(f"Target SQLite database: {db_path}")
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
        fpath = TEST_DATA_DIR / src_file
        print(f"Indexing {src_file}...")
        
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


def generate_submission(
    db_path: Path,
    model_path: Path,
    out_matching: Path,
    out_candidate: Path,
    batch_size: int = 3000,
    num_threads: int = 6,
    top_k: int = 8,
    threshold: float = 0.90,
    margin_ratio: float = 0.80,
    limit: int = None
) -> None:
    """
    Stream Source 1 entities, retrieve top candidates via FTS5 with BM25 ranking,
    extract features, score with calibrated hybrid ensemble, and stream output TSV lines in exact order.
    """
    print("=" * 70)
    print("STEP 2: Streaming Inference on Test Source 1")
    print(f"Loading matcher model from: {model_path}")
    print(f"Settings: top_k={top_k}, threshold={threshold}, margin_ratio={margin_ratio}, threads={num_threads}, limit={limit}")
    print("=" * 70)

    model = joblib.load(model_path)
    db_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"

    # Thread-local database connections
    thread_local = threading.local()

    def get_cursor():
        if not hasattr(thread_local, "conn"):
            conn = sqlite3.connect(db_uri, uri=True)
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA mmap_size = 2147483648;")  # 2 GB memory-mapped I/O
            conn.execute("PRAGMA cache_size = -64000;")     # 64 MB page cache
            conn.execute("PRAGMA temp_store = MEMORY;")
            thread_local.conn = conn
            thread_local.cur = conn.cursor()
        return thread_local.cur

    def process_subchunk(subchunk: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """
        Process a subchunk of Source 1 records in a worker thread.
        Returns: list of (s1_id, match_str, cand_str) in same order.
        """
        cur = get_cursor()
        
        batch_features = []
        batch_meta = []  # (s1_id, c_id, sim_score)
        cand_dict = {}   # s1_id -> [cand_ids]
        
        for s1 in subchunk:
            s1_id = s1["entity_id"]
            cntry = s1["country"] or "US"
            tbl = f"candidates_fts_{cntry}" if cntry in VALID_COUNTRIES else "candidates_fts_US"
            s1_name = s1["business_name"] or ""
            s1_addr = s1["business_address"] or ""
            
            q = make_query_dual(s1_name, s1_addr)
            if not q:
                cand_dict[s1_id] = []
                continue
            
            try:
                cur.execute(
                    f"SELECT entity_id, business_name, business_address FROM {tbl} WHERE {tbl} MATCH ? ORDER BY rank LIMIT {top_k};",
                    (q,)
                )
                c_rows = cur.fetchall()
                if not c_rows:
                    words = re.findall(r"[^\W_]+", s1_name or "", re.UNICODE)
                    dist = [w for w in words if len(w) >= 3 and w.lower() not in GENERIC_TERMS]
                    if dist:
                        first_tok = f"{dist[0]}*"
                        cur.execute(
                            f"SELECT entity_id, business_name, business_address FROM {tbl} WHERE {tbl} MATCH ? ORDER BY rank LIMIT {top_k};",
                            (first_tok,)
                        )
                        c_rows = cur.fetchall()
            except Exception:
                c_rows = []
            
            c_ids = [r[0] for r in c_rows]
            cand_dict[s1_id] = c_ids
            
            s1_name_lower = s1_name.lower()
            s1_addr_lower = s1_addr.lower()
            s1_tok_set = {t for t in s1_name_lower.split() if len(t) >= 2}
            
            for c_id, c_name, c_addr in c_rows:
                c_name_lower = c_name.lower()
                c_addr_lower = c_addr.lower()
                
                n_ratio = fuzz.ratio(s1_name_lower, c_name_lower)
                n_token_set = fuzz.token_set_ratio(s1_name_lower, c_name_lower)
                n_partial = fuzz.partial_ratio(s1_name_lower, c_name_lower)
                
                a_ratio = fuzz.ratio(s1_addr_lower, c_addr_lower)
                a_token_set = fuzz.token_set_ratio(s1_addr_lower, c_addr_lower)
                a_token_sort = fuzz.token_sort_ratio(s1_addr_lower, c_addr_lower)
                
                c_tok_set = {t for t in c_name_lower.split() if len(t) >= 2}
                name_jaccard = (len(s1_tok_set & c_tok_set) / len(s1_tok_set | c_tok_set)) if (s1_tok_set | c_tok_set) else 0.0
                combo = 0.5 * n_token_set + 0.5 * a_token_sort
                
                batch_features.append([
                    1.0, n_ratio, n_token_set, n_partial, name_jaccard,
                    a_ratio, a_token_set, a_token_sort, combo
                ])
                
                # Unsupervised hand-crafted similarity score normalized to 0-1
                sim_score = (0.50 * n_token_set + 0.35 * a_token_sort + 0.15 * a_token_set) / 100.0
                batch_meta.append((s1_id, c_id, sim_score))

        # Batched model inference with calibrated hybrid scoring
        match_dict = {s1["entity_id"]: [] for s1 in subchunk}
        if batch_features:
            X_batch = np.array(batch_features, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            
            s1_candidate_scores: Dict[str, List[Tuple[str, float]]] = {}
            for (s1_id, c_id, sim_score), prob in zip(batch_meta, probs):
                final_score = max(float(prob), float(sim_score))
                if s1_id not in s1_candidate_scores:
                    s1_candidate_scores[s1_id] = []
                s1_candidate_scores[s1_id].append((c_id, final_score))
                
            for s1_id, scored_cands in s1_candidate_scores.items():
                valid = [(c, s) for c, s in scored_cands if s >= threshold]
                if valid:
                    max_s = max(s for c, s in valid)
                    selected = [c for c, s in valid if s >= margin_ratio * max_s]
                    match_dict[s1_id] = selected

        # Format output strings
        results = []
        for s1 in subchunk:
            s1_id = s1["entity_id"]
            cand_str = ",".join(cand_dict.get(s1_id, []))
            match_str = ",".join(match_dict.get(s1_id, []))
            results.append((s1_id, match_str, cand_str))

        return results

    # Ensure output directory exists
    out_matching.parent.mkdir(parents=True, exist_ok=True)
    out_candidate.parent.mkdir(parents=True, exist_ok=True)

    f_match = open(out_matching, "w", encoding="utf-8", newline="\n")
    f_cand = open(out_candidate, "w", encoding="utf-8", newline="\n")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    s1_file = TEST_DATA_DIR / "test_source1.tsv"
    reader = pl.read_csv_batched(str(s1_file), separator="\t", batch_size=batch_size)

    total_s1 = 1732544 if limit is None else limit
    processed_count = 0
    total_matches_count = 0
    t_start = time.time()
    last_log_time = t_start

    executor = ThreadPoolExecutor(max_workers=num_threads)

    try:
        while True:
            if limit is not None and processed_count >= limit:
                break
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
            rows = batch_df.to_dicts()
            if limit is not None and (processed_count + len(rows)) > limit:
                rows = rows[:(limit - processed_count)]
            
            # Split batch into subchunks for threads
            subchunk_size = max(100, len(rows) // num_threads)
            subchunks = [rows[i:i + subchunk_size] for i in range(0, len(rows), subchunk_size)]
            
            # Run threads
            futures = [executor.submit(process_subchunk, sc) for sc in subchunks]
            for fut in futures:
                sub_results = fut.result()
                for s1_id, match_str, cand_str in sub_results:
                    f_match.write(f"{s1_id}\t{match_str}\n")
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    if match_str:
                        total_matches_count += len(match_str.split(","))
                processed_count += len(sub_results)

            # Flush periodically
            f_match.flush()
            f_cand.flush()

            # Logging
            now = time.time()
            if now - last_log_time >= 15.0 or processed_count == total_s1:
                elapsed = now - t_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                pct = (processed_count / total_s1) * 100
                rem_entities = max(0, total_s1 - processed_count)
                eta_sec = rem_entities / rate if rate > 0 else 0
                eta_min = eta_sec / 60.0
                print(
                    f"[{time.strftime('%H:%M:%S')}] {processed_count:,} / {total_s1:,} ({pct:5.1f}%) | "
                    f"Rate: {rate:6.1f} ent/s | Matches: {total_matches_count:,} | ETA: {eta_min:4.1f}m",
                    flush=True
                )
                last_log_time = now

    finally:
        executor.shutdown(wait=True)
        f_match.close()
        f_cand.close()

    total_time = time.time() - t_start
    print("-" * 70)
    print(f"Inference completed in {total_time/60:.2f} minutes ({total_time:.1f}s)")
    print(f"Total Source 1 entities processed: {processed_count:,}")
    print(f"Total matched entities emitted: {total_matches_count:,}")
    print(f"Average matches per entity: {total_matches_count / processed_count:.2f}" if processed_count > 0 else "N/A")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate official challenge test submission.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of S1 entities to process (for quick test)")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip FTS index creation if DB already exists")
    parser.add_argument("--keep-index", action="store_true", help="Keep the FTS index on disk after completion")
    parser.add_argument("--threads", type=int, default=6, help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=3000, help="Batch size for S1 streaming")
    parser.add_argument("--threshold", type=float, default=0.90, help="Decision threshold for match acceptance (optimal 0.90)")
    parser.add_argument("--margin-ratio", type=float, default=0.80, help="Entity-relative candidate margin ratio (optimal 0.80)")
    parser.add_argument("--db-path", type=str, default=None, help="Custom path for SQLite FTS index (e.g. /dev/shm/test_fts_index.db for RAM disk)")
    args = parser.parse_args()

    db_file = Path(args.db_path) if args.db_path else (OUTPUT_PATH / "test_fts_index.db")
    model_file = PROJECT_ROOT / "src" / "models" / "saved" / "matcher_model.pkl"
    matching_tsv = OUTPUT_PATH / "matching_results.tsv"
    candidate_tsv = OUTPUT_PATH / "candidate_pairs.tsv"

    # Step 1: Index Source 2 & Source 3 (unless skipped)
    if (args.skip_indexing or args.keep_index) and db_file.exists():
        print(f"Skipping indexing: reusing existing index at {db_file}")
    else:
        build_fts_indices(db_file)

    # Step 2: Stream inference on Source 1
    generate_submission(
        db_path=db_file,
        model_path=model_file,
        out_matching=matching_tsv,
        out_candidate=candidate_tsv,
        batch_size=args.batch_size,
        num_threads=args.threads,
        top_k=8,
        threshold=args.threshold,
        margin_ratio=args.margin_ratio,
        limit=args.limit
    )

    # Step 3: Remove temporary DB to free disk space (only if full run and not keep_index)
    if args.limit is None and db_file.exists() and not args.keep_index:
        print(f"Cleaning temporary index database: {db_file}...")
        try:
            db_file.unlink()
        except Exception as e:
            print(f"Warning: could not delete {db_file}: {e}")

    # Step 4: Validate generated submission (only on full run)
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

        # Step 5: Automatically rebuild and package final submission zip
        print("\n" + "=" * 70)
        print("STEP 4: Packaging Winning DocGuru_submission.zip")
        print("=" * 70)
        from scripts.build_submission_zip import build_submission_zip
        build_submission_zip(team_name="DocGuru")


if __name__ == "__main__":
    main()

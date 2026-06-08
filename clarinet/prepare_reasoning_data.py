"""
One-time PARALLEL download + re-shard of the reasoning corpus (FineMath) into
the same parquet layout nanochat/dataset.py expects for climbmix:
  <base_dir>/reasoning_data/finemath/shard_NNNNN.parquet
  - single 'text' column
  - DOCS_PER_ROW_GROUP docs per row group (needed for DDP row-group sharding)

FineMath is already parquet with a 'text' column, so we download the native
files in parallel (huggingface_hub.snapshot_download, max_workers) and reshard
them with pyarrow's C++ engine — no per-record Python.

This replaces an earlier `datasets`-streaming implementation that was
pathologically slow (a single 65k-doc shard could take hours): streaming
decoded one Python record at a time, CPU-bound on a single core. The
download-then-native-reshard path here is minutes, not hours.

Run once before clarinet training:

  python -m clarinet.prepare_reasoning_data -n 63        # ~all of finemath-4plus
  python -m clarinet.prepare_reasoning_data -n 1         # tiny smoke (2 files)

Tip: `pip install hf_transfer` and `export HF_HUB_ENABLE_HF_TRANSFER=1` makes the
download materially faster on a fat pipe.
"""

import argparse
import os

import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, snapshot_download

from clarinet.dataset import reasoning_data_dir

DEFAULT_REPO = "HuggingFaceTB/finemath"
DEFAULT_CONFIG = "finemath-4plus"
DOCS_PER_ROW_GROUP = 1024  # matches climbmix granularity for per-rank DDP sharding


def main():
    parser = argparse.ArgumentParser(description="Parallel download + reshard of the reasoning corpus")
    parser.add_argument("-n", "--num-shards", type=int, default=63,
                        help="Number of TRAIN shards to write; a +1 validation shard is always "
                             "added (last shard). One output shard per native file, capped at the "
                             "number of files available in the subset. Default 63 (~all of "
                             "finemath-4plus). Use a small value (e.g. 1) for a cheap smoke.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="HF dataset config / subset.")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel download workers.")
    parser.add_argument("--force", action="store_true",
                        help="Re-reshard even if output shards already exist (default: skip existing, "
                             "so re-runs / resumes are near-instant no-ops).")
    args = parser.parse_args()

    out_dir = reasoning_data_dir()
    os.makedirs(out_dir, exist_ok=True)

    # Enumerate the subset's parquet files and take only as many as we need
    # (train + 1 val), so a small -n stays a cheap smoke instead of pulling the
    # whole subset.
    all_files = sorted(
        f for f in list_repo_files(args.repo, repo_type="dataset")
        if f.startswith(f"{args.config}/") and f.endswith(".parquet")
    )
    if not all_files:
        raise SystemExit(f"No parquet files found under {args.repo}:{args.config}/")
    n_total = min(args.num_shards + 1, len(all_files))  # +1 for the validation shard
    take = all_files[:n_total]
    dsts = [os.path.join(out_dir, f"shard_{i:05d}.parquet") for i in range(n_total)]

    # Idempotent skip: if every target shard already exists, do nothing (don't
    # even hit the network). Lets re-runs / resumed pipelines be near-instant.
    if not args.force and all(os.path.exists(d) for d in dsts):
        print(f"All {n_total} shards already present in {out_dir}; nothing to do "
              f"(use --force to rebuild).")
        return

    print(f"{args.repo}/{args.config}: {len(all_files)} files available; "
          f"downloading {n_total} in parallel ({args.max_workers} workers)...")
    local = snapshot_download(
        repo_id=args.repo, repo_type="dataset",
        allow_patterns=take, max_workers=args.max_workers,
    )

    print(f"Resharding into {out_dir}")
    written = 0
    for i, rel in enumerate(take):
        dst = dsts[i]
        if not args.force and os.path.exists(dst):
            print(f"  skip shard_{i:05d}.parquet (already exists)")
            continue
        table = pq.read_table(os.path.join(local, rel), columns=["text"])  # C++ read
        pq.write_table(table, dst, compression="zstd", row_group_size=DOCS_PER_ROW_GROUP)
        print(f"  wrote shard_{i:05d}.parquet ({table.num_rows} docs)")
        written += 1

    print(f"Done. {n_total} shards present ({n_total - 1} train + 1 val) in {out_dir} "
          f"[{written} written, {n_total - written} already existed]")


if __name__ == "__main__":
    main()

"""
First-stage relevance probe for clarinet's IV conditioning.

Task benchmarks can't tell you whether the source marker does anything when the
model is at the task floor (the d18 base model is at chance on MMLU/ARC and 0%
on GSM8K). This probe asks the question directly at the distribution level:

    Does conditioning on a source marker change the model's held-out loss
    in the direction it should?

It computes conditional bits-per-byte on held-out text from each corpus, under
each of the three markers — a 2x3 grid:

                         <|src_reasoning|>   <|src_general|>   <|src_unknown|>
    FineMath   (val)            a                  b                 c
    climbmix   (val)            d                  e                 f

Reading it:
  - Diagonal wins (a < b and e < d, with c/f in between): the conditioning is
    REAL — the instrument has first-stage relevance. Flat task sweeps are then
    a statistical-power problem (model too weak), not evidence the IV is inert.
  - Rows are flat (a ~= b ~= c): the model learned to ignore the marker; the
    conditioning needs to be strengthened before anything else is worth running.

Numbers are computed with nanochat's evaluate_bpb, so they are directly
comparable to the val bpb printed during pretraining. The marker token itself
is a special token (0 bytes) and is automatically excluded from the metric, so
all three conditions are scored on exactly the same document tokens.

Usage:
    python -m scripts.iv_probe -g d18-iv -n 200
"""

import argparse

import pyarrow.parquet as pq
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.dataset import parquets_iter_batched
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes

from clarinet.dataloader import SRC_GENERAL, SRC_REASONING, SRC_UNKNOWN
from clarinet.dataset import list_reasoning_parquet_files


def pack_doc(doc_ids, bos_id, marker_id, max_seq_len):
    """
    Lay out one document the way training did: [BOS, marker, ...doc tokens],
    truncated so inputs/targets are at most max_seq_len long.
    Returns (inputs, targets) as python lists (unpadded).
    The marker is the target at position 0; it's a special token with
    token_bytes == 0, so evaluate_bpb excludes it — every condition is scored
    on the same doc tokens.
    """
    row = [bos_id, marker_id] + doc_ids
    row = row[:max_seq_len + 1]
    return row[:-1], row[1:]


def batched_bpb(model, tokenizer, doc_ids_list, marker_name, batch_size, max_seq_len, token_bytes, device):
    """Mean bpb of doc_ids_list under the given marker conditioning."""
    bos_id = tokenizer.get_bos_token_id()
    marker_id = tokenizer.encode_special(marker_name)
    rows = [pack_doc(ids, bos_id, marker_id, max_seq_len) for ids in doc_ids_list]

    def gen():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            width = max(len(x) for x, _ in chunk)
            inputs = torch.full((len(chunk), width), bos_id, dtype=torch.long)
            targets = torch.full((len(chunk), width), -1, dtype=torch.long)
            for r, (x, y) in enumerate(chunk):
                inputs[r, :len(x)] = torch.tensor(x, dtype=torch.long)
                targets[r, :len(y)] = torch.tensor(y, dtype=torch.long)
            yield inputs.to(device), targets.to(device)

    steps = -(-len(rows) // batch_size)  # ceil_div
    return evaluate_bpb(model, gen(), steps, token_bytes)


def load_val_docs_climbmix(n):
    docs = []
    for batch in parquets_iter_batched(split="val"):
        docs.extend(batch)
        if len(docs) >= n:
            return docs[:n]
    return docs


def load_val_docs_reasoning(n):
    files = list_reasoning_parquet_files()
    assert files, "No reasoning shards found — run clarinet.prepare_reasoning_data first"
    pf = pq.ParquetFile(files[-1])  # last shard = val, matching the train/val convention
    docs = []
    for rg in range(pf.num_row_groups):
        docs.extend(pf.read_row_group(rg).column("text").to_pylist())
        if len(docs) >= n:
            return docs[:n]
    return docs


def main():
    parser = argparse.ArgumentParser(description="Marker first-stage relevance probe (conditional BPB grid)")
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("-i", "--source", type=str, default="base", help="base|sft|rl (default: base)")
    parser.add_argument("-n", "--num-docs", type=int, default=200, help="held-out docs per corpus")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local, _world, device = compute_init(device_type)
    model, tokenizer, _meta = load_model(args.source, device, phase="eval",
                                         model_tag=args.model_tag, step=args.step)
    token_bytes = get_token_bytes(device=device)

    print0(f"Loading {args.num_docs} held-out docs per corpus...")
    corpora = {
        "FineMath (reasoning)": load_val_docs_reasoning(args.num_docs),
        "climbmix (general)  ": load_val_docs_climbmix(args.num_docs),
    }
    markers = [SRC_REASONING, SRC_GENERAL, SRC_UNKNOWN]

    grid = {}
    for corpus_name, texts in corpora.items():
        doc_ids = tokenizer.encode(texts)  # batched encode, no specials
        for marker in markers:
            bpb = batched_bpb(model, tokenizer, doc_ids, marker,
                              args.batch_size, args.max_seq_len, token_bytes, device)
            grid[(corpus_name, marker)] = bpb
            print0(f"  {corpus_name} | {marker:<18}: {bpb:.6f} bpb")

    print0("\n========== Conditional BPB grid (rows=text source, cols=marker) ==========")
    header = "text source".ljust(22) + "".join(m.replace('<|src_', '').replace('|>', '').rjust(12) for m in markers)
    print0(header)
    for corpus_name in corpora:
        row = corpus_name.ljust(22) + "".join(f"{grid[(corpus_name, m)]:>12.6f}" for m in markers)
        print0(row)

    print0("\n========== First-stage relevance (within-row deltas vs matched marker) ==========")
    fm, cm = list(corpora.keys())
    d_fm = grid[(fm, SRC_GENERAL)] - grid[(fm, SRC_REASONING)]
    d_cm = grid[(cm, SRC_REASONING)] - grid[(cm, SRC_GENERAL)]
    print0(f"FineMath:  bpb(general) - bpb(reasoning) = {d_fm:+.6f}   (positive => reasoning marker helps on reasoning text)")
    print0(f"climbmix:  bpb(reasoning) - bpb(general) = {d_cm:+.6f}   (positive => general marker helps on general text)")
    if d_fm > 0 and d_cm > 0:
        print0("=> DIAGONAL WINS: the marker conditioning is real (instrument has first-stage relevance).")
    elif d_fm <= 0 and d_cm <= 0:
        print0("=> FLAT/INVERTED: the model is ignoring the marker; strengthen the conditioning before further runs.")
    else:
        print0("=> MIXED: one corpus responds, the other doesn't — borderline relevance, interpret with care.")

    compute_cleanup()


if __name__ == "__main__":
    main()

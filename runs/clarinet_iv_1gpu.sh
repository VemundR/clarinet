#!/bin/bash
# Clarinet IV arm ONLY (the d18 "clarinet" half of the A/B), on a single H100.
#
# Pipeline: shared data + tokenizer -> d18 clarinet pretrain (markers + p_uncond)
#           -> base_eval (BPB) -> SFT -> dual-pass IV guidance sweep -> report.
#
# This is ARM B of runs/clarinet_ab.sh, standalone and single-GPU. Run the d24
# baseline (ARM A) separately when you want the reference number to compare to.
#
# HARDWARE: single H100. Plain `python -m` (no torchrun); nanochat auto-uses
# gradient accumulation. FP8 + FA3 apply (single H100 is Hopper).
#
# TIMING on 1xH100: ~5-6h pretrain + SFT (~1-2h) + IV sweep (~1-2h) ≈ ~8h total.
#
# Launch in tmux so an SSH drop doesn't kill it:
#   tmux new -s d18iv
#   WANDB_RUN=clarinet-iv bash runs/clarinet_iv_1gpu.sh 2>&1 | tee iv.log

set -e

export OMP_NUM_THREADS=1
export CLARINET_BASE_DIR="${CLARINET_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$CLARINET_BASE_DIR"

DEPTH=18
MIX=0.5            # reasoning_mix_ratio (50/50 climbmix + FineMath)
DBS=16             # device-batch-size (fits d18 on an 80GB H100; try 32 for more throughput)
RATIO=8            # target data:param ratio
WANDB_RUN="${WANDB_RUN:-dummy}"
IV_TAG="d${DEPTH}-iv"

# -----------------------------------------------------------------------------
# Environment
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Shared data + tokenizer. Sized for d18; climbmix tops up later if you also run
# the d24 baseline. FineMath uses the fast parallel download+reshard path.
python -m nanochat.dataset -n 8                 # enough for tokenizer training
python -m nanochat.dataset -n 60 &              # climbmix (general) half, ~enough for d18
DL=$!
python -m clarinet.prepare_reasoning_data -n 30 &   # FineMath (reasoning) half
RP=$!

# Tokenizer (retrained once; clarinet added 3 special tokens). Trained on
# climbmix only, so it overlaps with the FineMath prep above.
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for climbmix download..."; wait $DL
echo "Waiting for FineMath prep...";     wait $RP

# -----------------------------------------------------------------------------
# d18 clarinet pretraining (markers + CFG-style dropout). Single GPU: no torchrun.
python -m scripts.clarinet_train \
    --reasoning-mix-ratio=$MIX --p-uncond=0.1 \
    --depth=$DEPTH --target-param-data-ratio=$RATIO --device-batch-size=$DBS --fp8 \
    --model-tag=$IV_TAG --run=${WANDB_RUN}-iv

# Base-model BPB / CORE sanity (single-pass; optional but informative).
python -m scripts.base_eval --device-batch-size=$DBS --model-tag=$IV_TAG

# SFT
curl -L -o "$CLARINET_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl || true
python -m scripts.chat_sft --model-tag=$IV_TAG --device-batch-size=$DBS --run=${WANDB_RUN}-iv

# Dual-pass IV guidance sweep — the actual clarinet result.
# w=0 -> unconditional, w=1 -> cond-only (markers, no guidance), w>1 -> guided.
python -m scripts.iv_eval -i sft -g $IV_TAG \
    -a GSM8K,ARC-Easy,ARC-Challenge,MMLU,HumanEval,SpellingBee \
    --weights 0,0.5,1.0,1.5,2.0,3.0

# -----------------------------------------------------------------------------
python -m nanochat.report generate
echo
echo "=================== d18 CLARINET ARM COMPLETE ==================="
echo "IV sweep table: report section for tag $IV_TAG"
echo "Report: $CLARINET_BASE_DIR/report/report.md (also copied to ./report.md)"
echo "Checkpoint: $CLARINET_BASE_DIR/chatsft_checkpoints/$IV_TAG/"
echo "Run the d24 baseline later to get the comparison reference."

#!/bin/bash
# Clarinet A/B experiment on 8xH100.
#
# Holds the data mixture constant (reasoning_mix_ratio=0.5 for BOTH arms) and
# varies ONLY the IV mechanism:
#
#   ARM A  "baseline"  : classic nanochat at depth 24, trained on the 50/50
#                        climbmix+FineMath mix, NO source markers, single-pass eval.
#   ARM B  "clarinet"  : depth 18, same 50/50 mix WITH source markers + p_uncond
#                        dropout, dual-pass eval with an IV guidance-weight sweep.
#
# NOTE on interpretation: the two arms differ in BOTH depth (24 vs 18) AND the
# IV mechanism. So this is a size-vs-mechanism comparison — it answers "can a
# smaller (d18) IV-conditioned model match/beat a larger (d24) vanilla model?"
# rather than a fixed-size IV ablation. (For a pure IV ablation, set both depths
# equal.) The data mixture is held constant at 0.5 for both arms.
#
# Both arms share ONE tokenizer and ONE copy of the data (downloaded once), and
# write to distinct checkpoint tags (d24-base vs d18-iv).
#
# Cost note: a d24 + a d18 pretrain together run ~5-6h on 8xH100 (ballpark
# $90-130 depending on Vast pricing). Provision ~250GB disk (both checkpoint
# sets + shared data + caches).
#
# Launch (inside tmux so an SSH drop doesn't kill it):
#   tmux new -s clarinet
#   WANDB_RUN=clarinet-ab bash runs/clarinet_ab.sh 2>&1 | tee ab.log

set -e

export OMP_NUM_THREADS=1
export CLARINET_BASE_DIR="${CLARINET_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$CLARINET_BASE_DIR"

BASE_DEPTH=24      # ARM A "baseline": classic nanochat depth
IV_DEPTH=18        # ARM B "clarinet": IV-conditioned model depth
MIX=0.5            # reasoning_mix_ratio for BOTH arms — the held-constant variable
NPROC=8
DBS=16             # device-batch-size
RATIO=8            # target data:param ratio
WANDB_RUN="${WANDB_RUN:-dummy}"
BASE_TAG="d${BASE_DEPTH}-base"
IV_TAG="d${IV_DEPTH}-iv"

# -----------------------------------------------------------------------------
# Environment
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Shared data + tokenizer (downloaded/trained ONCE, used by both arms)
# At mix=0.5 each corpus supplies half the tokens, so we need roughly equal
# shard counts from each. These are generous; reduce if disk/time constrained.
python -m nanochat.dataset -n 8                 # enough for tokenizer training
python -m nanochat.dataset -n 150 &             # climbmix (general) half
DL=$!
python -m clarinet.prepare_reasoning_data -n 120 &   # FineMath (reasoning) half
RP=$!

# Tokenizer is retrained once (clarinet added 3 special tokens). BOTH arms use
# this same tokenizer so vocab/BPB are directly comparable.
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for climbmix download..."; wait $DL
echo "Waiting for FineMath prep...";     wait $RP

# =============================================================================
# ARM A — BASELINE (classic nanochat on the mix, NO markers)
# =============================================================================
torchrun --standalone --nproc_per_node=$NPROC -m scripts.clarinet_train -- \
    --reasoning-mix-ratio=$MIX --no-markers \
    --depth=$BASE_DEPTH --target-param-data-ratio=$RATIO --device-batch-size=$DBS --fp8 \
    --model-tag=$BASE_TAG --run=${WANDB_RUN}-base

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
    --device-batch-size=$DBS --model-tag=$BASE_TAG

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --model-tag=$BASE_TAG --device-batch-size=$DBS --run=${WANDB_RUN}-base

# Baseline eval is single-pass (no IV). This is the reference number.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- \
    -i sft -g $BASE_TAG

# =============================================================================
# ARM B — CLARINET (same mix, WITH markers + IV)
# =============================================================================
torchrun --standalone --nproc_per_node=$NPROC -m scripts.clarinet_train -- \
    --reasoning-mix-ratio=$MIX --p-uncond=0.1 \
    --depth=$IV_DEPTH --target-param-data-ratio=$RATIO --device-batch-size=$DBS --fp8 \
    --model-tag=$IV_TAG --run=${WANDB_RUN}-iv

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
    --device-batch-size=$DBS --model-tag=$IV_TAG

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --model-tag=$IV_TAG --device-batch-size=$DBS --run=${WANDB_RUN}-iv

# Clarinet eval is the dual-pass IV sweep. w=0 -> unconditional, w=1 -> cond-only
# (markers but no guidance), w>1 -> guided. Compare the curve against the
# baseline's single number above.
python -m scripts.iv_eval -i sft -g $IV_TAG \
    -a GSM8K,ARC-Easy,ARC-Challenge,MMLU,HumanEval,SpellingBee \
    --weights 0,0.5,1.0,1.5,2.0,3.0

# -----------------------------------------------------------------------------
python -m nanochat.report generate
echo
echo "=================== A/B COMPLETE ==================="
echo "Baseline (single-pass) : chat_eval section for tag $BASE_TAG"
echo "Clarinet (IV sweep)    : iv_eval table for tag    $IV_TAG"
echo "Report: $CLARINET_BASE_DIR/report/report.md (also copied to ./report.md)"
echo "Checkpoints: $CLARINET_BASE_DIR/chatsft_checkpoints/{$BASE_TAG,$IV_TAG}/"

#!/bin/bash
# Clarinet A/B experiment — configured for a SINGLE H100 (1xH100).
#
# Holds the data mixture constant (reasoning_mix_ratio=0.5 for BOTH arms) and
# varies the model + IV mechanism:
#
#   ARM A  "baseline"  : classic nanochat at depth 24, trained on the 50/50
#                        climbmix+FineMath mix, NO source markers, single-pass eval.
#   ARM B  "clarinet"  : depth 18, same 50/50 mix WITH source markers + p_uncond
#                        dropout, dual-pass eval with an IV guidance-weight sweep.
#
# NOTE on interpretation: the two arms differ in BOTH depth (24 vs 18) AND the
# IV mechanism. The depths are chosen so INFERENCE compute matches: clarinet's
# dual pass costs 2x per token, and d18 ~ 0.42x d24 per pass, so dual-pass d18
# ~ 0.84x single-pass d24 — a test-time-FLOPs-matched comparison ("same serving
# budget: bigger model, or guidance?"). Training compute is NOT matched (d24
# trains ~2.4x more; favors the baseline). Clarinet also needs 2x KV memory.
# For a fixed-size IV ablation set BASE_DEPTH == IV_DEPTH.
#
# SKIPPING ARMS: if one arm is already trained (e.g. d18-iv from a previous
# session), skip it: SKIP_IV=1 bash runs/clarinet_ab.sh  (or SKIP_BASE=1).
#
# Both arms share ONE tokenizer and ONE copy of the data, and write to distinct
# checkpoint tags (d24-base vs d18-iv).
#
# HARDWARE: single H100. We run plain `python -m` (no torchrun); nanochat
# auto-uses gradient accumulation. FP8 + FA3 still apply (single H100 is Hopper).
# To run on a multi-GPU node instead, set NPROC>1 below — the launcher switches
# to torchrun automatically.
#
# TIMING / COST on 1xH100 (≈8x slower than an 8xH100 node, per model):
#   - d24 baseline pretrain: ~24-28h  (!! a full day — this is the long pole)
#   - d18 clarinet pretrain: ~5-6h
#   - + SFT (~1-2h each) + evals (~1-3h)
#   total ≈ ~35-40h wall-clock, ballpark $80-110 at ~$2-2.5/H100-hr.
# Because the d24 baseline is so long, you may prefer to run the two arms as
# separate sessions (see ARM A / ARM B blocks) rather than one ~1.5-day job.
# Provision ~250GB disk (both checkpoint sets + shared data + caches).
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
NPROC=1            # GPUs to use. 1 = single H100 (this box). Set to 8 for an 8xH100 node.
DBS=16             # device-batch-size (per GPU; fits d24 on an 80GB H100)
RATIO=8            # target data:param ratio
WANDB_RUN="${WANDB_RUN:-dummy}"
BASE_TAG="d${BASE_DEPTH}-base"
IV_TAG="d${IV_DEPTH}-iv"

# Launcher: single GPU runs plain `python -m` (no torchrun, no `--` separator);
# multi-GPU uses torchrun, which needs `--` before the script's own args.
if [ "$NPROC" -gt 1 ]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m)
    SEP=(--)
else
    LAUNCH=(python -m)
    SEP=()
fi

# -----------------------------------------------------------------------------
# Environment
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Shared data + tokenizer (downloaded/trained ONCE, used by both arms).
# FineMath prep is the parallel download+reshard path (fast). climbmix supplies
# the general half; d24 @ mix 0.5 needs ~half its tokens from each source.
python -m nanochat.dataset -n 8                 # enough for tokenizer training
python -m nanochat.dataset -n 150 &             # climbmix (general) half
DL=$!
python -m clarinet.prepare_reasoning_data -n 63 &   # FineMath (reasoning) half, ~all of finemath-4plus
RP=$!

# Tokenizer is retrained once (clarinet added 3 special tokens). BOTH arms use
# this same tokenizer so vocab/BPB are directly comparable. (Trained on climbmix
# only — independent of the FineMath prep above, so they overlap fine.)
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for climbmix download..."; wait $DL
echo "Waiting for FineMath prep...";     wait $RP

# =============================================================================
# ARM A — BASELINE (classic nanochat on the mix, NO markers)   [the ~24-28h arm]
# =============================================================================
if [ -z "$SKIP_BASE" ]; then
"${LAUNCH[@]}" scripts.clarinet_train "${SEP[@]}" \
    --reasoning-mix-ratio=$MIX --no-markers \
    --depth=$BASE_DEPTH --target-param-data-ratio=$RATIO --device-batch-size=$DBS --fp8 \
    --model-tag=$BASE_TAG --run=${WANDB_RUN}-base

"${LAUNCH[@]}" scripts.base_eval "${SEP[@]}" \
    --device-batch-size=$DBS --model-tag=$BASE_TAG

"${LAUNCH[@]}" scripts.chat_sft "${SEP[@]}" \
    --model-tag=$BASE_TAG --device-batch-size=$DBS --run=${WANDB_RUN}-base

# Baseline eval is single-pass (no IV). This is the reference number.
"${LAUNCH[@]}" scripts.chat_eval "${SEP[@]}" \
    -i sft -g $BASE_TAG
fi # SKIP_BASE

# =============================================================================
# ARM B — CLARINET (same mix, WITH markers + IV)
# =============================================================================
if [ -z "$SKIP_IV" ]; then
"${LAUNCH[@]}" scripts.clarinet_train "${SEP[@]}" \
    --reasoning-mix-ratio=$MIX --p-uncond=0.1 \
    --depth=$IV_DEPTH --target-param-data-ratio=$RATIO --device-batch-size=$DBS --fp8 \
    --model-tag=$IV_TAG --run=${WANDB_RUN}-iv

"${LAUNCH[@]}" scripts.base_eval "${SEP[@]}" \
    --device-batch-size=$DBS --model-tag=$IV_TAG

# marker-aware SFT (keeps the source-marker conditioning through fine-tuning)
"${LAUNCH[@]}" scripts.clarinet_sft "${SEP[@]}" \
    --model-tag=$IV_TAG --device-batch-size=$DBS --p-uncond=0.1 --run=${WANDB_RUN}-iv

# Clarinet eval is the dual-pass IV sweep (always single-process). w=0 ->
# unconditional, w=1 -> cond-only (markers but no guidance), w>1 -> guided.
# Compare the curve against the baseline's single number above.
python -m scripts.iv_eval -i sft -g $IV_TAG \
    -a GSM8K,ARC-Easy,ARC-Challenge,MMLU,HumanEval,SpellingBee \
    --weights 0,0.5,1.0,1.5,2.0,3.0
fi # SKIP_IV

# -----------------------------------------------------------------------------
python -m nanochat.report generate
echo
echo "=================== A/B COMPLETE ==================="
echo "Baseline (single-pass) : chat_eval section for tag $BASE_TAG"
echo "Clarinet (IV sweep)    : iv_eval table for tag    $IV_TAG"
echo "Report: $CLARINET_BASE_DIR/report/report.md (also copied to ./report.md)"
echo "Checkpoints: $CLARINET_BASE_DIR/chatsft_checkpoints/{$BASE_TAG,$IV_TAG}/"

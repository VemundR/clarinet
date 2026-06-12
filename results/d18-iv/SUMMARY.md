# Clarinet d18 — results summary (v1 + inference-matched baseline + v2)

**Dates:** 2026-06-08 → 2026-06-12 · **Hardware:** 1×H100 80GB (Vast.ai) · **Author:** Vemund Rundberget

**One-line result:** Source-marker IV conditioning installs a *real* channel that roughly doubles d18 GSM8K, but it loses to depth at matched inference compute, and strengthening the conditioning's first stage ~17× does **not** improve task accuracy — a clean dissociation showing the bottleneck is model capacity, not conditioning.

## Setup

| Component | Value |
|---|---|
| Model | d18 (n_layer 18, n_embd 1152, seq 2048, vocab 32768, window `SSSL`) |
| Pretraining | 50/50 climbmix + FineMath-4plus, `p_uncond=0.1`, ratio 8, ≈2.6B tokens, FP8+FA3, 193 min, ~50% MFU |
| Pretrain quality | val bpb **0.847**, CORE **0.163** |
| SFT | `clarinet_sft` — marker-aware (step 487); marker-naive `chat_sft` (step 486) kept as a control |
| Guidance | `logit = uncond + w·(cond − uncond)`, dual-pass, constant scale |

## Findings, in causal order

### 1. Pretraining installs a real but weak first stage (`iv_probe_base.txt`)
Held-out conditional BPB orders textbook-correctly in all six cells (reasoning < unknown < general on FineMath; mirror on climbmix). Magnitude small: Δ ≈ **+0.0032 bpb** on FineMath. Relevant but weak.

### 2. Task accuracy can't see it at base level (`iv_sweep_base_x300.txt`)
Base model at the task floor (chance on ARC/MMLU, 0% GSM8K) → flat sweep is a power problem, not absence. Distribution-level probes, not task accuracy, are the valid first-stage test for base models.

### 3. Marker-naive SFT destroys the conditioning (`iv_sweep_sft_naive_full.txt`)
Fine-tuning without markers → fully flat sweep (GSM8K ≈10% at every w). Conditioning must be carried through every training stage → motivated `clarinet_sft`.

### 4. Marker-aware SFT yields a strong conditioning effect (`iv_sweep_sft_marker_aware_full.txt`)
**GSM8K 4.09% (w=0) → 11.68% (w=1)** — ~3×, n=1319, ≈8σ; concave peak at w∈[0.5,1], collapse by w=5. HumanEval *hurt* by w≥1 (channel encodes "math", not generic competence — domain specificity). Replicated across two independent trainings.

### 5. Inference-FLOPs-matched: depth beats guidance (`armB_d24_*`)
Dual-pass d18 ≈ 0.84× single-pass d24 per token (test-time-matched serving budget). The **d24 vanilla baseline wins every cell**:

| Task | **d24-base** (vanilla, 1-pass) | d18-iv w=0 | d18-iv **w=1** |
|---|---|---|---|
| GSM8K | **18.04%** | 5.08% | 10.69% |
| ARC-Easy | **51.60%** | 37.46% | 37.21% |
| ARC-Challenge | **41.04%** | 35.32% | 34.81% |
| HumanEval | **18.90%** | 13.41% | 14.02% |
| ChatCORE | **0.341** | — | ~0.25 |

GSM8K ladder: d18 vanilla ~5% → +IV ~11% (guidance ~doubles) → d24 vanilla ~18% (depth nearly doubles again, and is cheaper at inference). **Guidance helps; depth helps more, per serving dollar.** (Training compute is *not* matched — d24 trains ~2.4× more — so this favors the baseline; clarinet also needs 2× KV memory.)

### 6. v2: 17× stronger first stage → no task gain (the punchline) (`v2_repeated_markers.txt`)
Repeated markers (`CLARINET_MARKER_PERIOD=32`) reinforce the signal at every position. SFT-model probe, each model at its own period:

| | FineMath first-stage Δ | GSM8K peak |
|---|---|---|
| v1 single marker | +0.0054 | 10.7% |
| **v2 repeated** | **+0.0949 (~17×)** | 9.6% (flat) |

The first stage strengthened **~17×** — yet GSM8K did not improve (a hair lower; v2 absolute BPB also higher, 1.06 vs 0.89, as repeated markers cost ~3% of context). **First-stage relevance and downstream reasoning are decoupled:** the marker can be made dramatically more informative about the *distribution* of reasoning text without improving the model's ability to *solve* reasoning tasks.

## External calibration vs GPT-2 (same harness, `base_eval --hf-path`)

| Model | params | CORE | ARC-C | operators | cs_algo | HellaSwag |
|---|---|---|---|---|---|---|
| clarinet d18-iv | ~700M | 0.163 | 0.306 | 0.238 | 0.456 | 0.364 |
| GPT-2-large | 774M | 0.216 | 0.257 | 0.114 | 0.417 | 0.448 |
| GPT-2-XL | 1.6B | 0.254 | 0.287 | 0.157 | 0.411 | 0.512 |

Math-heavy mix beats size-matched GPT-2 on math/algorithmic slices at ~10× less data; loses on breadth — the designed trade.

## Interpretation

Three controlled results that build on each other: the IV source-conditioning channel is **real** (1, 4) but its value at this scale is **controllability, not amplification** — extrapolation past the conditional never helps. At matched inference compute it **loses to depth** (5). And directly intervening to make the first stage ~17× stronger leaves task accuracy flat (6), which rules out conditioning strength as the bottleneck by experiment, not speculation. In IV terms: **strong first-stage relevance does not imply an estimable structural effect on reasoning at d18/ratio-8.** The limiting factor is model capacity/data, not marker placement. The conditioning-placement lever is exhausted.

## Caveats

Single seed; single model size (d18, ratio 8); inference-matched (not train-matched) baseline; MMLU classed as "reasoning" in the SFT taxonomy; constant scale (`wald_scale=1`); no permuted-label placebo; v1/v2 absolute BPB not directly comparable (different layouts) — only within-model Δ is.

## Open question (deliberate, larger experiment — not a quick run)

Does the conditioning convert to task gains at **d24+ scale**, where the model has the capacity to exploit a strong first stage? That is the one lever not yet pulled, and it requires a planned multi-day run, not another tonight-on-the-box experiment. Secondary: permuted-label placebo; L1-adaptive scale vs constant control; GSM8K-only SFT taxonomy.

## Artifacts

Checkpoints + tokenizer: HF Hub `vemundr/clarinet-d18-iv` (private; base step 2475, SFT 486+487, d18-v2 SFT, tokenizer). Curves: wandb. Raw tables, GPT-2 baseline CSVs, d24 baseline report, v2 probe/sweep: this directory. v2 code: branch `clarinet/v2-repeated-markers` (PR #10).

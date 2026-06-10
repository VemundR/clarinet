# d18-iv: first full clarinet experiment — results summary

**Dates:** 2026-06-08 → 2026-06-10 · **Hardware:** 1×H100 80GB (Vast.ai) · **Author:** Vemund Rundberget

## Setup

| Component | Value |
|---|---|
| Model | d18 (n_layer 18, n_embd 1152, seq 2048, vocab 32768, window `SSSL`) |
| Pretraining | 50/50 climbmix + FineMath-4plus, `p_uncond=0.1`, ratio 8, 2,475 steps ≈ 2.6B tokens, FP8+FA3, 193 min, ~50% MFU |
| Pretrain quality | val bpb **0.847**, CORE **0.163** |
| SFT (run 1) | upstream `chat_sft` — **marker-naive** (step 486) |
| SFT (run 2) | `clarinet_sft` — **marker-aware**, `p_uncond=0.1`, reasoning tasks {GSM8K, MMLU} (step 487, 41 min, val bpb 0.325, ChatCORE 0.247) |
| Guidance | `logit = uncond + w·(cond − uncond)`, dual-pass, constant scale |

## Findings, in causal order

### 1. Pretraining installs a real but weak first stage (`iv_probe_base.txt`)
Held-out conditional BPB orders **textbook-correctly in all six cells** — reasoning < unknown < general on FineMath; general < unknown < reasoning on climbmix; the unknown marker sits at the marginal in both rows. Magnitude is small: Δ ≈ **0.0032 bpb (~0.5% rel.)** on FineMath, ~0.0007 on climbmix. The instrument is *relevant but weak* at d18/ratio-8 with a single position-1 marker.

### 2. Task accuracy can't see it at base level (`iv_sweep_base_x300.txt`)
The base model is at the task floor (chance on ARC/MMLU, 0% GSM8K), so the flat sweep there is a power problem, not evidence of absence. Distribution-level probes — not task accuracy — are the valid first-stage test for base models.

### 3. Marker-naive SFT destroys the conditioning (`iv_sweep_sft_naive_full.txt`)
Fine-tuning on `[BOS, user…]` without markers → fully flat sweep (GSM8K ≈10% at every w). The conditioning channel must be carried through every training stage.

### 4. Marker-aware SFT yields a strong conditioning channel (`iv_sweep_sft_marker_aware_full.txt`)
**GSM8K: 4.09% (w=0) → 11.68% (w=1) — ~3×, n=1319, ≈8σ.** Concave in w with peak at w∈[0.5, 1.0]; over-guidance degrades monotonically; w=5 collapses generation (SpellingBee 99.6→7.8%). HumanEval is *hurt* by w≥1 — the channel encodes "math," not generic competence (domain specificity). At its peak the marker-aware model also beats the naive SFT (GSM8K 11.68 vs ~10; ARC-C ~34.8 vs ~30.4 at every w).

## External calibration vs GPT-2 (same harness, `base_eval --hf-path`)

| Model | params | CORE | ARC-C | operators | cs_algo | HellaSwag |
|---|---|---|---|---|---|---|
| clarinet d18-iv | ~700M | 0.163 | 0.306 | 0.238 | 0.456 | 0.364 |
| GPT-2-large | 774M | 0.216 | 0.257 | 0.114 | 0.417 | 0.448 |
| GPT-2-XL | 1.6B | 0.254 | 0.287 | 0.157 | 0.411 | 0.512 |

The math-heavy mix beats size-matched GPT-2 on math/algorithmic slices at ~10× less data, and loses on breadth — the designed trade. Raw per-task CSVs in this directory.

## Interpretation

The source-marker channel works and survives fine-tuning when carried through SFT. But the value of guidance at this scale is **controllability, not amplification**: extrapolating past the conditional (w>1) never helps, consistent with weak-instrument econometrics — the pretrain first stage is too weak to define a useful direction beyond w=1. The "IV gain" realized in v1 is the conditioning itself.

## Caveats

Single seed, single model size (d18, ratio 8), no compute-matched no-marker baseline trained (naive-SFT comparison shares pretraining), MMLU classed as "reasoning" in SFT taxonomy, default `wald_scale=1`/constant scale, no permuted-label placebo. Single-marker, single-position conditioning.

## Next steps (v2)

1. Strengthen the first stage: repeated/multi-token markers, higher reasoning mix, longer training, bigger model.
2. Compute-matched no-marker baseline (and the deferred d24 comparison).
3. Permuted-label placebo run.
4. L1-adaptive scale sweep (`--scale-lo/--scale-hi`) vs constant-scale control.
5. Revisit SFT reasoning taxonomy (GSM8K-only).

## Artifacts

Checkpoints + tokenizer: HF Hub `vemundr/clarinet-d18-iv` (private; base step 2475, SFT steps 486+487, tokenizer). Curves: wandb. Raw tables + CSVs (incl. SFT-level probe `iv_probe_sft_marker_aware.txt` and GPT-2 baseline CSVs): this directory.

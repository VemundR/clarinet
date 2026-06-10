"""
Sweep the IV guidance weight `w` across evaluation tasks and report accuracy vs w.

Loads the model once. Two evaluation paths:

- Categorical tasks (ARC, MMLU): the conditional/unconditional logits at the
  answer position are weight-INDEPENDENT, so we compute them ONCE per batch
  (the expensive dual forward) and then combine + score for every w cheaply.
  This avoids re-running the model once per weight — a ~Nx speedup over the
  weight grid on these tasks (MMLU dominates).

- Generative tasks (GSM8K, HumanEval, SpellingBee): each weight produces a
  different sampled trajectory, so they genuinely must be re-run per weight.

Example:
    python -m scripts.iv_eval -i sft -g d18-iv -a GSM8K,ARC-Easy --weights 0,1,1.5,2 -x 500 -b 64
"""

import argparse
from functools import partial

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0

from clarinet.engine import ClarinetEngine
from scripts.chat_eval import run_generative_eval

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


def parse_weights(spec):
    return [float(x) for x in spec.split(",") if x]


# Task registry, mirrors scripts/chat_eval.run_chat_eval.
TASK_BUILDERS = {
    "HumanEval": HumanEval,
    "MMLU": partial(MMLU, subset="all", split="test"),
    "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "GSM8K": partial(GSM8K, subset="main", split="test"),
    "SpellingBee": partial(SpellingBee, size=256, split="test"),
}


def make_engine(model, tokenizer, iv_weight, wald_scale):
    """Engine with generate() bound to a fixed weight, for the generative path."""
    class _BoundClarinetEngine(ClarinetEngine):
        def generate(self, *args, **kwargs):
            kwargs.setdefault("iv_weight", iv_weight)
            kwargs.setdefault("wald_scale", wald_scale)
            yield from super().generate(*args, **kwargs)
        def categorical_logits_at(self, *args, **kwargs):
            kwargs.setdefault("iv_weight", iv_weight)
            kwargs.setdefault("wald_scale", wald_scale)
            return super().categorical_logits_at(*args, **kwargs)
    return _BoundClarinetEngine(model, tokenizer)


def sweep_categorical(task_object, tokenizer, engine, weights, batch_size, max_problems, wald_scale):
    """
    Cached categorical sweep: per batch, compute the dual-pass logits ONCE, then
    score every weight from the cached (cond, uncond) logits. Returns {w: acc}.
    Single-process (iv_eval is run on one GPU).
    """
    # max_problems takes the FIRST n examples — unbiased because every task
    # shuffles with a fixed seed at load time (ds.shuffle(seed=42) in tasks/).
    n = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda a, b: -(-a // b)
    num_batches = ceil_div(n, batch_size)
    letter_cache = {}
    passed = {w: 0 for w in weights}
    total = 0

    for bi in range(num_batches):
        i0, i1 = bi * batch_size, min((bi + 1) * batch_size, n)
        conversations = [task_object[i] for i in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(c) for c in conversations]
        answer_positions = [len(ids) - 1 for ids in prompt_ids]

        # The expensive part — run the two forward passes once for this batch.
        cond_at, uncond_at = engine.categorical_dual_logits(prompt_ids, answer_positions)
        # Scale is weight-independent too (constant CFG: scale_lo == scale_hi == 1).
        s = engine.l1_adaptive_scale(cond_at, uncond_at, wald_scale, 1.0, 1.0)

        # Precompute candidate-letter token ids per conversation (once, not per w).
        conv_letters, conv_letter_ids = [], []
        for c in conversations:
            letters = c["letters"]
            ids = []
            for letter in letters:
                if letter not in letter_cache:
                    enc = tokenizer.encode(letter)
                    assert len(enc) == 1, "Each letter must be a single token"
                    letter_cache[letter] = enc[0]
                ids.append(letter_cache[letter])
            conv_letters.append(letters)
            conv_letter_ids.append(ids)

        # Cheap per-weight combine + score from the cached logits.
        for w in weights:
            combined = engine.combine_logits(cond_at, uncond_at, w, s)  # (B, V)
            for idx, c in enumerate(conversations):
                focus = combined[idx, conv_letter_ids[idx]]
                pred = conv_letters[idx][focus.argmax().item()]
                passed[w] += int(task_object.evaluate(c, pred))
        total += len(conversations)
        if (bi + 1) % 25 == 0 or i1 == n:
            print0(f"  {task_object.__class__.__name__}: {total}/{n}")

    return {w: passed[w] / total for w in weights}


def main():
    parser = argparse.ArgumentParser(description="IV guidance weight sweep")
    parser.add_argument("-i", "--source", type=str, required=True, help="Model source: base|sft|rl")
    parser.add_argument("-a", "--task-names", type=str, default="GSM8K,ARC-Easy,ARC-Challenge,MMLU,HumanEval,SpellingBee",
                        help="Comma-separated task names from the registry.")
    parser.add_argument("--weights", type=str, default="0,0.5,1.0,1.5,2.0,3.0,5.0",
                        help="Comma-separated IV guidance weights to sweep.")
    parser.add_argument("--wald-scale", type=float, default=1.0)
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("-x", "--max-problems", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local, _world, device = compute_init(device_type)

    model, tokenizer, _meta = load_model(args.source, device, phase="eval",
                                         model_tag=args.model_tag, step=args.step)

    weights = parse_weights(args.weights)
    task_names = [t for t in args.task_names.split(",") if t]

    # results[task][weight] = accuracy
    results = {t: {} for t in task_names}
    # One unbound engine for the categorical sweep (weight passed explicitly).
    base_engine = ClarinetEngine(model, tokenizer)

    for task_name in task_names:
        task_object = TASK_BUILDERS[task_name]()
        if task_object.eval_type == "categorical":
            print0(f"\n========== {task_name} (categorical, cached sweep over {len(weights)} weights) ==========")
            accs = sweep_categorical(task_object, tokenizer, base_engine, weights,
                                     args.batch_size, args.max_problems, args.wald_scale)
            for w in weights:
                results[task_name][w] = accs[w]
                print0(f"  {task_name} w={w:.2f}: {100*accs[w]:.2f}%")
        else:  # generative — must re-run per weight (sampled trajectories diverge)
            print0(f"\n========== {task_name} (generative, per-weight) ==========")
            for w in weights:
                engine = make_engine(model, tokenizer, iv_weight=w, wald_scale=args.wald_scale)
                acc = run_generative_eval(task_object, tokenizer, model, engine,
                                          args.num_samples, args.max_new_tokens,
                                          args.temperature, args.top_k, max_problems=args.max_problems)
                results[task_name][w] = acc
                print0(f"  {task_name} w={w:.2f}: {100*acc:.2f}%")

    print0("\n========== Summary (rows=task, cols=w) ==========")
    header = "task".ljust(18) + "".join(f"{w:>9.2f}" for w in weights)
    print0(header)
    for task in task_names:
        row = task.ljust(18) + "".join(f"{100*results[task][w]:>8.2f}%" for w in weights)
        print0(row)

    compute_cleanup()


if __name__ == "__main__":
    main()

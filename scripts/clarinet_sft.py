"""
Marker-aware SFT entry point for clarinet.

Wraps scripts.chat_sft so the chat model keeps the source-marker conditioning
that was installed during pretraining. Without this, the upstream SFT trains on
conversations laid out as [BOS, <user_start>, ...] (no marker), which washes out
the pretraining-time conditioning — and then iv_eval bolts a marker on at
inference that the SFT model never saw. This shim makes the SFT layout match
pretraining and inference: [BOS, marker, <user_start>, ...].

Two hooks, patched before importing chat_sft (so its module-level
`from tasks.common import TaskMixture` and tokenizer use the patched versions):

  1. TaskMixture.get_example tags each conversation with `_is_reasoning` based on
     its SOURCE task (GSM8K / MMLU -> reasoning; SmolTalk / identity / spelling
     -> general). This mirrors the pretraining instrument (FineMath = reasoning,
     climbmix = general) at the conversation level, so the reasoning vs unknown
     contrast that iv_eval amplifies is actually meaningful.

  2. RustBPETokenizer.render_conversation splices the chosen marker in at
     position 1 (right after BOS), mask 0 (a conditioning input, never a training
     target). With probability p_uncond the true marker is replaced by
     <|src_unknown|> — the CFG-style dropout that defines the unconditional pass.

Usage (single GPU):
  python -m scripts.clarinet_sft --model-tag=d18-iv --device-batch-size=16 --p-uncond=0.1

Everything after the clarinet-specific flags is forwarded to chat_sft unchanged.
"""

import argparse
import random
import sys

import tasks.common as _tasks_common
import nanochat.tokenizer as _tokenizer_mod

from clarinet.dataloader import SRC_GENERAL, SRC_REASONING, SRC_UNKNOWN

# Source tasks whose conversations get the <|src_reasoning|> marker. Everything
# else in the SFT mixture gets <|src_general|>. Matched by Task subclass name.
# Aligns with the FineMath (=reasoning) pretraining instrument.
REASONING_TASK_NAMES = {"GSM8K", "MMLU"}


def _splice_marker(ids, mask, marker_id, max_tokens):
    """Insert marker_id at position 1 (after BOS); mask 0 so it's a pure
    conditioning input, never a training target. Re-truncate to max_tokens."""
    ids = [ids[0], marker_id, *ids[1:]]
    mask = [mask[0], 0, *mask[1:]]
    return ids[:max_tokens], mask[:max_tokens]


def _parse_and_strip_clarinet_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--p-uncond", type=float, default=0.1,
                     help="Probability of overriding the true source marker with <|src_unknown|>.")
    pre.add_argument("--clarinet-seed", type=int, default=0,
                     help="Seed for the per-conversation p_uncond dropout RNG.")
    clarinet_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return clarinet_args


def _install(p_uncond, seed):
    # --- Hook 1: tag each conversation with its source's is_reasoning flag ---
    OrigMixture = _tasks_common.TaskMixture

    class MarkerTaskMixture(OrigMixture):
        def get_example(self, index):
            task_idx, local_idx = self.index_map[index]
            conv = self.tasks[task_idx][local_idx]
            conv = dict(conv)  # shallow copy; don't mutate the task's dict
            conv["_is_reasoning"] = type(self.tasks[task_idx]).__name__ in REASONING_TASK_NAMES
            return conv

    _tasks_common.TaskMixture = MarkerTaskMixture

    # --- Hook 2: splice the marker into the rendered conversation ---
    Tokenizer = _tokenizer_mod.RustBPETokenizer
    orig_render = Tokenizer.render_conversation
    rng = random.Random(seed)

    def render_conversation(self, conversation, max_tokens=2048):
        is_reasoning = conversation.get("_is_reasoning") if isinstance(conversation, dict) else None
        ids, mask = orig_render(self, conversation, max_tokens=max_tokens)
        if is_reasoning is None:
            return ids, mask  # not from our tagged mixture -> behave like upstream
        if rng.random() < p_uncond:
            marker = self.encode_special(SRC_UNKNOWN)
        else:
            marker = self.encode_special(SRC_REASONING if is_reasoning else SRC_GENERAL)
        return _splice_marker(ids, mask, marker, max_tokens)

    Tokenizer.render_conversation = render_conversation


if __name__ == "__main__":
    clarinet_args = _parse_and_strip_clarinet_args()
    _install(clarinet_args.p_uncond, clarinet_args.clarinet_seed)
    print(f"[clarinet_sft] marker-aware SFT: reasoning tasks={sorted(REASONING_TASK_NAMES)}, "
          f"p_uncond={clarinet_args.p_uncond}")
    import scripts.chat_sft  # noqa: F401  (runs at import time)

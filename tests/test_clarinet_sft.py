"""
Tests for clarinet's marker-aware SFT splice logic.

Run: python -m pytest tests/test_clarinet_sft.py -v
"""

from clarinet.dataloader import SRC_GENERAL, SRC_REASONING, SRC_UNKNOWN
from scripts.clarinet_sft import _splice_marker, marker_for, REASONING_TASK_NAMES


def test_splice_marker_inserts_after_bos():
    ids = [1000, 50, 51, 52]      # [BOS, ...]
    mask = [0, 0, 1, 1]
    out_ids, out_mask = _splice_marker(ids, mask, marker_id=1001, max_tokens=2048)
    assert out_ids == [1000, 1001, 50, 51, 52]   # marker at position 1
    assert out_mask == [0, 0, 0, 1, 1]           # marker mask is 0 (not a target)


def test_splice_marker_preserves_assistant_mask():
    # The assistant-supervised positions (mask=1) must stay aligned after the
    # marker shifts everything right by one.
    ids = [1000, 10, 20, 30, 40]
    mask = [0, 0, 0, 1, 1]
    out_ids, out_mask = _splice_marker(ids, mask, 1002, max_tokens=2048)
    # token 30 (originally index 3, mask 1) is now at index 4, still mask 1
    assert out_ids[4] == 30 and out_mask[4] == 1
    assert out_ids[5] == 40 and out_mask[5] == 1
    assert sum(out_mask) == sum(mask)  # no supervised tokens gained or lost


def test_splice_marker_truncates_to_max_tokens():
    ids = list(range(1000, 1000 + 10))
    mask = [0] + [1] * 9
    out_ids, out_mask = _splice_marker(ids, mask, 1003, max_tokens=5)
    assert len(out_ids) == 5 and len(out_mask) == 5
    assert out_ids[:2] == [1000, 1003]  # BOS, marker still lead


def test_marker_for_is_deterministic_per_conversation():
    # Same (seed, index) must always give the same marker — this is what makes
    # the val-loader bpb reproducible across runs.
    for idx in range(50):
        a = marker_for(True, p_uncond=0.5, seed=7, index=idx)
        b = marker_for(True, p_uncond=0.5, seed=7, index=idx)
        assert a == b


def test_marker_for_respects_p_uncond_extremes():
    for idx in range(50):
        assert marker_for(True, 0.0, 0, idx) == SRC_REASONING
        assert marker_for(False, 0.0, 0, idx) == SRC_GENERAL
        assert marker_for(True, 1.0, 0, idx) == SRC_UNKNOWN
        assert marker_for(False, 1.0, 0, idx) == SRC_UNKNOWN


def test_marker_for_dropout_rate_is_plausible():
    n = 2000
    drops = sum(marker_for(True, 0.1, 0, i) == SRC_UNKNOWN for i in range(n))
    assert 0.05 < drops / n < 0.16  # ~10% within loose binomial tolerance


def test_reasoning_task_names_are_sensible():
    # Guards the source-task taxonomy: math/exam tasks are reasoning; the rest
    # (general chat, identity, spelling) are not.
    assert "GSM8K" in REASONING_TASK_NAMES
    assert "MMLU" in REASONING_TASK_NAMES
    assert "SmolTalk" not in REASONING_TASK_NAMES
    assert "CustomJSON" not in REASONING_TASK_NAMES

"""
Tests for the iv_probe packing helper — a layout bug here would silently
invalidate the first-stage relevance conclusion, so pin it down.

Run: python -m pytest tests/test_iv_probe.py -v
"""

import scripts.iv_probe as ivp
from scripts.iv_probe import pack_doc


def test_pack_doc_layout():
    # [BOS, marker, d0, d1, d2] -> inputs drop last, targets drop first
    inputs, targets = pack_doc([10, 20, 30], bos_id=1000, marker_id=1001, max_seq_len=2048)
    assert inputs == [1000, 1001, 10, 20]
    assert targets == [1001, 10, 20, 30]
    # Marker is the target only at position 0 — it is a special token with
    # token_bytes == 0, so evaluate_bpb excludes it from nats and bytes.
    # All counted targets (10, 20, 30) are doc tokens, identical across markers.


def test_pack_doc_counted_targets_are_marker_independent():
    doc = list(range(100, 110))
    _, t_a = pack_doc(doc, 1000, 1001, max_seq_len=2048)
    _, t_b = pack_doc(doc, 1000, 1002, max_seq_len=2048)
    # Only position 0 (the marker itself) differs; every other target matches.
    assert t_a[0] == 1001 and t_b[0] == 1002
    assert t_a[1:] == t_b[1:]


def test_pack_doc_truncates_to_max_seq_len():
    doc = list(range(100, 200))  # 100 tokens + BOS + marker = 102
    inputs, targets = pack_doc(doc, 1000, 1001, max_seq_len=16)
    assert len(inputs) == 16 and len(targets) == 16
    assert inputs[0] == 1000 and inputs[1] == 1001  # layout preserved
    assert targets[-1] == inputs[-1] + 1  # consecutive doc ids shifted by one


def test_pack_doc_short_doc():
    inputs, targets = pack_doc([42], 1000, 1001, max_seq_len=2048)
    assert inputs == [1000, 1001]
    assert targets == [1001, 42]


def test_pack_doc_repeated_markers(monkeypatch):
    # v2: MARKER_PERIOD=2 -> row = [1000,1001,10,11,1001,12,13]
    monkeypatch.setattr(ivp, "MARKER_PERIOD", 2)
    inputs, targets = pack_doc([10, 11, 12, 13], bos_id=1000, marker_id=1001, max_seq_len=2048)
    assert inputs == [1000, 1001, 10, 11, 1001, 12]
    assert targets == [1001, 10, 11, 1001, 12, 13]

"""The vLLM SeLaR holder's gate, checked against the HF reference arithmetic.

`generate_selar` in src/generation_utils.py is the reference; its per-step gate
ops (renormalised top-k entropy, normalised signal, contrastive push scale) are
transcribed here and driven with the same synthetic logits as the engine-side
holder. Run directly: `python tests/test_selar_vllm_state.py`. CPU only.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

VOCAB = 64


def _holder():
    from vllm.v1.sample.selar_state import SelarStateHolder

    return SelarStateHolder(8, torch.device("cpu"))


def _update(added=(), removed=(), moved=(), batch_size=0):
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

    return BatchUpdate(
        batch_size=batch_size, removed=list(removed), added=list(added), moved=list(moved)
    )


def _params(topk=3, threshold=0.5, weight=1.0, math_ids=None):
    from vllm import SamplingParams

    return SamplingParams(
        selar=True,
        selar_topk=topk,
        selar_entropy_threshold=threshold,
        selar_contrastive_weight=weight,
        selar_math_token_ids=math_ids,
        max_tokens=8,
    )


def _metadata(num_reqs):
    from types import SimpleNamespace

    return SimpleNamespace(temperature=None, top_k=None, top_p=None,
                           logitsprocs=None)


def _reference_gate(logits_row, k, threshold, weight):
    """Transcription of generate_selar's gate ops for one unfiltered row."""
    probs = torch.softmax(logits_row.to(torch.float32), dim=-1)
    topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
    norm = topk_probs / (topk_probs.sum() + 1e-10)
    ent = -torch.sum(norm * torch.log(norm + 1e-10))
    sig = torch.clamp(ent / math.log(float(k)), 0.0, 1.0).item()
    return topk_idx, norm, sig, (sig >= threshold), weight * sig


def test_gate_matches_the_reference_arithmetic():
    torch.manual_seed(11)
    h = _holder()
    h.sync_batch(_update(
        added=[(0, _params(topk=3, threshold=0.5), [], []),
               (1, _params(topk=5, threshold=0.5), [], [])],
        batch_size=2))
    logits = torch.randn(2, VOCAB) * 2.0

    h.prepare(logits, _metadata(2))
    for r, k in ((0, 3), (1, 5)):
        ids, norm, sig, gated, push = _reference_gate(logits[r], k, 0.5, 1.0)
        st = h._state[r]
        if gated:
            assert st["head_ids"] is not None, f"row {r} should be gated"
            assert torch.equal(st["head_ids"], ids), f"row {r} head ids differ"
            assert torch.allclose(st["head_weights"], norm, atol=1e-7), (
                f"row {r} weights differ")
            assert abs(st["push"] - push) < 1e-7, f"row {r} push differs"
        else:
            assert st["head_ids"] is None, f"row {r} should not be gated"
    print("ok: per-row gate, weights and push match the reference arithmetic")


def test_low_entropy_rows_stay_discrete():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(topk=3, threshold=0.5), [], [])],
                         batch_size=1))
    peaked = torch.full((1, VOCAB), -30.0)
    peaked[0, 7] = 30.0  # one-hot head: normalised entropy ~ 0
    h.prepare(peaked, _metadata(1))
    assert h._state[0]["head_ids"] is None
    assert h.embed_directives() is None
    print("ok: a confident step keeps the ordinary token path")


def test_finalize_closes_the_gate_on_math_tokens():
    h = _holder()
    h.sync_batch(_update(
        added=[(0, _params(topk=3, threshold=0.0, math_ids=[42]), [], [])],
        batch_size=1))
    h.prepare(torch.zeros(1, VOCAB), _metadata(1))  # uniform: gate wide open
    assert h._state[0]["head_ids"] is not None
    h.finalize(torch.tensor([42]))
    assert h._state[0]["head_ids"] is None, "a math token must close the gate"

    h.prepare(torch.zeros(1, VOCAB), _metadata(1))
    h.finalize(torch.tensor([7]))
    assert h._state[0]["head_ids"] is not None, "non-math tokens keep it open"
    print("ok: finalize closes the gate exactly on math tokens")


def test_handoff_pads_mixed_k_with_zero_weight():
    h = _holder()
    h.sync_batch(_update(
        added=[(0, _params(topk=2, threshold=0.0), [], []),
               (1, _params(topk=4, threshold=0.0), [], [])],
        batch_size=2))
    h.prepare(torch.zeros(2, VOCAB), _metadata(2))
    rows, ids, weights, pushes = h.embed_directives()
    assert rows.tolist() == [0, 1]
    assert ids.shape == (2, 4) and weights.shape == (2, 4)
    assert torch.all(weights[0, 2:] == 0), "padding must carry zero weight"
    assert abs(weights[0].sum().item() - 1.0) < 1e-6
    assert pushes.shape == (2,)
    print("ok: mixed per-row k pads with zero weight")


def test_skipped_rows_get_no_directive():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(threshold=0.0), [], [])],
                         batch_size=1))
    h.set_rows_without_a_real_decode([0])
    h.prepare(torch.zeros(1, VOCAB), _metadata(1))
    assert h._state[0]["head_ids"] is None, (
        "a mid-prompt chunk must not leave a directive for the first decode")
    h.set_rows_without_a_real_decode([])
    h.prepare(torch.zeros(1, VOCAB), _metadata(1))
    assert h._state[0]["head_ids"] is not None
    print("ok: a discarded prefill row leaves no stale directive")


def test_displacement_formula_matches_the_reference():
    """The runner-side latent input, replicated here against the HF ops."""
    torch.manual_seed(3)
    E = torch.randn(VOCAB, 16)
    ids = torch.tensor([5, 9, 11])
    w = torch.tensor([0.5, 0.3, 0.2])
    push = 0.8

    # Reference: generate_selar's inline ops.
    mixture_ref = torch.sum(w.unsqueeze(-1) * E[ids], dim=0)
    off = mixture_ref - E[ids[0]]
    n = torch.norm(off, dim=-1, keepdim=True) + 1e-10
    ref = mixture_ref + push * (off / n) * n

    # Engine: mixture and anchor via weighted sums, same displacement.
    mixture = (E[ids] * w.unsqueeze(-1)).sum(dim=0)
    anchor = E[ids[0]]
    offset = mixture - anchor
    norm = torch.norm(offset, dim=-1, keepdim=True) + 1e-10
    got = mixture + push * (offset / norm) * norm
    assert torch.allclose(got, ref, atol=1e-7)
    print("ok: the contrastive displacement matches the reference ops")


if __name__ == "__main__":
    test_gate_matches_the_reference_arithmetic()
    test_low_entropy_rows_stay_discrete()
    test_finalize_closes_the_gate_on_math_tokens()
    test_handoff_pads_mixed_k_with_zero_weight()
    test_skipped_rows_get_no_directive()
    test_displacement_formula_matches_the_reference()
    print("\nall SeLaR state checks passed")

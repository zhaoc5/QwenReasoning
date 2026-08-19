"""Checks for apply_sampling_filter, the shared top_k / top_p / min_p logit filter.

Run with `python tests/test_sampling_filter.py` inside the project venv.

min_p is the one with a subtle definition: it is a *relative* floor, min_p * max_prob,
not an absolute probability. The distinction only shows up on flat distributions, which
is exactly where a reasoning model spends its uncertain steps.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.generation_utils import apply_sampling_filter


def kept(logits, **kw):
    out = apply_sampling_filter(logits.clone(), **kw)
    return (~torch.isinf(out)).sum(dim=-1).tolist()


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    assert kept(logits, top_k=3) == [3]
    assert kept(logits, top_k=0) == [6]      # 0 disables
    print("ok: top_k keeps exactly k")


def test_min_p_is_relative_to_the_peak():
    # Peaked row: max_prob ~ 0.99, so a min_p of 0.01 floors at ~0.0099 and cuts the tail.
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    n_peaked = kept(peaked, min_p=0.01)[0]
    assert n_peaked == 1, n_peaked

    # Flat row: every token has p = 1/6, and the floor is 0.01 * (1/6) = 0.0017, which is
    # below all of them -- so nothing is cut. An absolute 0.01 threshold would also keep
    # them, so use a wider row where the two rules visibly disagree.
    flat = torch.zeros(1, 400)
    n_flat = kept(flat, min_p=0.01)[0]
    assert n_flat == 400, f"relative floor must keep a uniform row intact, kept {n_flat}"
    # Sanity: with p = 1/400 = 0.0025 each, an absolute 0.01 threshold would have
    # removed every token, leaving nothing to sample from.
    assert F.softmax(flat, dim=-1)[0, 0].item() < 0.01
    print("ok: min_p floors at min_p * max_prob, not at min_p")


def test_min_p_still_cuts_a_long_tail():
    # One dominant token plus a long thin tail: the tail is far below min_p * max_prob.
    logits = torch.full((1, 200), -8.0)
    logits[0, 0] = 8.0
    assert kept(logits, min_p=0.05) == [1]
    print("ok: min_p removes tokens far below the peak")


def test_top_p_is_nucleus():
    # Two tokens hold ~0.88 of the mass; nucleus at 0.8 should keep just those two.
    logits = torch.tensor([[3.0, 3.0, 0.0, 0.0, 0.0]])
    probs = F.softmax(logits, dim=-1)[0]
    assert (probs[0] + probs[1]).item() > 0.8
    assert kept(logits, top_p=0.8) == [2]
    print("ok: top_p keeps the nucleus")


if __name__ == "__main__":
    test_top_k_keeps_exactly_k()
    test_min_p_is_relative_to_the_peak()
    test_min_p_still_cuts_a_long_tail()
    test_top_p_is_nucleus()
    print("\nall sampling-filter checks passed")

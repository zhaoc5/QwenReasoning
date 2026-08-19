"""Numeric parity between the vLLM Soft Thinking kernel and the HF reference loop.

The vLLM backend is meant to be a faster route to the *same* method, so the arithmetic has
to agree with `src/generation_utils.py:generate_soft`, which is the implementation the
harness's published Soft Thinking numbers came from. This compares the two directly on
random logits: same filters, same mixture, same entropy, same Cold Stop bookkeeping.

Run directly: `python tests/test_soft_thinking_vllm_parity.py`. Needs torch and vLLM
importable; no GPU and no weights.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.generation_utils import apply_sampling_filter

VOCAB = 512
BATCH = 6
SOFT_TOPK = 10


def _hf_reference(logits, temperature, top_k, top_p, min_p, soft_topk):
    """Exactly the arithmetic generate_soft performs, lifted out of its decode loop."""
    scaled = logits / temperature
    filtered = apply_sampling_filter(scaled.clone(), top_k=top_k, top_p=top_p, min_p=min_p)
    probs = F.softmax(filtered, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=min(soft_topk, probs.shape[-1]), dim=-1)
    weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
    full_probs = F.softmax(scaled, dim=-1)
    entropy = -(full_probs * torch.log(full_probs.clamp_min(1e-12))).sum(dim=-1)
    return topk_indices, weights, entropy


def _vllm_side(logits, temperature, top_k, top_p, min_p, soft_topk):
    from vllm.v1.sample.soft_thinking import compute_concept_tokens

    n = logits.shape[0]
    return compute_concept_tokens(
        logits,
        temperature=torch.full((n,), temperature),
        soft_topk=soft_topk,
        top_k=torch.full((n,), top_k, dtype=torch.long) if top_k > 0 else None,
        top_p=torch.full((n,), top_p) if top_p < 1.0 else None,
        min_p=torch.full((n,), min_p) if min_p > 0 else None,
    )


def test_concept_tokens_match_the_reference():
    torch.manual_seed(0)
    recipes = [
        # (temperature, top_k, top_p, min_p) -- the paper's recipe, Qwen3.5's, and no filters
        (0.6, 30, 1.0, 0.001),
        (1.0, 20, 0.95, 0.0),
        (1.0, 0, 1.0, 0.0),
    ]
    for temperature, top_k, top_p, min_p in recipes:
        logits = torch.randn(BATCH, VOCAB) * 3.0
        ids_ref, w_ref, ent_ref = _hf_reference(
            logits, temperature, top_k, top_p, min_p, SOFT_TOPK
        )
        got = _vllm_side(logits, temperature, top_k, top_p, min_p, SOFT_TOPK)

        tag = f"T{temperature}/k{top_k}/p{top_p}/minp{min_p}"
        assert torch.equal(got.topk_ids, ids_ref), f"{tag}: mixture components differ"
        assert torch.allclose(got.topk_probs, w_ref, atol=1e-6), f"{tag}: weights differ"
        assert torch.allclose(got.entropy, ent_ref, atol=1e-5), f"{tag}: entropy differs"
        assert torch.equal(got.argmax_ids, ids_ref[:, 0]), f"{tag}: argmax differs"
        # The weights are a distribution: that is what makes the mixture a convex
        # combination of embeddings rather than an arbitrary vector.
        assert torch.allclose(
            got.topk_probs.sum(-1), torch.ones(BATCH), atol=1e-5
        ), f"{tag}: weights do not sum to 1"
    print("ok: mixture, weights and entropy match generate_soft on all three recipes")


def test_entropy_ignores_the_filters():
    """The regression that made Cold Stop inert: entropy must see the full vocabulary."""
    from vllm.v1.sample.soft_thinking import compute_concept_tokens

    torch.manual_seed(1)
    logits = torch.randn(BATCH, VOCAB) * 3.0
    n = BATCH
    loose = compute_concept_tokens(
        logits, temperature=torch.ones(n), soft_topk=SOFT_TOPK
    )
    tight = compute_concept_tokens(
        logits,
        temperature=torch.ones(n),
        soft_topk=SOFT_TOPK,
        top_k=torch.full((n,), 1, dtype=torch.long),
        top_p=torch.full((n,), 0.5),
    )
    # top_k=1 collapses the filtered distribution to a point; had the entropy been taken
    # after the filters it would read exactly 0 and Cold Stop would fire on every step.
    assert torch.allclose(loose.entropy, tight.entropy, atol=1e-6), (
        "entropy changed with the filters -- it is being measured after them"
    )
    assert (tight.entropy > 0.1).all(), f"entropy collapsed: {tight.entropy}"
    print("ok: entropy is taken on the full distribution, not the filtered one")


def test_cold_stop_counts_runs_and_resets():
    from vllm.v1.sample.soft_thinking import update_cold_stop

    n, tau, patience, eot = 3, 0.01, 3, 999
    thr = torch.full((n,), tau)
    pat = torch.full((n,), patience)
    end = torch.full((n,), eot)
    in_thinking = torch.ones(n, dtype=torch.bool)
    steps = torch.zeros(n, dtype=torch.long)
    other = torch.zeros(n, dtype=torch.long)  # an ordinary token, not </think>

    # Row 0 stays low-entropy; row 1 gets interrupted once; row 2 emits </think> at step 2.
    seq = [
        (torch.tensor([0.001, 0.001, 0.001]), other),
        (torch.tensor([0.001, 1.000, 0.001]), torch.tensor([0, 0, eot])),
        (torch.tensor([0.001, 0.001, 0.001]), other),
        (torch.tensor([0.001, 0.001, 0.001]), other),
    ]
    fired_at = {}
    for step, (entropy, argmax) in enumerate(seq):
        in_thinking, steps, fired = update_cold_stop(
            in_thinking, steps, entropy, argmax, thr, pat, end
        )
        for r in range(n):
            if fired[r] and r not in fired_at:
                fired_at[r] = step

    assert fired_at.get(0) == 2, f"row 0 should fire on its 3rd low step, got {fired_at}"
    assert 1 not in fired_at, "row 1's run was interrupted, so it must not fire"
    assert 2 not in fired_at, "row 2 left thinking on </think>, so Cold Stop must not fire"
    assert not in_thinking[0] and not in_thinking[2], "rows 0 and 2 should have left"
    assert in_thinking[1], "row 1 should still be thinking"
    print("ok: Cold Stop counts consecutive steps, resets, and yields to a real </think>")



def test_disabled_top_k_rows_cut_nothing():
    """Off is spelled two ways: <= 0 by convention, >= vocab by vLLM's input
    batch (it stores vocab_size for requests without top_k). Both must leave
    the row unfiltered, and neither may drag other rows' top_k wider."""
    from vllm.v1.sample.soft_thinking import compute_concept_tokens

    torch.manual_seed(3)
    logits = torch.randn(3, VOCAB) * 3.0
    ones = torch.ones(3)

    unfiltered = compute_concept_tokens(logits, temperature=ones, soft_topk=SOFT_TOPK)
    mixed = compute_concept_tokens(
        logits,
        temperature=ones,
        soft_topk=SOFT_TOPK,
        # Row 0: real top_k=2. Rows 1 and 2: off, in both spellings.
        top_k=torch.tensor([2, 0, VOCAB], dtype=torch.long),
    )

    for row in (1, 2):
        assert torch.equal(mixed.topk_ids[row], unfiltered.topk_ids[row]), (
            f"row {row} (top_k off) was filtered"
        )
        assert torch.allclose(mixed.topk_probs[row], unfiltered.topk_probs[row],
                              atol=1e-6), f"row {row} (top_k off) reweighted"
    # Row 0 keeps exactly two components; the rest of its mass is zero.
    assert (mixed.topk_probs[0, 2:] < 1e-9).all(), (
        f"top_k=2 row kept more than two components: {mixed.topk_probs[0]}"
    )
    assert mixed.topk_probs[0, :2].sum().item() > 1 - 1e-6
    print("ok: disabled top_k rows pass through unfiltered in a mixed batch")


if __name__ == "__main__":
    test_concept_tokens_match_the_reference()
    test_entropy_ignores_the_filters()
    test_cold_stop_counts_runs_and_resets()
    test_disabled_top_k_rows_cut_nothing()
    print("\nall vLLM Soft Thinking parity checks passed")

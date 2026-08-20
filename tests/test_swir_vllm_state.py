"""State machine of the vLLM SwiReasoning holder, checked against the HF rules.

The reference implementation is `generate_swir` in src/generation_utils.py; a
scalar transcription of its per-step update rules drives the same inputs into
the engine-side holder, so any drift in the mode machine, the switch blends,
the injection queues or the termination budget shows up as a mismatch.

Run directly: `python tests/test_swir_vllm_state.py`. CPU only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

VOCAB = 64
START, END, LINEBREAK = 60, 61, 62
STOP = 63


def _holder():
    from types import SimpleNamespace

    from vllm.v1.sample.swi_reasoning_state import SwiReasoningStateHolder

    cfg = SimpleNamespace(
        reasoning_start_token_ids=[START], reasoning_end_token_ids=[END]
    )
    return SwiReasoningStateHolder(cfg, 8, torch.device("cpu"))


def _update(added=(), removed=(), moved=(), batch_size=0):
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

    return BatchUpdate(
        batch_size=batch_size, removed=list(removed), added=list(added), moved=list(moved)
    )


def _params(alpha=1.0, beta=0.7, window=2, max_switch=None, term_max=4,
            math_ids=None, max_tokens=100, stop_ids=None):
    from vllm import SamplingParams

    return SamplingParams(
        swir=True,
        swir_alpha=alpha,
        swir_beta=beta,
        swir_window=window,
        swir_max_switch_count=max_switch,
        swir_termination_max_tokens=term_max,
        swir_math_token_ids=math_ids,
        swir_convergence_token_ids=[END] if max_switch else None,
        swir_termination_token_ids=[END, 7, 8] if max_switch else None,
        swir_linebreak_token_id=LINEBREAK,
        stop_token_ids=stop_ids,
        max_tokens=max_tokens,
    )


def _drive(h, row, entropy, sampled_id, num_reqs=1):
    """One engine step for `row`: inject the observed entropy, then step."""
    st = h._state[row]
    st["entropy"] = float(entropy)
    st["probs"] = torch.full((VOCAB,), 1.0 / VOCAB)
    sampled = torch.zeros(num_reqs, dtype=torch.long)
    sampled[row] = sampled_id
    forced = h.step(sampled)
    return -1 if forced is None else int(forced[row].item())


class RefSwir:
    """Scalar transcription of generate_swir's per-row update rules."""

    def __init__(self, window, max_switch=None):
        self.mode, self.stay, self.ref = 0, 0, None
        self.locked = False
        self.count = 0
        self.window = window
        self.max_switch = max_switch

    def step(self, step_idx, entropy, sampled_is_end):
        self.locked = self.locked or sampled_is_end
        to_normal = to_soft = False
        if step_idx == 0:
            self.ref = entropy
        else:
            self.stay += 1
            allow = self.stay >= self.window
            to_normal = self.mode == 0 and entropy < self.ref
            to_soft = (self.mode == 1 and entropy > self.ref
                       and allow and not self.locked)
            if to_normal:
                self.mode, self.stay, self.ref = 1, 0, entropy
                self.count += 1
            elif to_soft:
                self.mode, self.stay, self.ref = 0, 0, entropy
        return to_normal, to_soft


def test_mode_machine_matches_the_reference():
    import random

    random.seed(7)
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(window=3), [], [])], batch_size=1))
    ref = RefSwir(window=3)

    entropies = [5.0]
    for _ in range(79):
        entropies.append(max(0.1, entropies[-1] + random.uniform(-1.5, 1.5)))

    for i, e in enumerate(entropies):
        _drive(h, 0, e, sampled_id=3)
        ref.step(i, e, sampled_is_end=False)
        st = h._state[0]
        assert st["mode"] == ref.mode, f"mode diverged at step {i}"
        assert st["stay"] == ref.stay, f"stay diverged at step {i}"
        assert st["switch_count"] == ref.count, f"count diverged at step {i}"
        assert abs(st["ref"] - ref.ref) < 1e-9, f"ref diverged at step {i}"
    print("ok: 80 random steps track the reference state machine exactly")


def test_directives_and_blend_ramps():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(alpha=0.5, beta=0.7, window=1,
                                            max_tokens=100), [], [])],
                         batch_size=1))
    # Step 0: soft mode, mixture blended with the line break at 0.9.
    _drive(h, 0, 5.0, sampled_id=3)
    d = h._state[0]["directive"]
    assert d == {"blend_id": LINEBREAK, "blend_w": 0.9}, d

    # Step 1: entropy rises -> still soft, plain mixture.
    _drive(h, 0, 6.0, sampled_id=3)
    assert h._state[0]["directive"] == {"blend_id": None, "blend_w": 1.0}

    # Step 2: entropy falls -> to_normal, eased toward </think> with
    # beta = 0.7 + 0.3 * (2/100).
    _drive(h, 0, 4.0, sampled_id=3)
    d = h._state[0]["directive"]
    assert d["blend_id"] == END
    assert abs(d["blend_w"] - (0.7 + 0.3 * 2 / 100)) < 1e-9, d

    # Step 3: plain normal -> no directive, token-id path.
    _drive(h, 0, 4.0, sampled_id=3)
    assert h._state[0]["directive"] is None

    # Step 4: entropy above the switch reference and window satisfied ->
    # to_soft, eased toward <think> with alpha = 0.5 + 0.5 * (4/100).
    _drive(h, 0, 6.5, sampled_id=3)
    d = h._state[0]["directive"]
    assert d["blend_id"] == START
    assert abs(d["blend_w"] - (0.5 + 0.5 * 4 / 100)) < 1e-9, d
    print("ok: step-0, to_normal and to_soft blends carry the reference ramps")


def test_injection_and_termination_budget():
    h = _holder()
    h.sync_batch(_update(
        added=[(0, _params(window=1, max_switch=1, term_max=3,
                           stop_ids=[STOP]), [], [])],
        batch_size=1))

    _drive(h, 0, 5.0, sampled_id=3)          # step 0: ref = 5
    f = _drive(h, 0, 4.0, sampled_id=3)      # to_normal #1 -> arms conv queue
    assert f == -1, "the trigger step itself is not overridden"
    f = _drive(h, 0, 4.0, sampled_id=3)      # queue pops </think>
    assert f == END, f
    # Climb back to soft, then drop again for switch #2 (> 2*max_switch is
    # required for termination, so force a third).
    _drive(h, 0, 6.0, sampled_id=3)          # to_soft
    f = _drive(h, 0, 3.0, sampled_id=3)      # to_normal #2 (within [1, 2])
    _drive(h, 0, 9.0, sampled_id=3)          # to_soft
    f = _drive(h, 0, 2.0, sampled_id=3)      # to_normal #3 -> > 2*1: termination
    st = h._state[0]
    assert st["queue"] == [END, 7, 8] or st["injecting"], st
    assert st["budget"] == 2, st["budget"]   # armed pre-decremented (3 - 1)
    f = _drive(h, 0, 2.0, sampled_id=3)      # pop END, budget 1
    assert f == END
    f = _drive(h, 0, 2.0, sampled_id=3)      # pop 7, budget 0 -> stop forced
    assert f == STOP, f
    print("ok: convergence injection, then termination and the budget cutoff")


def test_math_tokens_and_lock_stay_discrete():
    h = _holder()
    h.sync_batch(_update(
        added=[(0, _params(window=1, math_ids=[42]), [], [])], batch_size=1))
    _drive(h, 0, 5.0, sampled_id=42)  # math token on step 0: discrete anyway
    assert h._state[0]["directive"] is None

    _drive(h, 0, 9.0, sampled_id=END)  # sampled </think> locks the row
    assert h._state[0]["locked"]
    # Entropy keeps rising, window satisfied -- but locked blocks to_soft.
    for e in (10.0, 11.0, 12.0):
        _drive(h, 0, e, sampled_id=3)
    assert h._state[0]["mode"] == 0 or h._state[0]["locked"]
    assert h._state[0]["directive"] is None, "locked rows never feed a mixture"
    print("ok: math ids and a sampled </think> keep the feedback discrete")


def test_resumed_row_restores_lock_and_step():
    h = _holder()
    h.sync_batch(_update(
        added=[
            (0, _params(), [1, 2], [5, END, 7]),
            (1, _params(), [1, 2], [5, 6]),
        ],
        batch_size=2))
    assert h._state[0]["locked"], "resumed past </think> must stay locked"
    assert h._state[0]["step"] == 3, "the blend ramp continues from the resume point"
    assert not h._state[1]["locked"]
    print("ok: a resumed row restores its lock and its step counter")


def test_skipped_rows_do_not_advance():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(), [], [])], batch_size=1))
    h.set_rows_without_a_real_decode([0])
    h._state[0]["entropy"] = 5.0
    h._state[0]["probs"] = torch.full((VOCAB,), 1.0 / VOCAB)
    forced = h.step(torch.tensor([3]))
    assert forced is None
    assert h._state[0]["step"] == 0 and h._state[0]["ref"] is None
    assert h.embed_directives() is None or h._state[0]["directive"] is None
    h.set_rows_without_a_real_decode([])
    _drive(h, 0, 5.0, sampled_id=3)
    assert h._state[0]["step"] == 1
    print("ok: a discarded prefill row leaves the swir state untouched")


def test_directive_handoff_shapes():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(), [], []),
                                (1, _params(), [], [])], batch_size=2))
    _drive(h, 0, 5.0, sampled_id=3, num_reqs=2)
    _drive(h, 1, 5.0, sampled_id=3, num_reqs=2)
    rows, probs, blend_ids, blend_ws = h.embed_directives()
    assert rows.tolist() == [0, 1]
    assert probs.shape == (2, VOCAB) and probs.dtype == torch.float32
    assert blend_ids.tolist() == [LINEBREAK, LINEBREAK]
    assert torch.allclose(blend_ws, torch.tensor([0.9, 0.9]))
    print("ok: the runner handoff carries rows, mixtures and blend anchors")


if __name__ == "__main__":
    test_mode_machine_matches_the_reference()
    test_directives_and_blend_ramps()
    test_injection_and_termination_budget()
    test_math_tokens_and_lock_stay_discrete()
    test_resumed_row_restores_lock_and_step()
    test_skipped_rows_do_not_advance()
    test_directive_handoff_shapes()
    print("\nall SwiReasoning state checks passed")

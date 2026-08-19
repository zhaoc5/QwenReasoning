"""State machine of the vLLM Soft Thinking holder: tracking, Cold Stop, hand-off.

Covers what the parity test cannot: which rows the holder tracks as the batch
changes, when Cold Stop forces `</think>`, and which mixtures it hands the model
runner to feed back on the next step.

Run directly: `python tests/test_soft_thinking_vllm_state.py`. CPU only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

# Inside the toy vocabulary below, since _peaked indexes columns by it.
EOT = 63
VOCAB = 64


def _holder(max_num_seqs=8):
    from types import SimpleNamespace

    from vllm.v1.sample.soft_thinking_state import SoftThinkingStateHolder

    cfg = SimpleNamespace(reasoning_end_token_ids=[EOT])
    return SoftThinkingStateHolder(cfg, max_num_seqs, torch.device("cpu"))


def _update(added=(), removed=(), moved=(), batch_size=0):
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

    return BatchUpdate(
        batch_size=batch_size, removed=list(removed), added=list(added), moved=list(moved)
    )


def _params(soft=True, topk=4, tau=0.01, patience=2, stop_ids=None):
    from vllm import SamplingParams

    return SamplingParams(
        soft_thinking=soft,
        soft_topk=topk,
        soft_entropy_threshold=tau,
        soft_patience=patience,
        stop_token_ids=stop_ids,
        max_tokens=8,
    )


def _peaked(n, peak_row_ids):
    """Logits with a near-deterministic peak per row: entropy well under tau."""
    logits = torch.full((n, VOCAB), -30.0)
    for r, tok in enumerate(peak_row_ids):
        logits[r, tok] = 30.0
    return logits


def _flat(n):
    """Uniform logits: entropy is log(VOCAB), far above any sane tau."""
    return torch.zeros(n, VOCAB)


def test_only_soft_requests_are_tracked():
    h = _holder()
    h.sync_batch(
        _update(
            added=[
                (0, _params(soft=True), [], []),
                (1, _params(soft=False), [], []),
            ],
            batch_size=2,
        )
    )
    assert h.has_tracked_requests()
    assert h._tracked_rows() == [0], h._tracked_rows()

    h.sync_batch(_update(removed=[0], batch_size=1))
    assert not h.has_tracked_requests(), "removing the only soft row must clear it"
    print("ok: tracks soft_thinking rows only, and drops them on removal")


def test_rows_follow_swaps_and_moves():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(), [], [])], batch_size=1))
    h._state[0]["low_entropy_steps"] = 7

    from vllm.v1.sample.logits_processor.interface import MoveDirectionality

    h.sync_batch(_update(moved=[(0, 3, MoveDirectionality.UNIDIRECTIONAL)], batch_size=4))
    assert h._tracked_rows() == [3], h._tracked_rows()
    assert h._state[3]["low_entropy_steps"] == 7, "the counter must move with the row"
    print("ok: state follows a row when the batch is compacted")


def test_cold_stop_fires_after_patience_and_forces_think_end():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(patience=2), [], [])], batch_size=1))
    temps = torch.ones(1)

    # Step 1: low entropy, counter 1 of 2. The trace records the argmax, since
    # the thinking phase commits to no token.
    forced = h.step(_peaked(1, [5]), temps, None, None, None)
    assert forced is not None and forced[0].item() == 5, forced
    assert h._state[0]["low_entropy_steps"] == 1
    assert h._state[0]["in_thinking"]

    # Step 2: still low, counter reaches patience -- fire.
    forced = h.step(_peaked(1, [5]), temps, None, None, None)
    assert forced[0].item() == EOT, f"Cold Stop must force </think>, got {forced}"
    assert not h._state[0]["in_thinking"], "the row must leave the thinking phase"
    assert h.concept_tokens_for_rows() is None, "a finished row feeds back no mixture"
    print("ok: Cold Stop fires on the patience-th consecutive low-entropy step")


def test_a_high_entropy_step_resets_the_run():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(patience=2), [], [])], batch_size=1))
    temps = torch.ones(1)

    h.step(_peaked(1, [5]), temps, None, None, None)
    assert h._state[0]["low_entropy_steps"] == 1
    h.step(_flat(1), temps, None, None, None)                   # interruption
    assert h._state[0]["low_entropy_steps"] == 0, "the run must reset"
    forced = h.step(_peaked(1, [5]), temps, None, None, None)   # count restarts
    assert forced[0].item() != EOT, "one low step after a reset must not fire"
    assert h._state[0]["low_entropy_steps"] == 1
    assert h._state[0]["in_thinking"]
    print("ok: a single high-entropy step resets the counter")


def test_natural_think_end_beats_cold_stop():
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(patience=2), [], [])], batch_size=1))
    # A confident </think>: low entropy, but the argmax is the end token itself.
    # The forced id is EOT either way here, so the discriminator is the counter:
    # Cold Stop never reached its patience.
    forced = h.step(_peaked(1, [EOT]), torch.ones(1), None, None, None)
    assert forced[0].item() == EOT
    assert h._state[0]["low_entropy_steps"] == 0, "Cold Stop must not have counted"
    assert not h._state[0]["in_thinking"]
    print("ok: a real </think> ends thinking without Cold Stop firing")


def test_handoff_carries_only_still_thinking_rows():
    h = _holder()
    h.sync_batch(
        _update(
            added=[
                (0, _params(patience=99), [], []),   # keeps thinking
                (1, _params(patience=99), [], []),   # emits </think>
            ],
            batch_size=2,
        )
    )
    logits = _flat(2)
    logits[1] = _peaked(1, [EOT])[0]
    h.step(logits, torch.ones(2), None, None, None)

    handoff = h.concept_tokens_for_rows()
    assert handoff is not None
    rows, ids, probs = handoff
    assert rows.tolist() == [0], f"only the thinking row hands back a mixture: {rows}"
    assert ids.shape[0] == 1 and probs.shape[0] == 1
    assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)
    print("ok: only rows still thinking hand a mixture to the model runner")


def test_a_mixture_survives_a_batch_move():
    """The concept token must follow its row when the batch is compacted.

    It is produced on one step and fed back on the next, and vLLM may reorder
    rows in between. Holding it in a tensor keyed by the producing step's row
    indices silently feeds the wrong mixture to the wrong sequence.
    """
    from vllm.v1.sample.logits_processor.interface import MoveDirectionality

    h = _holder()
    h.sync_batch(
        _update(
            added=[(0, _params(patience=99), [], []), (1, _params(patience=99), [], [])],
            batch_size=2,
        )
    )
    h.step(_flat(2), torch.ones(2), None, None, None)
    rows, ids, _ = h.concept_tokens_for_rows()
    assert rows.tolist() == [0, 1]
    mixture_of_row0 = ids[0].clone()

    h.sync_batch(_update(removed=[1], batch_size=1))
    h.sync_batch(_update(moved=[(0, 5, MoveDirectionality.UNIDIRECTIONAL)], batch_size=6))

    rows, ids, _ = h.concept_tokens_for_rows()
    assert rows.tolist() == [5], f"the row moved to 5, got {rows.tolist()}"
    assert torch.equal(ids[0], mixture_of_row0), "the mixture must move with its row"
    print("ok: a concept token follows its row across a batch move")


def test_a_stop_token_never_ends_a_thinking_row():
    """vLLM stops on a stop id, so one must not reach the trace mid-thinking.

    The reference implementation ignores stop tokens while a row is still
    consuming continuous inputs -- honouring one would cut the sample off before
    it ever produced an answer. Here the most likely token *is* a stop token, so
    the recorded id has to be the next candidate instead.
    """
    STOP = 40
    h = _holder()
    h.sync_batch(
        _update(added=[(0, _params(patience=99, stop_ids=[STOP]), [], [])],
                batch_size=1)
    )
    # STOP is the peak; 41 is a clear second.
    logits = torch.full((1, VOCAB), -30.0)
    logits[0, STOP] = 30.0
    logits[0, 41] = 20.0

    forced = h.step(logits, torch.ones(1), None, None, None)
    assert forced[0].item() != STOP, "a stop id must not be recorded while thinking"
    assert forced[0].item() == 41, f"expected the next candidate, got {forced[0].item()}"
    assert h._state[0]["in_thinking"], "the row must still be thinking"

    # The mixture is untouched: the stop token keeps its weight in the concept.
    rows, ids, probs = h.concept_tokens_for_rows()
    assert ids[0][0].item() == STOP, "the mixture still leads with the true argmax"
    print("ok: a stop token is kept out of the trace but stays in the mixture")



def test_a_partial_prefill_row_does_not_advance_the_state():
    """A row whose sampled token the runner will discard must be left alone.

    During a chunked (or resumed) prefill the sampler still sees the row, but
    its logits belong to a mid-prompt position. Without the skip, a peaked
    distribution there would advance Cold Stop, and an argmax that happened to
    be `</think>` would end the thinking block before generation ever started.
    """
    h = _holder()
    h.sync_batch(_update(added=[(0, _params(patience=1, tau=0.5), [], [])],
                         batch_size=1))

    # The runner marks row 0 as having no real decode this step. Even logits
    # peaked on `</think>` itself must change nothing.
    h.set_rows_without_a_real_decode([0])
    forced = h.step(_peaked(1, [EOT]), torch.ones(1), None, None, None)
    assert forced is None, "a skipped row must not produce a forced token"
    assert h._state[0]["in_thinking"], "a mid-prompt </think> argmax must not count"
    assert h._state[0]["low_entropy_steps"] == 0, "Cold Stop must not advance"
    assert h.concept_tokens_for_rows() is None, "no mixture from a prompt position"

    # Next step the prefill has finished; the same logits now do count.
    h.set_rows_without_a_real_decode([])
    forced = h.step(_peaked(1, [EOT]), torch.ones(1), None, None, None)
    assert forced[0].item() == EOT
    assert not h._state[0]["in_thinking"]
    print("ok: a discarded prefill row leaves the thinking state untouched")


def test_a_resumed_row_that_closed_its_thinking_stays_discrete():
    """Preemption re-adds a request with its outputs so far; `</think>` in them
    means the row is in its answer phase and must not re-enter thinking."""
    h = _holder()
    h.sync_batch(_update(
        added=[
            (0, _params(), [1, 2], [5, EOT, 7]),   # already answered past </think>
            (1, _params(), [1, 2], [5, 6, 7]),     # resumed mid-thinking
        ],
        batch_size=2,
    ))
    assert not h._state[0]["in_thinking"], "</think> in the outputs ends thinking"
    assert h._state[1]["in_thinking"], "no </think> yet: still thinking"
    assert h._tracked_rows() == [1], h._tracked_rows()
    print("ok: a resumed row that already closed its thinking block stays discrete")



def test_stop_strings_are_refused_with_soft_thinking():
    """Stop token ids are kept out of the thinking trace; stop *strings* are
    matched by the detokenizer against it and would cut the request mid-thought,
    so the combination is refused at validation."""
    from vllm import SamplingParams

    try:
        SamplingParams(soft_thinking=True, stop=["\n\n"], max_tokens=8)
    except Exception as e:
        assert "stop" in str(e), e
    else:
        raise AssertionError("soft_thinking + stop strings must be refused")
    # Token ids stay allowed -- the holder keeps them out of the trace itself.
    SamplingParams(soft_thinking=True, stop_token_ids=[7], max_tokens=8)
    print("ok: stop strings are refused, stop token ids are not")


if __name__ == "__main__":
    test_only_soft_requests_are_tracked()
    test_rows_follow_swaps_and_moves()
    test_cold_stop_fires_after_patience_and_forces_think_end()
    test_a_high_entropy_step_resets_the_run()
    test_natural_think_end_beats_cold_stop()
    test_handoff_carries_only_still_thinking_rows()
    test_a_mixture_survives_a_batch_move()
    test_a_stop_token_never_ends_a_thinking_row()
    test_a_partial_prefill_row_does_not_advance_the_state()
    test_a_resumed_row_that_closed_its_thinking_stays_discrete()
    test_stop_strings_are_refused_with_soft_thinking()
    print("\nall Soft Thinking state checks passed")

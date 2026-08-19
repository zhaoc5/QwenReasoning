"""Checks for the presence penalty added to the generation loops.

Run with `python tests/test_presence_penalty.py` inside the project venv; it needs torch
but no GPU and no model weights.

The interesting case is not the arithmetic, it is the bookkeeping: the loops drop finished
sequences out of the running batch mid-generation, and the `[batch, vocab]` mask of
already-emitted tokens has to be reindexed along with every other per-row tensor. If it is
not, a surviving row inherits some other row's history.
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.generation_utils import apply_presence_penalty, generate_cot

VOCAB = 6
STOP = 5


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = STOP


class StubModel(torch.nn.Module):
    """Emits a fixed logit row per *original* batch position.

    The original position is smuggled through the KV cache as a plain tensor, so it is
    shrunk by `batch_select_hybrid_cache` exactly like a real cache would be. That makes
    the stub's own row identity track the same reindexing the penalty mask must follow.
    """

    def __init__(self, row_logits):
        super().__init__()
        self.row_logits = row_logits
        self.embed = torch.nn.Embedding(VOCAB, 4)
        self.config = SimpleNamespace(vocab_size=VOCAB, eos_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=None)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=True, **kwargs):
        if past_key_values is None:
            row_ids = torch.arange(input_ids.shape[0])
            seq_len = input_ids.shape[1]
        else:
            row_ids = past_key_values[0].long()
            seq_len = 1
        logits = self.row_logits[row_ids].unsqueeze(1).expand(-1, seq_len, -1).contiguous()
        return SimpleNamespace(logits=logits, past_key_values=(row_ids.float(),))


def _run(row_logits, prompt, presence_penalty, max_new_tokens):
    model = StubModel(row_logits)
    input_ids = torch.tensor(prompt, dtype=torch.long)
    return generate_cot(
        model, StubTokenizer(),
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
        do_sample=False, max_new_tokens=max_new_tokens,
        stop_token_ids=[STOP], presence_penalty=presence_penalty,
    )


def test_helper_is_binary_and_subtractive():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    seen = torch.tensor([[True, False, True]])
    out = apply_presence_penalty(logits, seen, 1.5)
    assert torch.allclose(out, torch.tensor([[-1.5, 1.0, 0.5]])), out
    # A token seen many times is penalised no more than one seen once -- the mask is
    # boolean, so there is nothing to accumulate. Counting is frequency_penalty's job.
    assert torch.allclose(apply_presence_penalty(logits, seen, 0.0), logits)
    print("ok: helper is binary and subtractive")


def test_penalty_walks_down_the_ranking():
    # One row, logits strictly ordered 1 > 2 > 3 > 4 with unit gaps. Greedy without a
    # penalty repeats token 1 forever; with a penalty larger than the whole spread of the
    # row, each emitted token drops below every unseen one, so the argmax walks the list.
    # (A penalty merely larger than one gap is not enough: at 2.0 the third step ties
    # token 1 at 3-2 against an untouched token 3 at 1, and argmax takes the lower index.)
    row = torch.tensor([[-10.0, 3.0, 2.0, 1.0, 0.0, -10.0]])
    plain = _run(row, [[0]], presence_penalty=0.0, max_new_tokens=4)
    assert plain[0, 1:].tolist() == [1, 1, 1, 1], plain
    penalised = _run(row, [[0]], presence_penalty=10.0, max_new_tokens=4)
    assert penalised[0, 1:].tolist() == [1, 2, 3, 4], penalised
    print("ok: penalty walks the argmax down the ranking")


def test_mask_survives_a_dropped_sequence():
    # Row 0 stops immediately and is dropped; row 1 must keep its *own* penalty history
    # as it slides from index 1 to index 0. If the mask is not reindexed with the batch,
    # row 1 inherits row 0's empty history and repeats token 1 instead of walking down.
    rows = torch.tensor([
        [-10.0, 0.0, 0.0, 0.0, 0.0, 10.0],   # row 0: emits STOP at once
        [-10.0, 3.0, 2.0, 1.0, 0.0, -10.0],  # row 1: the walking row
    ])
    out = _run(rows, [[0], [0]], presence_penalty=10.0, max_new_tokens=4)
    assert out[0, 1] == STOP, out
    assert out[1, 1:].tolist() == [1, 2, 3, 4], out[1]
    print("ok: mask is reindexed when a sequence is dropped")


def test_zero_penalty_is_a_no_op():
    # The default must reproduce the old behaviour exactly, mask never even allocated.
    rows = torch.tensor([[-10.0, 3.0, 2.0, 1.0, 0.0, -10.0]])
    assert _run(rows, [[0]], 0.0, 4).tolist() == _run(rows, [[0]], 0, 4).tolist()
    print("ok: zero penalty is a no-op")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_helper_is_binary_and_subtractive()
    test_penalty_walks_down_the_ranking()
    test_mask_survives_a_dropped_sequence()
    test_zero_penalty_is_a_no_op()
    print("\nall presence-penalty checks passed")

"""Checks for generate_soft (Soft Thinking, arXiv:2505.15778).

Run with `python tests/test_soft_thinking.py` inside the project venv; needs torch but no
GPU and no model weights.

The parts worth pinning down are the state machine, not the mixture arithmetic:
  - Cold Stop counts *consecutive* low-entropy steps and resets on any step above tau
  - a row leaving the thinking phase emits `</think>` itself, so the model sees the
    boundary before it answers
  - only the answer phase may terminate on a stop token
  - the embedding fed back is the concept mixture while thinking, never a single token
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.generation_utils import generate_soft

VOCAB = 8
END_THINK = 6
STOP = 7
HIDDEN = 4


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = STOP

    def convert_tokens_to_ids(self, tok):
        return END_THINK if tok == "</think>" else None


class StubModel(torch.nn.Module):
    """Replays a scripted logit row per step, and records what it was fed each step."""

    def __init__(self, script):
        super().__init__()
        self.script = script            # list of [vocab] logit rows, one per step
        # Default random init is fine: the mixture assertions compare against rows of
        # this same matrix, they do not depend on its values.
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.config = SimpleNamespace(vocab_size=VOCAB, eos_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.seen_inputs = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                inputs_embeds=None, use_cache=True, **kwargs):
        self.seen_inputs.append(inputs_embeds.squeeze(1).clone() if inputs_embeds is not None else None)
        step = len(self.seen_inputs) - 1
        row = self.script[min(step, len(self.script) - 1)]
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        logits = row.unsqueeze(0).expand(batch, -1).unsqueeze(1).contiguous()
        return SimpleNamespace(logits=logits, past_key_values=(torch.zeros(batch),))


def peaked(idx, sharp=30.0):
    """A near-deterministic row: entropy over the renormalized top-n is ~0."""
    v = torch.full((VOCAB,), -sharp)
    v[idx] = sharp
    return v


def flat():
    """A high-entropy row whose argmax is still unambiguous.

    A perfectly uniform row will not do: torch.topk breaks ties arbitrarily, so the
    argmax can land on END_THINK and end the thinking phase for reasons that have
    nothing to do with the Cold Stop counter under test. Nudging one non-special token
    pins the argmax while leaving entropy near its maximum (~2.06 of log(8)=2.08).
    """
    v = torch.zeros(VOCAB)
    v[2] = 0.5
    return v


def run(script, **kw):
    model = StubModel(script)
    ids = torch.tensor([[1]])
    opts = dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, do_sample=False,
                stop_token_ids=[STOP], soft_topk=4, soft_entropy_threshold=0.01,
                soft_patience=3, end_thinking_token_id=END_THINK)
    opts.update(kw)
    out = generate_soft(model, StubTokenizer(), input_ids=ids,
                        attention_mask=torch.ones_like(ids), **opts)
    return out[0, 1:].tolist(), model


def test_cold_stop_fires_after_consecutive_low_entropy():
    # Five confident steps with patience=3: the counter hits 3 on step 3, which forces
    # </think>; steps 4 and 5 are then the discrete answer phase.
    toks, _ = run([peaked(2)] * 5, max_new_tokens=5)
    assert toks[:3] == [2, 2, END_THINK], toks
    print("ok: Cold Stop fires on the 3rd consecutive low-entropy step")


def test_counter_resets_on_a_high_entropy_step():
    # low, low, HIGH, low, low -> the run of 3 is broken, so no Cold Stop within 5 steps.
    script = [peaked(2), peaked(2), flat(), peaked(2), peaked(2)]
    toks, _ = run(script, max_new_tokens=5)
    assert END_THINK not in toks, f"counter should have reset, got {toks}"
    print("ok: a single high-entropy step resets the Cold Stop counter")


def test_natural_end_of_thinking_emits_end_token():
    # </think> becomes the argmax on step 2: the row must emit it and switch to answering.
    script = [peaked(2), peaked(END_THINK), peaked(3), peaked(STOP)]
    toks, _ = run(script, max_new_tokens=6, soft_patience=999)
    assert toks[0] == 2 and toks[1] == END_THINK, toks
    assert toks[2] == 3 and toks[3] == STOP, toks
    print("ok: </think> as argmax ends thinking and is itself emitted")


def test_stop_token_ignored_while_thinking():
    # A stop token that is merely the argmax of a concept token must not end the run --
    # the row is still consuming continuous inputs and has not answered yet.
    toks, _ = run([peaked(STOP)] * 4, max_new_tokens=4, soft_patience=999)
    assert len(toks) == 4, f"thinking phase should not stop on {STOP}: {toks}"
    print("ok: stop tokens do not terminate the thinking phase")


def test_concept_token_is_a_mixture_not_an_embedding_row():
    # With two tokens at equal probability the fed-back vector must be their average,
    # which is not equal to either row of E.
    row = torch.full((VOCAB,), -30.0)
    row[2] = row[3] = 5.0
    _, model = run([row] * 3, max_new_tokens=3, soft_patience=999)
    fed = model.seen_inputs[1]            # what step 2 was fed
    E = model.get_input_embeddings().weight
    assert torch.allclose(fed[0], (E[2] + E[3]) / 2, atol=1e-4), fed
    assert not torch.allclose(fed[0], E[2], atol=1e-3)
    print("ok: the fed-back vector is the probability-weighted mixture")


def test_cold_stop_entropy_ignores_top_p():
    """Cold Stop must read the model's distribution, not the nucleus filter's output.

    This is the case the other tests miss: they leave top_p at 1.0, so the filter never
    bites. With top_p on, a confident-looking row whose top token clears the nucleus
    threshold collapses to a single surviving token, and the *renormalized* entropy is
    exactly 0 no matter how much mass the model actually spread elsewhere. Reading the
    entropy there makes Cold Stop fire on every such step -- it reports on top_p rather
    than on the model.

    Here the top token holds ~0.72 and the rest is spread over the remaining vocabulary,
    so the full distribution carries real entropy (~1.0 nats) while top_p=0.7 truncates
    to one token. Cold Stop must not fire.
    """
    row = torch.tensor([-10.0, 1.4, 0.0, 0.0, 0.0, -10.0])
    probs = torch.softmax(row, dim=-1)
    ent_full = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    assert ent_full > 0.5, f"fixture must have real entropy, got {ent_full}"

    toks, _ = run([row] * 6, max_new_tokens=6, soft_patience=3,
                  top_p=0.7, soft_entropy_threshold=0.01)
    assert END_THINK not in toks, (
        f"Cold Stop fired on a genuinely uncertain step -- entropy is being read after "
        f"top_p truncation (full-vocab entropy was {ent_full:.3f} nats): {toks}")
    print("ok: Cold Stop entropy is taken before top_p truncation")


def test_soft_topk_limits_the_mixture():
    # Four tokens carry mass but soft_topk=2 keeps only the top two, renormalized.
    row = torch.full((VOCAB,), -30.0)
    row[2], row[3], row[4], row[5] = 5.0, 5.0, 4.0, 4.0
    _, model = run([row] * 3, max_new_tokens=3, soft_topk=2, soft_patience=999)
    E = model.get_input_embeddings().weight
    assert torch.allclose(model.seen_inputs[1][0], (E[2] + E[3]) / 2, atol=1e-4)
    print("ok: soft_topk truncates and renormalizes the mixture")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_cold_stop_fires_after_consecutive_low_entropy()
    test_counter_resets_on_a_high_entropy_step()
    test_natural_end_of_thinking_emits_end_token()
    test_stop_token_ignored_while_thinking()
    test_concept_token_is_a_mixture_not_an_embedding_row()
    test_cold_stop_entropy_ignores_top_p()
    test_soft_topk_limits_the_mixture()
    print("\nall Soft Thinking checks passed")

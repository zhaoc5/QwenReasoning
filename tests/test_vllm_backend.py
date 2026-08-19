"""Checks for the vLLM baseline backend and its log naming.

Run directly: `python tests/test_vllm_backend.py`. Constructing SamplingParams needs vLLM
importable but not a GPU; the naming checks need neither, so they run either way.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.log_naming import log_stem
from src.vllm_backend import SUPPORTED_METHODS, check_method_supported


def test_unsupported_methods_are_rejected():
    # soft is supported now: the fork implements Soft Thinking in the engine.
    # selar and swir rewrite the running sequence, which it still cannot express.
    for method in ("selar", "swir"):
        try:
            check_method_supported(method)
        except ValueError as e:
            assert method in str(e), e
            assert "--backend hf" in str(e), "the error should say what to do instead"
        else:
            raise AssertionError(f"{method} should not be accepted by the vLLM backend")
    for method in SUPPORTED_METHODS:
        check_method_supported(method)
    assert "soft" in SUPPORTED_METHODS
    print("ok: soft is accepted, selar and swir are refused with a way out")


def test_hf_stems_are_unchanged():
    """The default backend must not move any existing log filename."""
    # Exactly the name written by the pre-backend code, byte for byte.
    assert log_stem("Qwen/Qwen3.5-4B", "aime_2024", "cot", 81920, 0,
                    temperature=1.0, presence_penalty=1.5) == \
        "Qwen3.5-4B_aime_2024_cot_81920_temp1.0_pp1.5_seed0"
    assert log_stem("Qwen/Qwen3.5-4B", "aime_2024", "cot_greedy", 81920, 41,
                    temperature=1.0, presence_penalty=1.5) == \
        "Qwen3.5-4B_aime_2024_cot_greedy_81920_pp1.5_seed41"
    # And passing the default explicitly is the same as omitting it.
    assert log_stem("Qwen/Qwen3.5-4B", "aime_2024", "cot", 81920, 0,
                    temperature=1.0, presence_penalty=1.5, backend="hf") == \
        log_stem("Qwen/Qwen3.5-4B", "aime_2024", "cot", 81920, 0,
                 temperature=1.0, presence_penalty=1.5)
    print("ok: hf log names are byte-identical to the ones already in logs/")


def test_vllm_stems_do_not_collide_with_hf():
    hf = log_stem("Qwen/Qwen3.5-4B", "aime_2024", "cot", 81920, 0,
                  temperature=1.0, presence_penalty=1.5, backend="hf")
    vllm = log_stem("Qwen/Qwen3.5-4B", "aime_2024", "cot", 81920, 0,
                    temperature=1.0, presence_penalty=1.5, backend="vllm")
    assert hf != vllm, "a vLLM run must not overwrite the HF run it is checked against"
    assert vllm.endswith("_seed0"), "the seed stays last so merge.py's glob still works"
    assert "_vllm_" in vllm, vllm
    print("ok: vllm logs get their own name and keep the seed suffix")


def test_stop_ids_survive_a_composite_config():
    """The vLLM path must find every terminator the HF path finds.

    Qwen3.5 is the awkward case and the reason this check exists: it ships no
    generation_config.json, and its top-level config is a composite whose own
    eos_token_id is None -- the real one sits in .text_config. Reading the top level
    alone silently yields just the tokenizer's <|im_end|>, so sequences that ended on
    <|endoftext|> would have kept decoding to max_new_tokens on the vLLM side only.
    """
    from transformers import AutoTokenizer

    from src.generation_utils import resolve_stop_token_ids_without_model

    model_name = "Qwen/Qwen3.5-0.8B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ids = resolve_stop_token_ids_without_model(model_name, tokenizer)
    except Exception as e:
        print(f"skipped: {model_name} config not reachable ({type(e).__name__})")
        return
    # 248046 is the tokenizer's <|im_end|>; 248044 is <|endoftext|>, reachable only
    # through the text sub-config.
    assert ids == [248044, 248046], (
        f"expected both Qwen3.5 terminators, got {ids} "
        f"({[tokenizer.decode([i]) for i in ids]})"
    )
    print("ok: the composite Qwen3.5 config still yields both terminators")


def _sampling_params(**overrides):
    from src.vllm_backend import build_sampling_params

    kwargs = dict(greedy=False, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                  presence_penalty=1.5, max_new_tokens=81920, seed=3,
                  stop_token_ids=[248044, 248046])
    kwargs.update(overrides)
    return build_sampling_params(**kwargs)


def test_sampling_params_pass_the_recipe_through():
    sp = _sampling_params()
    assert (sp.temperature, sp.top_p, sp.top_k, sp.min_p) == (1.0, 0.95, 20, 0.0)
    assert sp.presence_penalty == 1.5
    assert sp.max_tokens == 81920
    assert sp.seed == 3, "seeds are the whole point of the four cot runs"
    assert sp.n == 1, "one sample per prompt; repeats come from separate seeds"
    assert set(sp.stop_token_ids) == {248044, 248046}
    print("ok: the Qwen3.5 recipe reaches vLLM unchanged")


def test_greedy_neutralises_filters_but_keeps_the_penalty():
    sp = _sampling_params(greedy=True)
    assert sp.temperature == 0.0, "vLLM spells greedy as temperature 0"
    # These three are argmax-invariant, so zeroing them changes no token -- it only stops
    # the recorded params from claiming a nucleus was applied when none was.
    assert (sp.top_p, sp.top_k, sp.min_p) == (1.0, 0, 0.0)
    assert sp.seed is None, "greedy has nothing to seed"
    # The penalty is the one knob here that reorders logits, so it must survive.
    assert sp.presence_penalty == 1.5
    assert set(sp.stop_token_ids) == {248044, 248046}
    print("ok: greedy drops the filters and keeps the presence penalty")


if __name__ == "__main__":
    test_unsupported_methods_are_rejected()
    test_hf_stems_are_unchanged()
    test_vllm_stems_do_not_collide_with_hf()
    test_stop_ids_survive_a_composite_config()
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("skipped: SamplingParams checks need vLLM installed")
    else:
        test_sampling_params_pass_the_recipe_through()
        test_greedy_neutralises_filters_but_keeps_the_penalty()
    print("\nall vLLM backend checks passed")

"""vLLM backend for all five methods: `cot`, `cot_greedy`, `soft`, `swir`, `selar`.

`soft`, `swir` and `selar` run on the fork's engine-side implementations; the
sampling-equivalence argument below covers `cot` and `cot_greedy`, and each
method module in the fork documents its own parity against the HF reference.

`soft` runs on the fork's own Soft Thinking support (`vllm/v1/sample/soft_thinking.py`),
which needs three engine flags -- a reasoning parser, prompt embeds, and prefix caching
off -- all set by `build_llm` below. See that module for why each is required.

The point of a second backend is a faster baseline, not a different one, so the sampling
has to mean the same thing on both sides. It does:

  * Presence penalty is subtracted from the *output* tokens only, never the prompt
    (`logits -= presence_penalties * output_mask` in
    vllm/model_executor/layers/utils.py). That is the same scope as `generate_cot`'s
    `seen_tokens`, which is allocated empty at the first decode step and only ever marks
    tokens the model itself produced.
  * Penalties land before temperature, and temperature before top_k/top_p/min_p -- the
    same order `generate_cot` applies them in.
  * vLLM's min_p is relative to the peak (`min_p * max_prob`), matching
    `apply_sampling_filter`. An absolute floor would be a different filter entirely.
  * Greedy: vLLM takes the argmax straight after the penalties and skips temperature and
    the filters, whereas `generate_cot` applies them and then takes the argmax. Same
    token either way -- dividing by a positive constant is monotonic, and top_k/top_p/min_p
    only ever mask tokens strictly below the maximum, so none of them can move it. This is
    also why the penalty still has to be applied for `cot_greedy`: it is the one knob here
    that *does* reorder logits.

What is deliberately not shared is the batching. The HF path decodes a fixed batch in
lockstep and pays for the longest sequence in it; vLLM schedules continuously, so
`--batch_size` has no meaning here and is ignored.
"""

SUPPORTED_METHODS = ("cot", "cot_greedy", "soft", "swir", "selar")


def check_method_supported(method):
    """Fail loudly on the methods this backend cannot express."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"--backend vllm supports {', '.join(SUPPORTED_METHODS)}, not "
            f"'{method}'. Unknown methods run with --backend hf if the HF "
            f"decode loops implement them."
        )


def build_llm(model_name, *, tensor_parallel_size=1, gpu_memory_utilization=0.90,
              max_model_len=None, seed=None, enforce_eager=False, max_num_seqs=None,
              soft_thinking=False, swir=False, selar=False, reasoning_parser="qwen3"):
    """Construct the engine. Kept separate so the import cost lands only on this path."""
    if soft_thinking or swir or selar:
        # soft, swir and selar are implemented in vLLM's V1 GPU model runner.
        # Dense non-MoE models (all of Qwen3) default to the V2 runner, which
        # would silently decode plain CoT -- the engine refuses these requests
        # under V2, and this pin is how the refusal never fires from run.py.
        # Hybrid models (Qwen3.5) select V1 regardless, so for them this is a
        # no-op. Must be set before the engine builds: workers inherit it.
        import os
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM

    kwargs = {
        "model": model_name,
        "dtype": "auto",
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    if soft_thinking or swir or selar:
        # All refused by the engine if missing, so set them here rather than
        # make every caller remember: the latent input is fed back through
        # inputs_embeds, and the prefix cache is keyed by token ids that a
        # latent step's KV did not come from. The reasoning parser resolves
        # <think>/</think>; selar has no thinking block and does not need it.
        kwargs["enable_prompt_embeds"] = True
        kwargs["enable_prefix_caching"] = False
        if soft_thinking or swir:
            kwargs["reasoning_parser"] = reasoning_parser
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if seed is not None:
        kwargs["seed"] = seed
    if max_num_seqs is not None:
        # Qwen3.5 is hybrid: every concurrent decode also holds one Mamba cache block, and
        # vLLM refuses to start when max_num_seqs exceeds the blocks it could fit rather
        # than quietly running fewer sequences. Lowering this is the fix it asks for.
        kwargs["max_num_seqs"] = max_num_seqs
    return LLM(**kwargs)


def swir_sampling_kwargs(tokenizer, model_name, *, alpha, max_switch_count):
    """The tokenizer-dependent half of swir's SamplingParams.

    The engine cannot tokenize phrases per request, so the convergence and
    termination phrases, the math-symbol ids and the line-break id are encoded
    here -- with the same calls the HF path uses, so both backends inject
    byte-identical sequences.
    """
    from src.generation_utils import get_math_symbols_ids

    convergence_words = "</think>" if "Qwen" in model_name else "\n\n</think>\n\n"
    termination_words = "</think>\n\nThe final answer is"
    return {
        "swir": True,
        "swir_alpha": alpha,
        "swir_max_switch_count": max_switch_count,
        "swir_math_token_ids": sorted(get_math_symbols_ids(tokenizer)),
        "swir_convergence_token_ids": tokenizer.encode(
            convergence_words, add_special_tokens=False),
        "swir_termination_token_ids": tokenizer.encode(
            termination_words, add_special_tokens=False),
        "swir_linebreak_token_id": tokenizer.encode(
            "\n", add_special_tokens=False)[-1],
    }


def selar_sampling_kwargs(tokenizer, *, selar_topk, entropy_threshold):
    """The tokenizer-dependent half of selar's SamplingParams: the math ids
    come from the same call the HF path uses, so both backends keep the same
    tokens discrete."""
    from src.generation_utils import get_math_symbols_ids

    return {
        "selar": True,
        "selar_topk": selar_topk,
        "selar_entropy_threshold": entropy_threshold,
        "selar_math_token_ids": sorted(get_math_symbols_ids(tokenizer)),
    }


def build_sampling_params(*, greedy, temperature, top_p, top_k, min_p,
                          presence_penalty, max_new_tokens, seed, stop_token_ids,
                          soft_thinking=False, soft_topk=10,
                          soft_entropy_threshold=0.01, soft_patience=256,
                          swir_kwargs=None, selar_kwargs=None):
    """Translate the run.py flags into vLLM's SamplingParams.

    For greedy the filters are set to their neutral values rather than passed through.
    vLLM ignores them once temperature is 0 -- it branches to argmax before they are ever
    applied -- so this changes nothing about the tokens; it just keeps the recorded params
    honest about what actually ran. The presence penalty is *not* neutralised, because
    unlike the filters it does change which token is the argmax.
    """
    from vllm import SamplingParams

    if greedy:
        temperature, top_p, top_k, min_p, seed = 0.0, 1.0, 0, 0.0, None
    soft_kwargs = {}
    if soft_thinking:
        soft_kwargs = {
            "soft_thinking": True,
            "soft_topk": soft_topk,
            "soft_entropy_threshold": soft_entropy_threshold,
            "soft_patience": soft_patience,
        }
    assert not (swir_kwargs and selar_kwargs), "one method per run"
    if swir_kwargs:
        soft_kwargs = dict(swir_kwargs)
    if selar_kwargs:
        soft_kwargs = dict(selar_kwargs)
    return SamplingParams(
        n=1,
        **soft_kwargs,
        temperature=temperature,
        top_p=top_p,
        # vLLM spells "no top_k" as 0; run.py inherits transformers' convention where 0
        # is also off, so the two agree without translation.
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
    )


def generate_token_ids(llm, tokenizer, texts, sampling_params):
    """Generate for every prompt at once; return the generated ids, prompt excluded.

    Prompts are handed over pre-tokenised. The chat template was already applied with
    `tokenize=False`, so letting vLLM re-tokenise the string would risk a second set of
    special tokens on families that add them; tokenising here with the same call the HF
    path uses removes the question.
    """
    from vllm import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=tokenizer(text).input_ids) for text in texts]
    # Requests finish out of order under continuous batching, but LLM.generate sorts by
    # request id before returning, so this list is back in submission order.
    outputs = llm.generate(prompts, sampling_params)
    return [list(o.outputs[0].token_ids) for o in outputs]

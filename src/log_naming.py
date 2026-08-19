"""Log filename construction, shared by scripts/run.py and scripts/merge.py.

Kept in one place because the two used to build the name independently and drifted --
merge.py's glob missed every selar and swir log and silently wrote a merged file
reporting 0% accuracy. The seed tag below has exactly that failure mode, so both
callers now go through this function.
"""


def log_stem(model_name, dataset_name, method, max_new_tokens, seed,
             temperature=None, presence_penalty=None,
             selar_topk=None, entropy_threshold=None,
             alpha=None, max_switch_count=None,
             soft_topk=None, soft_entropy_threshold=None, soft_patience=None,
             backend="hf"):
    """Per-run log filename stem, without the `_rank{N}.json` / `_merged.json` suffix.

    The seed is part of the name because cot, selar and swir all sample: seeds 0..3 are
    four distinct runs of one configuration and would otherwise overwrite each other.
    It is tagged for every method, cot_greedy included, so the two callers never have to
    agree on which methods are stochastic -- `--method cot --no-do_sample` is greedy too.

    The sampling recipe is tagged because Qwen3 and Qwen3.5 recommend different
    thinking-mode values (0.6/0.0 vs 1.0/1.5), so the same model and dataset can be run
    under two recipes that must not land on one filename.

    `presence_penalty` is tagged for **every** method, cot_greedy included: it subtracts
    from the logits of already-emitted tokens, which reorders them and so changes what
    argmax picks. Temperature is dropped for cot_greedy only, where it genuinely is a
    no-op -- dividing by a positive constant is monotonic, and top_k/top_p/min_p only ever
    mask tokens below the maximum, so none of them can move the argmax.

    `backend` is tagged for everything except the default "hf". The two backends are meant
    to agree, but they are separate implementations of the sampler and their kernels are
    not bit-identical, so a vLLM run is a different measurement and must not overwrite the
    HF run it is being checked against. Leaving "hf" untagged keeps every log written
    before the vLLM backend existed matchable by the same stem.
    """
    model_name = model_name.split("/")[-1]
    if method == "selar":
        method_tag = f"{method}_k{selar_topk}_t{entropy_threshold}"
    elif method == "swir":
        method_tag = f"{method}_a{alpha}_s{max_switch_count}"
    elif method == "soft":
        method_tag = f"{method}_n{soft_topk}_t{soft_entropy_threshold}_k{soft_patience}"
    else:
        method_tag = method
    stem = f"{model_name}_{dataset_name}_{method_tag}_{max_new_tokens}"
    if method != "cot_greedy":
        # "temp" rather than "t": selar already spends "_t" on its entropy threshold.
        stem += f"_temp{temperature}"
    stem += f"_pp{presence_penalty}"
    if backend and backend != "hf":
        stem += f"_{backend}"
    return f"{stem}_seed{seed}"

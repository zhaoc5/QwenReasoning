import os
import re
import torch
import torch.nn.functional as F
import random
import numpy as np

try:
    from .hybrid_cache_compat import batch_select_hybrid_cache
except ImportError:  # imported as a top-level module rather than as part of the src package
    from hybrid_cache_compat import batch_select_hybrid_cache


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except Exception:
        pass


def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if min_p > 0:
        # min_p is defined *relative to the most likely token*: drop everything below
        # min_p * max_prob. Comparing against min_p directly is a different filter --
        # on a flat distribution an absolute 0.001 can cut nearly the whole vocabulary,
        # while the intended threshold scales down with the peak and cuts almost nothing.
        probs = F.softmax(logits, dim=-1)
        threshold = min_p * probs.amax(dim=-1, keepdim=True)
        logits = torch.where(probs < threshold, float('-inf'), logits)
    return logits


def disable_cudnn_sdpa():
    """Take cuDNN out of the SDPA backend choice. Call once before generating.

    These loops decode one token at a time against a mask that grows by one column every
    step, so every step presents `scaled_dot_product_attention` with a shape it has never
    seen. cuDNN caches an execution plan per shape and never evicts, which costs about
    1 MiB *per generated token* of host memory -- 80GB over an 81920-token budget, enough
    to blow a cgroup memory limit and get the process OOM-killed mid-run.

    Measured on Qwen3.5-4B, batch 4, H100, decoding 400 steps from a fresh length range:

        cudnn_sdp=on    +1044 KiB/step host    72.4 ms/step
        cudnn_sdp=off      -32 KiB/step host   35.6 ms/step

    So dropping cuDNN both fixes the growth and halves the step time -- for this shape
    (q_len=1, long kv) the flash/mem-efficient backends are simply the better choice.
    Every backend computes the same attention, so results are unaffected.
    """
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def apply_presence_penalty(logits, seen_tokens, presence_penalty):
    """OpenAI-style presence penalty: a flat subtraction from every token already emitted.

    Binary in the count -- a token emitted once is penalised exactly as much as one emitted
    fifty times (counting is ``frequency_penalty``'s job) -- and subtractive on the logit
    rather than divisive (``repetition_penalty``'s job).

    ``seen_tokens`` is a ``[batch, vocab]`` bool mask over the tokens this run has
    *generated*; prompt tokens are excluded. That, and applying this to the raw logits
    before the temperature divide, matches vLLM's sampler, which is the implementation the
    Qwen model cards' recommended values were tuned against.
    """
    if not presence_penalty:
        return logits
    return logits - presence_penalty * seen_tokens.to(logits.dtype)


def _new_seen_mask(presence_penalty, batch, vocab_size, device):
    """Allocate the presence-penalty bookkeeping mask, or None when the penalty is off.

    ``[batch, vocab]`` of bool is ~7MB at Qwen3.5's 248320-token vocabulary and a batch of
    30, so there is no reason to be cleverer than a dense mask.
    """
    if not presence_penalty:
        return None
    return torch.zeros((batch, vocab_size), dtype=torch.bool, device=device)


def _mark_seen(seen_tokens, next_tokens):
    """Record the tokens just emitted. Call after the emitted token is final."""
    if seen_tokens is None:
        return
    rows = torch.arange(next_tokens.shape[0], device=next_tokens.device)
    seen_tokens[rows, next_tokens] = True


def get_math_symbols_ids(tokenizer):
    math_symbols = [
        "+", "-", "*", "/", "^", "=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx", "\\sim", "\\equiv", "\\to", "\\implies", "\\iff",
        "(", ")", "[", "]", "{", "}", "\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}",
        "\\begin{pmatrix}", "\\end{pmatrix}",
        "\\frac", "\\dfrac", "\\sqrt", "\\sqrt[]",
        "\\in", "\\notin", "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\cup", "\\cap", "\\emptyset", "\\varnothing",
        "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\lambda", "\\mu", "\\nu",
        "\\sin", "\\cos", "\\tan", "\\arcsin", "\\arccos", "\\arctan", "\\log", "\\ln", "\\exp",
        "_", "\\binom", "\\choose", "\\cdot", "\\dots", "\\ldots", "\\cdots", "\\vdots", "\\ddots",
        "\\mathbb", "\\mathbf", "\\mathrm", "\\text", "\\mbox",
        "\\infty", "\\circ", "\\prime", "\\ast", "\\star", "\\triangle", "\\triangleleft", "\\triangleright", "\\perp", "\\parallel", "\\angle",
        "\\boxed", "\\overline", "\\underline", "\\lceil", "\\rceil", "\\lfloor", "\\rfloor", "\\left", "\\right", "\\mid", "|", "\\vert", "\\Vert",
        "\\because", "\\therefore", "\\forall", "\\exists", "\\wedge", "\\vee", "\\neg",
        "\\sum", "\\prod", "\\int", "\\lim", "\\min", "\\max", "\\arg", "\\deg", "\\gcd", "\\operatorname",
        "\\cot", 
        "\\cotg", "\\sec", "\\csc",
    ]
    math_symbols += [chr(c) for c in range(ord('0'), ord('9')+1)]
    math_symbols += [chr(c) for c in range(ord('a'), ord('z')+1)]
    math_symbols += [chr(c) for c in range(ord('A'), ord('Z')+1)]
    math_token_ids = set()
    for symbol in math_symbols:
        math_token_ids.update(tokenizer.encode(symbol, add_special_tokens=False))
    return math_token_ids


def resolve_stop_token_ids(model, tokenizer):
    """Every token id that should terminate generation, across Qwen3 / Qwen3.5.

    ``tokenizer.eos_token_id`` alone is not enough on either family: Qwen3's generation
    config lists both ``<|im_end|>`` (151645) and ``<|endoftext|>`` (151643) while the
    tokenizer exposes only the first, and Qwen3.5's generation config names
    ``<|endoftext|>`` (248044) where the tokenizer's ``eos_token`` is ``<|im_end|>``
    (248046). Missing a terminator does not corrupt the text -- it just lets a finished
    sequence keep decoding to ``max_new_tokens``, which at 38912 steps is expensive.
    """
    ids = set()
    for source in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        eos = getattr(source, "eos_token_id", None) if source is not None else None
        if eos is None:
            continue
        ids.update(eos if isinstance(eos, (list, tuple, set)) else [eos])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(i for i in ids if i is not None)


def resolve_stop_token_ids_without_model(model_name, tokenizer):
    """Same terminator set as ``resolve_stop_token_ids``, read straight from the hub.

    The vLLM backend never builds a ``transformers`` model, but it has to be handed the
    same stop ids -- otherwise the two backends disagree about when a sequence is finished
    and their length statistics stop being comparable.

    Reproducing what the model object would have exposed takes two steps that are easy to
    miss, and Qwen3.5 needs both:

    * It ships **no** ``generation_config.json``. ``AutoModelForCausalLM`` silently falls
      back to ``GenerationConfig.from_model_config`` in that case, so this does too.
    * ``Qwen3_5Config`` is a composite (text tower plus vision tower) whose *top level*
      ``eos_token_id`` is ``None``; the real value lives in ``.text_config``. The HF path
      never notices because ``AutoModelForCausalLM`` resolves the text tower and hands
      back its config. Reading the top level alone finds nothing, and the run quietly
      loses ``<|endoftext|>`` (248044) as a terminator -- leaving only the tokenizer's
      ``<|im_end|>``.
    """
    from types import SimpleNamespace
    from transformers import AutoConfig, GenerationConfig

    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        config = None
    if config is not None:
        # Match what AutoModelForCausalLM hands back for composite checkpoints. On plain
        # causal LMs this returns the config itself.
        config = getattr(config, "get_text_config", lambda: config)() or config

    try:
        generation_config = GenerationConfig.from_pretrained(model_name)
    except Exception:
        generation_config = None
        if config is not None:
            try:
                generation_config = GenerationConfig.from_model_config(config)
            except Exception:
                # The tokenizer's eos is the floor; resolve_stop_token_ids tolerates a
                # missing source entirely.
                pass

    shim = SimpleNamespace(generation_config=generation_config, config=config)
    return resolve_stop_token_ids(shim, tokenizer)


def _stop_ids_tensor(stop_token_ids, tokenizer, device):
    """Normalise the ``stop_token_ids`` kwarg into a tensor, falling back to the eos."""
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    elif isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    stop_token_ids = [i for i in stop_token_ids if i is not None]
    if not stop_token_ids:
        return None
    return torch.tensor(sorted(set(stop_token_ids)), dtype=torch.long, device=device)


def _finished_mask(next_tokens, stop_ids_tensor):
    """Boolean mask over the running batch: which sequences just emitted a stop token."""
    if stop_ids_tensor is None:
        return torch.zeros_like(next_tokens, dtype=torch.bool)
    return (next_tokens.unsqueeze(-1) == stop_ids_tensor).any(dim=-1)


def generate_cot(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)
    stop_token_ids  = kwargs.get("stop_token_ids", None)
    presence_penalty = kwargs.get("presence_penalty", 0.0)

    stream_callback = kwargs.pop("stream_callback", None)

    # ============================================

    batch_size = input_ids.shape[0]
    device = input_ids.device
    stop_ids_tensor = _stop_ids_tensor(stop_token_ids, tokenizer, device)

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    generated = input_ids.clone()
    attn_mask = attention_mask.clone()
    past_key_values = None
    seen_tokens = None

    for step in range(max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attn_mask}
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
            attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
            model_inputs = {"input_ids": next_tokens.unsqueeze(1), "past_key_values": past_key_values, "attention_mask": attn_mask}

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]  # [cur_batch, vocab]
        if seen_tokens is None:
            seen_tokens = _new_seen_mask(presence_penalty, cur_batch, next_token_logits.shape[-1], device)
        logits = apply_presence_penalty(next_token_logits, seen_tokens, presence_penalty)
        logits = logits / temperature
        logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        _mark_seen(seen_tokens, next_tokens)

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        cur_finished = _finished_mask(next_tokens, stop_ids_tensor)
        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        attention_mask = attention_mask[keep_idx]
        attn_mask = attn_mask[keep_idx]
        if seen_tokens is not None:
            seen_tokens = seen_tokens[keep_idx]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=generated.device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out

def generate_soft(model, tokenizer, **kwargs):
    """Soft Thinking (arXiv:2505.15778), training-free continuous-concept reasoning.

    The thinking phase never commits to a token. At each step the next input is the
    probability-weighted mixture of token embeddings -- the paper's "concept token",
    Eq. 5: ``e_next = sum_k p[k] e(k)``. Definition 2 writes that over the whole
    vocabulary; the Complexity Analysis states the implementation "apply[s] a top-k
    top-p filter ... select[s] top-n tokens with highest probability, renormalize[s],
    and then perform[s] a single dense matrix-vector multiplication over the filtered
    subset", which is what ``soft_topk`` controls here.

    Only the thinking phase is continuous. Once ``</think>`` becomes the most probable
    token -- or Cold Stop fires -- that row switches to ordinary discrete sampling for
    the answer, per "All output stage tokens are sampled in the usual discrete manner".

    Cold Stop (Eq. 6) guards against the OOD collapse that continuous inputs can cause:
    count consecutive steps whose concept-token entropy is below ``soft_entropy_threshold``,
    reset the counter whenever it is not, and on reaching ``soft_patience`` force
    ``</think>`` to end reasoning.

    The entropy is computed on the model's **full** distribution, before top_k/top_p/min_p.
    Measuring it after the filters does not work: top_p adapts to confidence, so the moment
    the top token clears 0.95 the nucleus holds one token and the renormalized entropy is
    exactly 0. Over 1500 real decoding steps on Qwen3.5-4B the filtered entropy had
    p95 = 0.0000 under both sampling recipes tried -- it measures top_p, not the model --
    against p95 of 0.0724 and 0.0014 for the full-vocabulary entropy.

    Cold Stop is also sensitive to the sampling recipe, because tau and the patience were
    tuned together with it. On the same 1500 steps, the longest run of consecutive
    sub-tau steps was 249 under the paper's recipe (temperature 0.6, presence_penalty 0)
    but only 71 under Qwen3.5's thinking recipe (temperature 1.0, presence_penalty 1.5) --
    against a patience of 256. Run this method at its published sampling settings, or
    Cold Stop never fires and every sample burns the whole token budget.

    A row that is still thinking records its argmax token into the returned sequence, so
    the trace stays readable and ``</think>`` can be detected, while the *embedding* fed
    back is the mixture. That mirrors what selar and swir already do here.
    """
    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0.0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)
    stop_token_ids  = kwargs.get("stop_token_ids", None)
    presence_penalty = kwargs.get("presence_penalty", 0.0)

    # ---- Soft Thinking ----
    soft_topk               = kwargs.pop("soft_topk", 10)
    soft_entropy_threshold  = kwargs.pop("soft_entropy_threshold", 0.01)
    soft_patience           = kwargs.pop("soft_patience", 256)
    after_temperature       = kwargs.pop("after_thinking_temperature", None)
    after_top_p             = kwargs.pop("after_thinking_top_p", None)
    after_top_k             = kwargs.pop("after_thinking_top_k", None)
    end_thinking_token_id   = kwargs.pop("end_thinking_token_id", None)

    stream_callback = kwargs.pop("stream_callback", None)

    # The answer phase gets its own sampling knobs in the reference implementation; when
    # they are not given, fall back to the thinking-phase ones.
    after_temperature = temperature if after_temperature is None else after_temperature
    after_top_p       = top_p if after_top_p is None else after_top_p
    after_top_k       = top_k if after_top_k is None else after_top_k

    # ============================================

    batch_size = input_ids.shape[0]
    device = input_ids.device
    stop_ids_tensor = _stop_ids_tensor(stop_token_ids, tokenizer, device)
    E = model.get_input_embeddings().weight  # [vocab, hidden]

    if end_thinking_token_id is None:
        end_thinking_token_id = tokenizer.convert_tokens_to_ids("</think>")
    end_thinking_emb = E[end_thinking_token_id]

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    attn_mask = attention_mask.clone()
    past_key_values = None
    next_embeds = None

    # Per-row Cold Stop state. Qwen3.5's chat template pre-fills "<think>\n" into the
    # prompt, so every row opens inside the thinking phase.
    in_thinking = torch.ones(batch_size, dtype=torch.bool, device=device)
    low_entropy_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    seen_tokens = None
    cold_stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        cur_batch = input_ids.shape[0] if next_embeds is None else next_embeds.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": input_ids, "attention_mask": attn_mask}
        else:
            attn_mask = torch.cat(
                [attn_mask, torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)], dim=1)
            model_inputs = {
                "inputs_embeds": next_embeds.unsqueeze(1),
                "past_key_values": past_key_values,
                "attention_mask": attn_mask,
            }

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        logits_original = outputs.logits[:, -1, :]  # [cur_batch, vocab]

        if seen_tokens is None:
            seen_tokens = _new_seen_mask(presence_penalty, cur_batch, logits_original.shape[-1], device)
        penalised = apply_presence_penalty(logits_original, seen_tokens, presence_penalty)

        # ---- thinking phase: build the concept token ----
        think_logits = apply_sampling_filter(
            penalised / temperature, top_k=top_k, top_p=top_p, min_p=min_p)
        think_probs = F.softmax(think_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(
            think_probs, k=min(soft_topk, think_probs.shape[-1]), dim=-1)
        weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        concept_embeds = torch.sum(weights.unsqueeze(-1) * E[topk_indices], dim=1)

        # Eq. 6 over the model's *full* distribution, before top_k/top_p/min_p. Measuring
        # it after the filters is degenerate: top_p adapts to confidence, so as soon as
        # the top token clears 0.95 the nucleus is a single token and the renormalized
        # entropy is exactly 0. Measured on Qwen3.5-4B over 1500 real decoding steps, the
        # filtered entropy had p95 = 0.0000 under both sampling recipes tried -- no
        # variance at all, so it reports on top_p rather than on the model -- while the
        # full-vocabulary entropy had p95 = 0.0724 and 0.0014 respectively.
        full_probs = F.softmax(penalised / temperature, dim=-1)
        entropy = -(full_probs * torch.log(full_probs.clamp_min(1e-12))).sum(dim=-1)
        think_tokens = topk_indices[:, 0]  # argmax; what gets recorded and checks </think>

        # ---- answer phase: ordinary discrete sampling ----
        ans_logits = apply_sampling_filter(
            penalised / after_temperature, top_k=after_top_k, top_p=after_top_p, min_p=min_p)
        ans_probs = F.softmax(ans_logits, dim=-1)
        if do_sample:
            ans_tokens = torch.multinomial(ans_probs, num_samples=1).squeeze(-1)
        else:
            ans_tokens = torch.argmax(ans_probs, dim=-1)

        # ---- Cold Stop ----
        low = (entropy < soft_entropy_threshold) & in_thinking
        low_entropy_steps = torch.where(low, low_entropy_steps + 1,
                                        torch.zeros_like(low_entropy_steps))
        fire = in_thinking & (low_entropy_steps >= soft_patience)
        cold_stopped = cold_stopped | fire

        # A row leaves the thinking phase when </think> is its most probable token, or
        # when Cold Stop forces the issue. Either way the token it emits *this* step is
        # </think> itself, so the model sees the boundary it is about to answer after.
        natural_end = in_thinking & (think_tokens == end_thinking_token_id)
        leaving = natural_end | fire

        next_tokens = torch.where(in_thinking, think_tokens, ans_tokens)
        next_tokens = torch.where(
            leaving, torch.full_like(next_tokens, end_thinking_token_id), next_tokens)
        _mark_seen(seen_tokens, next_tokens)

        # Rows still thinking after this step feed back the concept token; rows that just
        # left feed back </think>; rows already answering feed back their sampled token.
        still_thinking = in_thinking & ~leaving
        next_embeds = torch.where(
            still_thinking.unsqueeze(-1), concept_embeds,
            torch.where(leaving.unsqueeze(-1),
                        end_thinking_emb.unsqueeze(0).expand_as(concept_embeds),
                        E[ans_tokens]),
        )
        in_thinking = in_thinking & ~leaving
        low_entropy_steps = torch.where(leaving, torch.zeros_like(low_entropy_steps),
                                        low_entropy_steps)

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        # Only the answer phase can terminate: a stop token is not meaningful while the
        # row is still consuming continuous inputs.
        cur_finished = _finished_mask(next_tokens, stop_ids_tensor) & (~in_thinking)
        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break

        next_embeds = next_embeds[keep_idx]
        attn_mask = attn_mask[keep_idx]
        in_thinking = in_thinking[keep_idx]
        low_entropy_steps = low_entropy_steps[keep_idx]
        cold_stopped = cold_stopped[keep_idx]
        if seen_tokens is not None:
            seen_tokens = seen_tokens[keep_idx]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(
            keep_idx, dtype=torch.long, device=device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def generate_selar(model, tokenizer, **kwargs):
    """
    SeLaR: Self-Adaptive Latent Reasoning

    Key mechanisms:
    1. Compute top-k token probabilities and entropy at each step
    2. When entropy > threshold:
       - Use weighted embedding (top-k probabilities) as next input
       - Apply contrastive regularization to prevent collapsing to top-1
    3. When entropy <= threshold:
       - Standard discrete decoding (sample from top-k)
    """
    
    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)
    stop_token_ids  = kwargs.get("stop_token_ids", None)

    presence_penalty = kwargs.get("presence_penalty", 0.0)

    # ---- **SeLaR-specific ----
    selar_topk          = kwargs.pop("selar_topk", 5)
    entropy_threshold    = kwargs.pop("entropy_threshold", 0.3)
    math_ids_tensor      = kwargs.pop("math_ids_tensor", None)
    contrastive_weight   = kwargs.pop("contrastive_weight", 1.0)  # Contrastive regularization strength
    
    stream_callback = kwargs.pop("stream_callback", None)

    # ============================================

    batch_size = input_ids.shape[0]
    device = input_ids.device
    stop_ids_tensor = _stop_ids_tensor(stop_token_ids, tokenizer, device)

    # Get embedding matrix
    E = model.get_input_embeddings().weight  # [vocab_size, hidden_dim]
    
    # Compute max entropy for normalization
    max_entropy = torch.log(torch.tensor(float(selar_topk), device=device))

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    generated = input_ids.clone()
    attn_mask = attention_mask.clone()
    past_key_values = None
    
    # State tracking for SeLaR
    use_soft_input = torch.zeros(batch_size, dtype=torch.bool, device=device)
    soft_embeds = None
    next_tokens = None
    seen_tokens = None
        
    for step in range(max_new_tokens):
        cur_batch = generated.shape[0] if soft_embeds is None else soft_embeds.shape[0]
        if cur_batch == 0:
            break

        # Prepare model inputs
        if past_key_values is None:
            # First step: use input_ids
            model_inputs = {"input_ids": generated, "attention_mask": attn_mask}
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
            attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
            
            # Use soft embeddings for high-entropy samples, discrete tokens for low-entropy
            if use_soft_input.any():
                model_inputs = {
                    "inputs_embeds": soft_embeds.unsqueeze(1),  # [cur_batch, 1, hidden_dim]
                    "past_key_values": past_key_values,
                    "attention_mask": attn_mask
                }
            else:
                model_inputs = {
                    "input_ids": next_tokens.unsqueeze(1),
                    "past_key_values": past_key_values,
                    "attention_mask": attn_mask
                }

        # Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True, return_dict=True)
        
        past_key_values = outputs.past_key_values
        
        # Get logits directly from model output
        logits_original = outputs.logits[:, -1, :]  # [cur_batch, vocab_size]

        # SeLaR reads its entropy gate off the same distribution it samples from, so a
        # non-zero presence penalty shifts the gate as well as the sampling. That is the
        # consistent reading -- the gate is about how uncertain the *decoding* step is --
        # but it does mean selar with a penalty is not comparable to selar without one.
        if seen_tokens is None:
            seen_tokens = _new_seen_mask(presence_penalty, cur_batch, logits_original.shape[-1], device)
        logits = apply_presence_penalty(logits_original, seen_tokens, presence_penalty)
        logits = logits / temperature
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
        probs = F.softmax(logits_filtered, dim=-1)
        
        # Get top-k for entropy computation and potential soft embedding
        topk_probs, topk_indices = torch.topk(probs, k=selar_topk, dim=-1)  # [cur_batch, selar_topk]
        
        # Compute entropy from top-k probabilities
        probs_sum = topk_probs.sum(dim=-1, keepdim=True)
        probs_normalized = topk_probs / (probs_sum + 1e-10)
        log_probs = torch.log(probs_normalized + 1e-10)
        entropy = -torch.sum(probs_normalized * log_probs, dim=-1)  # [cur_batch]
        normalized_entropy = torch.clamp(entropy / max_entropy, 0.0, 1.0)  # [cur_batch]
        
        # Sample tokens (for low-entropy cases and for output)
        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [cur_batch]
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # [cur_batch]
        _mark_seen(seen_tokens, next_tokens)

        # ===== Math Symbol Special Handling =====
        # Force discrete tokens for math symbols
        force_discrete = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            force_discrete = force_discrete | is_math_token
        
        # Decide: high entropy -> soft input next step; low entropy -> discrete sampling
        # But override with force_discrete for math symbols
        is_high_entropy = (normalized_entropy >= entropy_threshold) & (~force_discrete)
        
        # Prepare next step inputs. The gate is per-sample: rows below the entropy
        # threshold must keep their discrete embedding even when another row in the same
        # batch is above it, otherwise the selective mechanism collapses into
        # always-on soft reasoning as soon as the batch is wide.
        discrete_embeds = E[next_tokens]  # [cur_batch, hidden_dim]
        if is_high_entropy.any():
            # Get embeddings for top-k tokens
            topk_embeddings = E[topk_indices]  # [cur_batch, selar_topk, hidden_dim]

            # Construct base soft embedding as weighted sum using original probabilities
            base_soft_embed = torch.sum(
                probs_normalized.unsqueeze(-1) * topk_embeddings,  # [cur_batch, selar_topk, 1] * [cur_batch, selar_topk, hidden_dim]
                dim=1
            )  # [cur_batch, hidden_dim]

            # ===== Contrastive Regularization =====
            # Push soft embedding away from top-1 to prevent early collapse
            top1_embed = topk_embeddings[:, 0, :]  # [cur_batch, hidden_dim]

            # Compute direction from top-1 to soft_embed
            diff_direction = base_soft_embed - top1_embed  # [cur_batch, hidden_dim]
            diff_norm = torch.norm(diff_direction, dim=-1, keepdim=True) + 1e-10
            diff_unit = diff_direction / diff_norm

            # Scale contrastive strength by entropy: higher entropy -> stronger push
            contrastive_strength = contrastive_weight * normalized_entropy.unsqueeze(-1)  # [cur_batch, 1]

            # Apply contrastive regularization
            regularized = base_soft_embed + contrastive_strength * diff_unit * diff_norm
            soft_embeds = torch.where(is_high_entropy.unsqueeze(-1), regularized, discrete_embeds)
        else:
            soft_embeds = discrete_embeds  # Use discrete embedding when no high entropy

        # Store state for next iteration
        use_soft_input = is_high_entropy

        # Record generated tokens
        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])

        # Check for finished sequences
        cur_finished = _finished_mask(next_tokens, stop_ids_tensor)

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        
        # Filter ongoing sequences
        generated = generated[keep_idx] if generated is not None else None
        next_tokens = next_tokens[keep_idx]
        attention_mask = attention_mask[keep_idx]
        attn_mask = attn_mask[keep_idx]
        use_soft_input = use_soft_input[keep_idx]
        if soft_embeds is not None:
            soft_embeds = soft_embeds[keep_idx]
        if seen_tokens is not None:
            seen_tokens = seen_tokens[keep_idx]
        
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)

    # Final output
    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out

def generate_swir(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids      = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature     = kwargs.get("temperature", 1.0)
    top_p           = kwargs.get("top_p", 1.0)
    top_k           = kwargs.get("top_k", 0)
    min_p           = kwargs.get("min_p", 0)
    max_new_tokens  = kwargs.get("max_new_tokens", 32768)
    do_sample       = kwargs.get("do_sample", True)
    stop_token_ids  = kwargs.get("stop_token_ids", None)

    presence_penalty = kwargs.get("presence_penalty", 0.0)

    # ---- swir ----
    alpha_0                = kwargs.pop("alpha_0", 1.0) # adjustable
    beta_0                 = kwargs.pop("beta_0", 0.7)
    window_size            = kwargs.pop("window_size", 512)
    thinking_token_id      = kwargs.pop("thinking_token_id", None)
    end_thinking_token_id  = kwargs.pop("end_thinking_token_id", None)
    max_switch_count       = kwargs.pop("max_switch_count", None) # adjustable for efficiency
    math_ids_tensor        = kwargs.pop("math_ids_tensor", None)
    convergence_words      = kwargs.get("convergence_words", "</think>")
    termination_words      = kwargs.get("termination_words", "</think>\n\nThe final answer is")
    termination_max_tokens = kwargs.pop("termination_max_tokens", 32)

    stream_callback       = kwargs.pop("stream_callback", None)

    # ============================================

    batch_size, device = input_ids.shape[0], input_ids.device
    stop_ids_tensor = _stop_ids_tensor(stop_token_ids, tokenizer, device)
    E = model.get_input_embeddings().weight  # [vocab_size, dim]
    if thinking_token_id is None or end_thinking_token_id is None:
        thinking_token_id = tokenizer.convert_tokens_to_ids("<think>")
        end_thinking_token_id = tokenizer.convert_tokens_to_ids("</think>")
    start_thinking_emb, end_thinking_emb = E[thinking_token_id], E[end_thinking_token_id]
    # Encode an actual newline: convert_tokens_to_ids("\\n") looks up the two-character
    # token "\n" (id 1699 on Qwen3, 1639 on Qwen3.5), not the line break (id 198).
    line_break_emb = E[tokenizer.encode("\n", add_special_tokens=False)[-1]]
    past_key_values = None
    seen_tokens = None
        
    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size)) # bs >= 1 is supported
    mode = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: soft, 1: normal
    mode_stay_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    locked_normal_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    if max_switch_count is not None:
        switch_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        convergence_ids = tokenizer.encode(convergence_words, add_special_tokens=False)
        termination_ids = tokenizer.encode(termination_words, add_special_tokens=False)
        injecting = torch.zeros(batch_size, dtype=torch.bool, device=device)
        inject_queues = [[] for _ in range(batch_size)]
        answer_budget = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        cur_batch = attention_mask.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids.clone(), 
                "attention_mask": attention_mask,
            }
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "inputs_embeds": last_emb.unsqueeze(1), 
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        
        logits_original = outputs.logits[:, -1, :]
        # probs_original stays penalty-free on purpose: it drives the entropy signal that
        # switches soft/normal mode and it builds the soft embedding, both of which are
        # meant to read the model's own distribution, not the sampler's view of it.
        probs_original = F.softmax(logits_original, dim=-1)
        if seen_tokens is None:
            seen_tokens = _new_seen_mask(presence_penalty, cur_batch, logits_original.shape[-1], device)
        logits = apply_presence_penalty(logits_original, seen_tokens, presence_penalty)
        logits = logits / temperature
        logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)  # [B, N, V]
        probs = F.softmax(logits_filtered, dim=-1)

        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # [B, N]
        locked_normal_mask = locked_normal_mask | (next_tokens == end_thinking_token_id)

        if max_switch_count is not None and injecting.any():
            mask_list = [injecting[i].item() and len(inject_queues[i]) > 0 for i in range(cur_batch)]
            force_mask = torch.tensor(mask_list, device=device, dtype=torch.bool)
            if force_mask.any():
                force_toks = torch.tensor([inject_queues[i].pop(0) for i in range(cur_batch) if mask_list[i]], \
                                          device=device, dtype=torch.long)
                next_tokens[force_mask] = force_toks
            if injecting.any():
                done_mask = torch.tensor([injecting[i] and (len(inject_queues[i]) == 0) for i in range(cur_batch)], \
                                         device=device, dtype=torch.bool)
                injecting[done_mask] = False

        # After the injection block, not before it -- a forced convergence/termination token
        # overwrites what was sampled, and the penalty must track what was actually emitted.
        _mark_seen(seen_tokens, next_tokens)

        cur_entropy = -(probs_original * (probs_original.clamp(min=1e-12).log())).sum(dim=-1)
        if step == 0:
            cur_ref_entropy = cur_entropy.clone()
        else:
            mode_stay_steps += 1
            allow_switch = (mode_stay_steps >= window_size)
            to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy)
            to_soft = (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch & (~locked_normal_mask)
            mode[to_normal] = 1
            mode[to_soft] = 0
            mode_stay_steps[to_normal | to_soft] = 0
            cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]
            if max_switch_count is not None:
                switch_count = switch_count + to_normal.long() 
            
        is_normal = (mode == 1) | locked_normal_mask
        if math_ids_tensor is not None:
            is_math_token = (next_tokens.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
            is_normal[is_math_token] = True
        is_soft = ~is_normal
        
        normal_emb = E[next_tokens]
        soft_emb = torch.matmul(probs_original, E)

        alpha = alpha_0 + (1 - alpha_0) * float(step) / float(max_new_tokens)
        if step == 0:
            soft_emb = 0.9 * soft_emb + 0.1 * line_break_emb
        else:
            mixed_emb = alpha * soft_emb + (1 - alpha) * start_thinking_emb
            soft_emb = torch.where(to_soft[:, None], mixed_emb, soft_emb)
        beta = beta_0 + (1 - beta_0) * float(step) / float(max_new_tokens)
        if step > 0:
            mixed_emb = beta * soft_emb + (1 - beta) * end_thinking_emb
            normal_emb = torch.where(to_normal[:, None], mixed_emb, normal_emb)
        last_emb = torch.where(is_soft[:, None], soft_emb, normal_emb)

        if max_switch_count is not None and step > 0:
            trigger = (switch_count >= max_switch_count) & (switch_count <= 2 * max_switch_count) & to_normal
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(convergence_ids)
                injecting = injecting | trigger

            trigger = (switch_count > 2 * max_switch_count) & to_normal
            if trigger.any():
                idx_list = trigger.nonzero(as_tuple=False).squeeze(-1).tolist()
                for i in idx_list:
                    inject_queues[i] = list(termination_ids) 
                injecting = injecting | trigger 
                answer_budget[trigger] = termination_max_tokens
            active = (answer_budget >= 0)
            if active.any():
                answer_budget = torch.where(active, answer_budget - 1, answer_budget)

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if stream_callback is not None:
                stream_callback(all_generated[orig][-1])
        
        cur_finished = _finished_mask(next_tokens, stop_ids_tensor)

        if max_switch_count is not None:
            budget_done = (answer_budget == 0) 
            cur_finished = cur_finished | budget_done

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]
        if len(unfinished_idx) == 0:
            break
        last_emb = last_emb[keep_idx]
        attention_mask = attention_mask[keep_idx]
        mode = mode[keep_idx]
        mode_stay_steps = mode_stay_steps[keep_idx]
        cur_ref_entropy = cur_ref_entropy[keep_idx]
        locked_normal_mask = locked_normal_mask[keep_idx]
        if seen_tokens is not None:
            seen_tokens = seen_tokens[keep_idx]
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)
        if max_switch_count is not None:
            switch_count = switch_count[keep_idx]
            injecting = injecting[keep_idx]
            inject_queues = [inject_queues[i] for i in keep_idx.tolist()]
            answer_budget = answer_budget[keep_idx]

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out

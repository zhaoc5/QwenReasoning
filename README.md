<h1 align="center">QwenReasoning</h1>

<p align="center">
  Test-time reasoning methods for the <b>Qwen3</b>, <b>Qwen3.5</b> and <b>Qwen3.8</b>
  model families, evaluated on five math and science benchmarks.
</p>

## 📌 Overview

An evaluation harness for training-free reasoning methods that intervene in the decoding
loop, feeding soft (probability-weighted) embeddings back into the model instead of always
committing to a discrete token. One entry point, `scripts/run.py`: pick a model, a dataset
and a method.

## 🧩 Methods

| `--method` | What it does | Key flags |
|---|---|---|
| `cot` | Chain-of-thought decoding with sampling | `--temperature`, `--top_p`, `--top_k` |
| `cot_greedy` | Same, argmax instead of sampling | — |
| `soft` | Soft Thinking: the thinking phase feeds back the probability-weighted mixture of the top-n embeddings; Cold Stop forces `</think>` after enough consecutive low-entropy steps | `--soft_topk`, `--soft_entropy_threshold`, `--soft_patience` |
| `swir` | Switches between soft and discrete thinking on entropy trends | `--alpha`, `--max_switch_count` |
| `selar` | Gates soft embeddings on per-step normalized entropy, plus a contrastive term | `--selar_topk`, `--entropy_threshold` |

`swir` and `selar` keep math symbols discrete. All five methods run on both backends;
`--backend vllm` uses the fork's engine-side implementations of `soft`, `swir` and
`selar`.

## 🤖 Supported Models

Any Qwen3, Qwen3.5 or Qwen3.8 checkpoint; everything model-specific is read from the
tokenizer and config.

| | Qwen3 | Qwen3.5 and Qwen3.8 |
|---|---|---|
| Checkpoint class | `Qwen3ForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| Attention | full, every layer | hybrid: 3 linear per full-attention layer |
| `</think>` | `151668` | `248069` |
| Stop tokens | `151645`, `151643` | `248046`, `248044` |
| Sampling temperature | 0.6 | **1.0** |
| `presence_penalty` | — | **1.5** (Qwen3.8: **0.0**) |
| Output length (competition / general) | 38912 / 32768 | **81920** / 32768 |

`top_p 0.95`, `top_k 20`, `min_p 0.0` throughout. `scripts/run.py`'s defaults are Qwen3's —
pass the others explicitly. Qwen3.5 and Qwen3.8 need `transformers>=5.15`. cuDNN is dropped
from the SDPA backends unless you pass `--cudnn_sdp`.

## 🚀 Quick Start

```bash
git clone --recurse-submodules https://github.com/zhaoc5/QwenReasoning.git
conda create -n qwenreasoning python=3.12 && conda activate qwenreasoning
pip install -r requirements.txt
```

Evaluate, then merge the per-rank logs with the **same** seed, sampling values and method
hyperparameters — all are folded into the filename by `src/log_naming.py`:

```bash
torchrun --nproc_per_node 1 --master_port $((RANDOM + 20000)) \
    scripts/run.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k \
    --batch_size 256 --max_new_tokens 32768 --method selar \
    --selar_topk 3 --entropy_threshold 0.5

python scripts/merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k \
    --max_new_tokens 32768 --method selar --seed 41 \
    --selar_topk 3 --entropy_threshold 0.5
```

`--dataset_name` accepts `gsm8k`, `math500`, `aime_2024`, `aime_2025`, `gpqa_diamond`.
The first four ship pre-tokenized; GPQA is gated and its authors ask that plaintext not
be republished, so build it locally -- `run.py` prints the expected format. Answers are
graded with [math-verify](https://github.com/huggingface/Math-Verify), preferring
`\boxed{}`.

### The vLLM backend

vLLM comes in as a submodule — a fork with Soft Thinking implemented in the engine. It is
not in `requirements.txt`; install it separately:

```bash
cd third_party/vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.27.3.dev189+gf4b161d7fc VLLM_USE_PRECOMPILED=1 \
    pip install -e .
```

```bash
python scripts/run.py --backend vllm \
    --model_name Qwen/Qwen3.5-4B --dataset_name aime_2024 --method cot \
    --max_new_tokens 81920 --max_model_len 86016 --seed 0 \
    --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0.0 --presence_penalty 1.5
```

- **No torchrun.** Shard with `--dp_size N --dp_rank K`, one process per shard on its own
  GPU; pass `merge.py --backend vllm` too.
- **`--batch_size` is ignored.** Logs carry a `_vllm` tag, so they never overwrite HF runs.
- **Set `--max_model_len`**, or vLLM reserves KV cache for the checkpoint maximum (262144).
- **Lower `--max_num_seqs`** if vLLM refuses to start — Qwen3.5 holds a Mamba cache block
  per concurrent decode.
- **Prefer `--dp_size` over `--tensor_parallel_size`**; both work.

`--method soft`, `swir` and `selar` set the engine flags they require: prompt embeds,
prefix caching off, and (for the two thinking-block methods) a reasoning parser. `n > 1`
and speculative decoding are refused.

Speed on Qwen3.5-4B / AIME 2024 / H100: HF on 2 GPUs 99 min, vLLM on 1 GPU 23 min, on
2 GPUs 14.5–17 min.

## 📊 Results

AIME 2024 and 2025, 30 problems each, `--max_new_tokens 81920`, vLLM backend, accuracy
in percent. **avg@8 ± std** is pass@1 over seeds 0–7; `cot_greedy` is one deterministic
run.

| Model | Method | AIME 2024 | AIME 2025 | mean tokens | no `</think>` |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | `cot` | **85.0 ± 2.9** | **81.7 ± 5.5** | 35k / 38k | 6% / 10% |
| | `cot_greedy` | 56.7 | 66.7 | 50k / 37k | 47% / 27% |
| | `soft` | 67.5 ± 3.6 | 64.6 ± 5.0 | 35k / 38k | 0% / 0% |
| **Qwen3.5-9B** | `cot` | **91.2 ± 3.3** | **87.9 ± 2.9** | 29k / 34k | 2% / 6% |
| | `cot_greedy` | 76.7 | 70.0 | 32k / 32k | 23% / 20% |
| | `soft` | 82.1 ± 3.7 | 65.8 ± 2.8 | 29k / 37k | 0% / 0% |
| **Qwen3.8-27B** | `cot` | **97.1 ± 2.6** | 95.0 ± 2.9 | 18k / 20k | 3% / 2% |
| | `cot_greedy` | 96.7 | **100.0** | 17k / 17k | 3% / 0% |
| | `soft` | 96.7 ± 2.4 | 99.2 ± 1.4 | 14k / 16k | 3% / 1% |

Last two columns are 2024 / 2025. `cot` and `cot_greedy` run at each family's vendor
recipe, `soft` at the one its paper tuned (T0.6/pp0.0). `scripts/length_report.py` prints
the length and truncation columns per run.

**Greedy fails by looping — until scale fixes it.** Qwen3.5 greedy cuts off 20–47% of its
samples, mostly in repetition loops, and sampling repairs that. On 27B the loops all but
vanish (0–3% cut off) and greedy turns competitive, including 30/30 on AIME 2025.

**Scale buys shorter reasoning.** 27B averages 18–20k tokens against 4B's 35–38k while
scoring 12–13 points higher on `cot`.

**Soft Thinking scales into parity.** On Qwen3.5 it trails `cot` by 9–22 points with no
token saving; on 27B it matches `cot` (96.7 / 99.2 against 97.1 / 95.0) at the shortest
lengths in the table.

### Caveats

- `soft` and `cot` do not share a sampling recipe, so their gap confounds method with
  sampling. A matched `cot` control at T0.6/pp0.0 is the missing run.
- 30 problems is 3.3 points per problem, and a single seed moves by 2 problems on execution
  details alone. Only avg@N is comparable.

## 📁 Repository Structure

```
src/       generation_utils.py  decode loops and sampling
           vllm_backend.py      vLLM route for cot / cot_greedy / soft
           grader.py  log_naming.py  hybrid_cache_compat.py
scripts/   run.py  merge.py  length_report.py
tests/     eight scripts, no GPU or weights: `python tests/<f>.py`
datasets/  four benchmarks pre-tokenized; GPQA is built locally
third_party/vllm/  submodule: the vLLM fork, branch soft-thinking
```

Runs write to `logs/`, gitignored, collapsed by `merge.py`.

## 🔧 Differences from the Upstream Code

Four bug fixes in the code this was built from; three change results.

1. **`selar`'s entropy gate was applied batch-wide** — one row over the threshold fed soft
   embeddings to every row. Now gated per sample.
2. **Generation stopped only on `tokenizer.eos_token_id`**, so a finished sequence could
   decode on to `max_new_tokens`.
3. **`swir`'s line-break embedding looked up the wrong token.**
4. **`merge.py` matched no logs for `selar` and `swir`**, silently reporting 0% accuracy.

`third_party/vllm` implements Soft Thinking, SwiReasoning and SeLaR inside the vLLM
engine, which upstream does not have, and fixes an unrelated upstream bug where an incompatible `flashinfer` import kills
every tensor-parallel run below Python 3.12. The implementation lives in vLLM's V1 GPU
model runner; dense models default to the V2 runner, so `--method soft` pins
`VLLM_USE_V2_MODEL_RUNNER=0` (hybrid Qwen3.5 selects V1 regardless) and the engine refuses
soft requests under V2 rather than silently decoding plain CoT.

That implementation is checked against `src/generation_utils.py:generate_soft`: the
arithmetic agrees to 1e-6 on identical logits, TP=2 and TP=4 track TP=1 for 74 tokens, and
four seeds of AIME 2024 give 73.3% (HF) against 70.0% (vLLM) with 0/120 unclosed
`</think>` on both.

## 📚 Acknowledgements

Built on these projects. Soft Thinking comes first because it introduced the
continuous-concept decoding that `swir` and `selar` build on.

- **[Soft-Thinking](https://github.com/UCSB-AI/Soft-Thinking)** — the `soft` method, from
  *Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space*
  ([arXiv:2505.15778](https://arxiv.org/abs/2505.15778)).
- **[SwiReasoning](https://github.com/sdc17/SwiReasoning)** — the `swir` method, the
  decoding-loop structure, and `src/hybrid_cache_compat.py`.
- **[SeLaReasoning](https://github.com/Parker-rfu/SeLaReasoning)** — the `selar` method,
  from *SeLaR: Selective Latent Reasoning in Large Language Models*
  ([arXiv:2604.08299](https://arxiv.org/abs/2604.08299)).

Please cite those works if you use the corresponding methods. Thanks also to
[Transformers](https://github.com/huggingface/transformers) and
[Qwen](https://github.com/QwenLM/Qwen3).

## 📖 Citation

```bibtex
@misc{qwenreasoning2026,
  author       = {Zhao, Chongyang},
  title        = {QwenReasoning: Test-time Reasoning Methods for Qwen Models},
  year         = {2026},
  howpublished = {\url{https://github.com/zhaoc5/QwenReasoning}}
}
```

## 📄 License

[Apache License 2.0](LICENSE). Derived portions remain under their upstream licenses --
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

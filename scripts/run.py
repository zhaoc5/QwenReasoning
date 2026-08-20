import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.generation_utils import (
    set_seed,
    disable_cudnn_sdpa,
    get_math_symbols_ids,
    resolve_stop_token_ids,
    resolve_stop_token_ids_without_model,
    generate_cot,
    generate_selar,
    generate_soft,
    generate_swir,
)
from src.grader import answer_match
from src.log_naming import log_stem
from src import vllm_backend


def main(args):
    use_vllm = args.backend == "vllm"
    if use_vllm:
        vllm_backend.check_method_supported(args.method)
        if any(v is not None for v in (args.after_thinking_temperature,
                                       args.after_thinking_top_p,
                                       args.after_thinking_top_k)):
            # The engine-side Soft Thinking samples the answer phase with the same
            # recipe as the thinking phase; there is no separate after-thinking
            # recipe to hand these to. Refuse rather than silently run without them.
            raise ValueError(
                "--after_thinking_temperature/top_p/top_k apply to --backend hf only. "
                "The vLLM engine samples the answer phase with the thinking-phase "
                "recipe; leave these unset (they default to the main sampling flags)."
            )
    set_seed(args.seed)
    if not use_vllm and not args.cudnn_sdp:
        # Default off: cuDNN caches an SDPA plan per shape, and a growing mask means a new
        # shape every step -- ~1 MiB per generated token, plus it is 2x slower here.
        # Irrelevant to vLLM, which runs its own paged attention kernels instead of SDPA.
        disable_cudnn_sdpa()

    if use_vllm:
        # Data parallelism is explicit here rather than launched by torchrun: each shard is
        # an ordinary process with its own engine on its own GPUs, and no two shards ever
        # need to talk. Under torchrun every rank would instead evaluate the whole dataset
        # and write to one log path, so refuse that outright.
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            raise RuntimeError(
                "--backend vllm cannot be launched with torchrun. Shard it with "
                "--dp_size/--dp_rank instead: one plain process per shard, each pinned to "
                "its own GPU(s) with CUDA_VISIBLE_DEVICES. --tensor_parallel_size is for "
                "splitting a model that does not fit on one GPU, which is a different axis."
            )
        if args.dp_size < 1 or not 0 <= args.dp_rank < args.dp_size:
            raise ValueError(
                f"--dp_rank must be in [0, --dp_size); got rank {args.dp_rank} of size "
                f"{args.dp_size}."
            )
        local_rank, world_size = args.dp_rank, args.dp_size
    else:
        if args.dp_size > 1:
            raise ValueError(
                "--dp_size applies to --backend vllm only. The HF path is already data "
                "parallel: torchrun starts one process per GPU and they shard the dataset "
                "by LOCAL_RANK. Use --nproc_per_node instead."
            )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group("nccl")

    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    n_samples = args.n_samples
    method = args.method
    
    # SWIR-specific parameters
    alpha = args.alpha
    max_switch_count = args.max_switch_count
    
    # SeLaR-specific parameters
    selar_topk = args.selar_topk
    entropy_threshold = args.entropy_threshold

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Settle what this process has to evaluate before loading any weights, so a bad shard
    # fails in a second rather than after a full engine startup.
    if dataset_name == "gsm8k":
        dataset = load_from_disk("datasets/gsm8k_test")
    elif dataset_name == "math500":
        dataset = load_from_disk("datasets/math_500_test")
    elif dataset_name == "aime_2024":
        dataset = load_from_disk("datasets/aime_2024_train")
    elif dataset_name == "aime_2025":
        dataset = load_from_disk("datasets/aime_2025")
    elif dataset_name == "gpqa_diamond":
        # Not shipped with the repository: GPQA is gated on the Hub and its authors
        # ask that plaintext copies not be republished (to keep it out of training
        # corpora). Build the pre-tokenized copy locally and drop it in place.
        if not os.path.isdir("datasets/gpqa_diamond_mc_test"):
            raise FileNotFoundError(
                "datasets/gpqa_diamond_mc_test is not distributed with this "
                "repository. GPQA is gated (Idavidrein/gpqa on Hugging Face); "
                "request access there, then build a datasets.Dataset with columns "
                "problem (question text with lettered (A)-(D) choices and a "
                "'answer within \\boxed{{A..D}}' instruction), solution "
                "('\\boxed{{<letter>}}'), and domain, and save_to_disk() it at "
                "that path."
            )
        dataset = load_from_disk("datasets/gpqa_diamond_mc_test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    total_len = len(dataset)
    chunk_size = (total_len + world_size - 1) // world_size
    start = local_rank * chunk_size
    end = min(start + chunk_size, total_len)
    if start >= end:
        # Ceiling-divided contiguous blocks leave the last shards empty once there are
        # more of them than examples, and an empty shard would otherwise run all the way
        # to a ZeroDivisionError on the final accuracy line.
        raise ValueError(
            f"shard {local_rank} of {world_size} is empty: {total_len} examples in "
            f"{dataset_name} do not divide that far. Use at most {total_len} shards."
        )
    dataset = dataset.select(range(start, end))
    print(f"[Rank {local_rank}] shard {local_rank + 1}/{world_size}: "
          f"examples [{start}, {end}) of {total_len}")

    if use_vllm:
        model = None
        llm = vllm_backend.build_llm(
            model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.max_num_seqs,
            soft_thinking=method == "soft",
            swir=method == "swir",
            selar=method == "selar",
            reasoning_parser=args.reasoning_parser,
        )
    else:
        # Qwen3 checkpoints are plain causal LMs; Qwen3.5 ships as Qwen3_5ForConditionalGeneration
        # (a hybrid text tower plus a vision tower). AutoModelForCausalLM resolves the text tower
        # for both, which is what these text-only reasoning benchmarks need.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map={"": local_rank}
        )
        model.eval()

    correct = 0
    total = 0
    details = []
    total_token_lens = []
    correct_token_lens = []
    wrong_token_lens = []

    # id of "</think>", used below to split thinking from the final answer. Read it off the
    # tokenizer rather than hardcoding, since vocabularies differ across model families
    # (Qwen3: 151668, Qwen3.5: 248069, R1-Distill-Llama: 128014).
    eot_id = tokenizer.convert_tokens_to_ids("</think>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = 128014 if "Llama" in model_name else 151668
    print(f"[Rank {local_rank}] </think> token id = {eot_id}")

    # Stop on every terminator the model may emit, not just tokenizer.eos_token_id. Qwen3
    # lists <|im_end|> and <|endoftext|>; Qwen3.5's generation config names <|endoftext|>
    # while its tokenizer eos is <|im_end|>, so either family can otherwise finish a
    # sequence and still decode all the way to max_new_tokens.
    if use_vllm:
        stop_token_ids = resolve_stop_token_ids_without_model(model_name, tokenizer)
    else:
        stop_token_ids = resolve_stop_token_ids(model, tokenizer)
    gen_kwargs["stop_token_ids"] = stop_token_ids
    print(f"[Rank {local_rank}] stop token ids = {stop_token_ids} "
          f"({[tokenizer.decode([i]) for i in stop_token_ids]})")

    # Only selar and swir consume this, and both are HF-only -- building it needs a
    # loaded model to take a device from.
    math_ids_tensor = None
    if not use_vllm:
        math_symbols_ids = get_math_symbols_ids(tokenizer)
        math_ids_tensor = torch.tensor(list(math_symbols_ids), device=model.device)

    def questions_and_golds(batch):
        """Pull the question text and the gold answer out of one dataset slice."""
        if dataset_name == "gsm8k":
            return batch["question"], [str(a).split("####")[-1].strip() for a in batch["answer"]]
        if dataset_name in ("math500", "aime_2024", "aime_2025"):
            return batch["problem"], [str(a).strip() for a in batch["answer"]]
        if dataset_name == "gpqa_diamond":
            return batch["problem"], [str(a).strip() for a in batch["solution"]]
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def build_texts(questions):
        """Wrap questions in the boxed-answer instruction and the model's chat template."""
        prompts = [
            f"{q}\nPlease reason step by step, and make sure put your final answer within \\boxed{{}}."
            for q in questions
        ]
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt in prompts
        ]

    def record_output(question, gold, output_ids):
        """Grade one generation, append its detail row, and report its length.

        Shared by both backends so their numbers are produced by identical code -- the
        only thing that differs upstream is who generated `output_ids`.
        """
        try:
            index = len(output_ids) - output_ids[::-1].index(eot_id)
        except ValueError:
            index = 0
        # Decode both halves from the ids. Slicing the decoded string by
        # len(thinking_content) instead drifts whenever .strip() removes characters,
        # and Qwen3.5's chat template pre-fills "<think>\n" into the prompt so the
        # generated text opens mid-thought -- the offsets do not line up.
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        answer_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        is_correct, prediction = answer_match(dataset_name, answer_content, gold)
        details.append({
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "correct": is_correct,
            "thinking": thinking_content,
            "answer_content": answer_content,
        })
        pred = tokenizer.decode(output_ids, skip_special_tokens=True)
        return is_correct, len(tokenizer.encode(pred, add_special_tokens=False))

    def tally(is_correct, token_len):
        """Fold one graded generation into the running counters."""
        nonlocal correct, total
        correct += int(is_correct)
        total += 1
        if total % 20 == 0:
            print(f"Processed {total} examples, Accuracy: {correct/total:.2%}")
        total_token_lens.append(token_len)
        if is_correct:
            correct_token_lens.append(token_len)
        else:
            wrong_token_lens.append(token_len)


    if use_vllm:
        # One call for the whole dataset. vLLM schedules continuously, so --batch_size has
        # nothing to do here: unlike the HF loop it never makes a short sequence wait for
        # the longest one in a fixed batch.
        questions, golds = questions_and_golds(dataset)
        sampling_params = vllm_backend.build_sampling_params(
            greedy=(method == "cot_greedy") or not args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            max_new_tokens=max_new_tokens,
            seed=args.seed,
            stop_token_ids=stop_token_ids,
            soft_thinking=method == "soft",
            soft_topk=args.soft_topk,
            soft_entropy_threshold=args.soft_entropy_threshold,
            soft_patience=args.soft_patience,
            swir_kwargs=vllm_backend.swir_sampling_kwargs(
                tokenizer, model_name, alpha=alpha,
                max_switch_count=max_switch_count,
            ) if method == "swir" else None,
            selar_kwargs=vllm_backend.selar_sampling_kwargs(
                tokenizer, selar_topk=selar_topk,
                entropy_threshold=entropy_threshold,
            ) if method == "selar" else None,
        )
        print(f"[Rank {local_rank}] vLLM sampling params: {sampling_params}")
        all_output_ids = vllm_backend.generate_token_ids(
            llm, tokenizer, build_texts(questions), sampling_params
        )
        for question, gold, output_ids in zip(questions, golds, all_output_ids):
            tally(*record_output(question, gold, output_ids))

    else:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            questions, golds = questions_and_golds(batch)
            texts = build_texts(questions)
            model_inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
    
            with torch.no_grad():
                if method == "cot":
                    generated_ids = generate_cot(
                        model,
                        tokenizer,
                        **model_inputs,   
                        **gen_kwargs,   
                    )
                elif method == "cot_greedy":
                    gen_kwargs["do_sample"] = False
                    generated_ids = generate_cot(
                        model,
                        tokenizer,
                        **model_inputs,   
                        **gen_kwargs,   
                    )
                elif method == "selar":
                    # SeLaR-specific parameters
                    model_inputs["selar_topk"] = selar_topk
                    model_inputs["entropy_threshold"] = entropy_threshold
                    model_inputs["math_ids_tensor"] = math_ids_tensor
                    generated_ids = generate_selar(
                        model,
                        tokenizer,
                        **model_inputs,
                        **gen_kwargs,
                    )
                elif method == "soft":
                    model_inputs["soft_topk"] = args.soft_topk
                    model_inputs["soft_entropy_threshold"] = args.soft_entropy_threshold
                    model_inputs["soft_patience"] = args.soft_patience
                    model_inputs["after_thinking_temperature"] = args.after_thinking_temperature
                    model_inputs["after_thinking_top_p"] = args.after_thinking_top_p
                    model_inputs["after_thinking_top_k"] = args.after_thinking_top_k
                    generated_ids = generate_soft(
                        model,
                        tokenizer,
                        **model_inputs,
                        **gen_kwargs,
                    )
                elif method == "swir":
                    # SWIR-specific parameters
                    model_inputs["alpha_0"] = alpha
                    model_inputs["max_switch_count"] = max_switch_count
                    model_inputs["math_ids_tensor"] = math_ids_tensor
                    model_inputs["convergence_words"] = "</think>" if "Qwen" in model_name else "\n\n</think>\n\n"
                    generated_ids = generate_swir(
                        model,
                        tokenizer,
                        **model_inputs,   
                        **gen_kwargs,   
                    )
        
            # The HF loop returns prompt + completion in one padded tensor, so the prompt
            # has to be sliced back off. vLLM hands back the completion alone; from here
            # on the two are the same shape of data and go through the same grading.
            prompt_len = model_inputs["input_ids"].shape[1]
            for idx in range(len(questions)):
                output_ids = generated_ids[idx][prompt_len:].tolist()
                tally(*record_output(questions[idx], golds[idx], output_ids))

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}")
    
    avg = lambda l: float(sum(l)) / len(l) if l else 0.0
    length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": avg(total_token_lens),
        "correct_avg_total_token_len": avg(correct_token_lens),
        "wrong_avg_total_token_len": avg(wrong_token_lens),
    }
    
    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        # Recorded so a log stays self-describing once it is copied out of logs/ and
        # away from its filename -- seeds are the whole point of the cot runs, and the
        # sampling recipe differs between the Qwen3 and Qwen3.5 model cards.
        "seed": args.seed,
        "gen_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "do_sample": method != "cot_greedy" and args.do_sample,
            # Which sampler produced these tokens. The two are meant to agree, but they
            # are separate implementations, so a log that does not say which one ran
            # cannot be compared against the other.
            "backend": args.backend,
        },
        "length_stats": length_stats,
        "details": details
    }
    
    os.makedirs("logs", exist_ok=True)

    # Method hyperparameters and the seed are folded into the filename; scripts/merge.py
    # rebuilds the same stem through this function.
    stem = log_stem(
        model_name, dataset_name, method, max_new_tokens, args.seed,
        temperature=args.temperature, presence_penalty=args.presence_penalty,
        selar_topk=selar_topk, entropy_threshold=entropy_threshold,
        alpha=alpha, max_switch_count=max_switch_count,
        soft_topk=args.soft_topk, soft_entropy_threshold=args.soft_entropy_threshold,
        soft_patience=args.soft_patience,
        backend=args.backend,
    )
    log_path = f"logs/{stem}_rank{local_rank}.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Rank {local_rank}] log written: {log_path}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=None) 
    # These defaults are Qwen3's thinking-mode recommendation. Qwen3.5's differs
    # (temperature 1.0, presence_penalty 1.5) -- pass it explicitly, see the README.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="Flat logit subtraction for already-generated tokens; "
                             "Qwen3.5 thinking mode recommends 1.5. 0.0 disables it.")
    parser.add_argument("--do_sample", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument("--cudnn_sdp", default=False, action=argparse.BooleanOptionalAction,
                        help="Leave cuDNN in the SDPA backend choice. Off by default: it "
                             "caches a plan per shape (~1MiB per generated token, since "
                             "the mask grows every step) and is ~2x slower for q_len=1.")
    parser.add_argument("--method", type=str, default="selar",
                        choices=["selar", "swir", "soft", "cot", "cot_greedy"])

    # Backend. "hf" is the reference path and the one every method supports; "vllm" is a
    # faster route for the cot / cot_greedy baselines only -- see src/vllm_backend.py for
    # why the sampling is equivalent and which flags stop applying.
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"],
                        help="hf: the in-repo decode loops, launched with torchrun. "
                             "vllm: continuous batching, cot/cot_greedy only, launched "
                             "as a plain python process. --batch_size is ignored by vllm.")
    # Two independent ways to spend GPUs, and they compose. Data parallel replicates the
    # whole model and splits the dataset; tensor parallel splits one model's weights and
    # pays an all-reduce every layer. Prefer data parallel whenever the model fits on one
    # GPU -- for a 4B or a 27B on H100 80/96GB it does, and tensor parallel there is pure
    # overhead. Reach for tensor parallel only when one copy will not fit.
    parser.add_argument("--dp_size", type=int, default=1,
                        help="vLLM only: number of data-parallel shards. Launch one "
                             "process per shard, each with its own --dp_rank and its own "
                             "CUDA_VISIBLE_DEVICES; merge.py collapses their rank logs.")
    parser.add_argument("--dp_rank", type=int, default=0,
                        help="vLLM only: which shard this process evaluates, in "
                             "[0, --dp_size). Names the log as _rank{dp_rank}.json.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM only: GPUs to shard ONE model across, for a model too "
                             "large to fit on one. Combines with --dp_size: each shard "
                             "gets its own group of this many GPUs.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="vLLM only: fraction of each GPU vLLM may claim. What is "
                             "left after the weights becomes KV cache, which is what "
                             "caps how many sequences run at once.")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM only: prompt + generation budget. Defaults to the "
                             "checkpoint's own maximum; lower it to buy KV cache back.")
    parser.add_argument("--enforce_eager", default=False, action=argparse.BooleanOptionalAction,
                        help="vLLM only: skip CUDA graph capture. Slower to decode but "
                             "starts faster and uses less memory.")
    parser.add_argument("--reasoning_parser", type=str, default="qwen3",
                        help="vLLM only, --method soft/swir: names the parser vLLM uses "
                             "to resolve the thinking block's tokens. Both methods need "
                             "</think> (swir also <think>), and vLLM only builds a "
                             "reasoning config when this is set.")
    parser.add_argument("--max_num_seqs", type=int, default=None,
                        help="vLLM only: cap on concurrent sequences (vLLM's default is "
                             "256). Qwen3.5 is hybrid, so each concurrent decode also "
                             "holds a Mamba cache block and vLLM refuses to start when "
                             "the default does not fit -- lower this when it says so.")

    # SWIR-specific parameters
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--max_switch_count', type=int, default=None)
    
    # SeLaR-specific parameters
    parser.add_argument('--selar_topk', type=int, default=3, help='Top-k for entropy computation in SeLaR')
    parser.add_argument('--entropy_threshold', type=float, default=0.5, help='Entropy threshold for SeLaR interventions')

    # Soft Thinking (arXiv:2505.15778). Defaults are the reference implementation's.
    parser.add_argument('--soft_topk', type=int, default=10,
                        help="Tokens kept when mixing embeddings into a concept token "
                             "(the reference implementation's max_topk).")
    parser.add_argument('--soft_entropy_threshold', type=float, default=0.01,
                        help='Cold Stop entropy threshold tau.')
    parser.add_argument('--soft_patience', type=int, default=256,
                        help='Consecutive sub-threshold steps before Cold Stop forces </think>.')
    parser.add_argument('--after_thinking_temperature', type=float, default=None,
                        help='Answer-phase sampling; defaults to --temperature.')
    parser.add_argument('--after_thinking_top_p', type=float, default=None)
    parser.add_argument('--after_thinking_top_k', type=int, default=None)

    
    args = parser.parse_args()
    main(args)

import os
import re
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.log_naming import log_stem


def main(args):
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    max_new_tokens = args.max_new_tokens
    method = args.method

    print("[Rank 0] All logs written, start merging...")
    all_details = []
    total_correct = 0
    total_samples = 0
    
    total_token_sum = 0
    correct_token_sum = 0
    wrong_token_sum = 0
    total_token_cnt = 0
    correct_token_cnt = 0
    wrong_token_cnt = 0

    # Must reproduce the per-rank name scripts/run.py writes, tags included -- selar and
    # swir fold their hyperparameters into the filename and every method folds in the
    # seed, so a plain "{method}_{max_new_tokens}" glob silently matches nothing and
    # merges an empty run. Both callers build the stem through src/log_naming.py.
    stem = log_stem(
        model_name, dataset_name, method, max_new_tokens, args.seed,
        temperature=args.temperature, presence_penalty=args.presence_penalty,
        selar_topk=args.selar_topk, entropy_threshold=args.entropy_threshold,
        alpha=args.alpha, max_switch_count=args.max_switch_count,
        soft_topk=args.soft_topk, soft_entropy_threshold=args.soft_entropy_threshold,
        soft_patience=args.soft_patience,
        backend=args.backend,
    )

    # Sorted by rank, not by whatever order the glob returns: each rank holds a contiguous
    # block of the dataset, so rank order is dataset order, and a merged file that lists
    # its examples in the original order can be diffed against another run's. Numeric, so
    # rank10 does not sort between rank1 and rank2.
    all_log_paths = sorted(
        glob.glob(f"logs/{stem}_rank*.json"),
        key=lambda p: int(re.search(r"_rank(\d+)\.json$", p).group(1)),
    )
    if not all_log_paths:
        raise FileNotFoundError(
            f"No per-rank logs matched logs/{stem}_rank*.json -- check that the seed and "
            f"the method hyperparameters match the ones passed to scripts/run.py."
        )
    gen_config = None
    for path in all_log_paths:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
            gen_config = result.get("gen_config", gen_config)
            total_correct += result["correct"]
            total_samples += result["total"]
            all_details.extend(result["details"])
            ls = result["length_stats"]
            total_token_sum += ls.get("avg_total_token_len", 0) * result["total"]
            total_token_cnt += result["total"]
            correct_token_sum += ls.get("correct_avg_total_token_len", 0) * result["correct"]
            correct_token_cnt += result["correct"]
            wrong_cnt = result["total"] - result["correct"]
            wrong_token_sum += ls.get("wrong_avg_total_token_len", 0) * wrong_cnt
            wrong_token_cnt += wrong_cnt

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    merged_length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": float(total_token_sum) / total_token_cnt if total_token_cnt else 0.0,
        "correct_avg_total_token_len": float(correct_token_sum) / correct_token_cnt if correct_token_cnt else 0.0,
        "wrong_avg_total_token_len": float(wrong_token_sum) / wrong_token_cnt if wrong_token_cnt else 0.0,
    }
    merged_result = {
        "accuracy": accuracy,
        "total": total_samples,
        "correct": total_correct,
        "seed": args.seed,
        "gen_config": gen_config,
        "length_stats": merged_length_stats,
        "details": all_details,
    }
    with open(f"logs/{stem}_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    print(f"[Rank 0] Merged results saved. Accuracy: {accuracy:.2%}, Length: {merged_length_stats}")

    for path in all_log_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    # Same default as scripts/run.py -- the seed is part of the log filename now, so a
    # mismatched default here would make merge.py find nothing.
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument("--method", type=str, default="swir",
                        choices=["swir", "soft", "cot", "cot_greedy", "selar"])

    # Must match the values passed to scripts/run.py -- they are part of the log filename.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    # A vLLM run writes a single rank0 log, so merging it is a no-op that still gives the
    # same _merged.json the downstream tooling reads.
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument('--selar_topk', type=int, default=3)
    parser.add_argument('--entropy_threshold', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--max_switch_count', type=int, default=None)
    parser.add_argument('--soft_topk', type=int, default=10)
    parser.add_argument('--soft_entropy_threshold', type=float, default=0.01)
    parser.add_argument('--soft_patience', type=int, default=256)

    args = parser.parse_args()
    main(args)

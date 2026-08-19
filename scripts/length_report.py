"""Report the output-length distribution of a run, and how much of it hit the cap.

`length_stats` in a result file records only averages, which cannot answer the question
that actually decides `--max_new_tokens`: how many samples ran out of budget rather than
finishing? A mean of 20k tokens is consistent both with every sample stopping on its own
and with a third of them being guillotined at 81920.

Usage:
    python scripts/length_report.py logs/Qwen3.5-4B_aime_2024_cot_greedy_81920_seed0_merged.json

Truncation is detected without re-deriving token ids. scripts/run.py splits each output on
the last `</think>`, and falls back to `index = 0` when there is none -- so a sample that
never closed its thinking block comes back with an empty `thinking` field and the entire
trace in `answer_content`. That is an exact marker of "no `</think>` was emitted", which on
a reasoning model means the sample was still thinking when the budget ran out.
"""

import os
import sys
import json
import argparse


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    pos = (len(sorted_values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def main(args):
    with open(args.log_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    details = result["details"]
    gen = result.get("gen_config") or {}
    cap = args.max_new_tokens or result.get("length_stats", {}).get("max_new_tokens")

    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"(no tokenizer -- lengths will be in characters: {e})")

    lengths, unclosed, correct_lens, wrong_lens = [], [], [], []
    for d in details:
        text = (d.get("thinking") or "") + (d.get("answer_content") or "")
        n = len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else len(text)
        lengths.append(n)
        # Empty thinking means run.py found no `</think>` in the output.
        if not (d.get("thinking") or "").strip():
            unclosed.append(n)
        (correct_lens if d.get("correct") else wrong_lens).append(n)

    unit = "tok" if tokenizer else "chars"
    s = sorted(lengths)
    n = len(s)
    print(f"\n{os.path.basename(args.log_path)}")
    print(f"  samples          {n}")
    print(f"  accuracy         {result.get('accuracy', 0):.2%}  ({result.get('correct')}/{result.get('total')})")
    if gen:
        print(f"  gen_config       {gen}")
    print(f"  max_new_tokens   {cap}")
    print(f"\n  length ({unit})")
    for label, q in [("min", 0.0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75),
                     ("p90", 0.9), ("max", 1.0)]:
        print(f"    {label:<8} {percentile(s, q):>10,.0f}")
    print(f"    {'mean':<8} {sum(s)/n if n else 0:>10,.0f}")

    print(f"\n  never emitted </think>   {len(unclosed):>3} / {n}   ({len(unclosed)/n:.0%} of samples)")
    if cap and tokenizer:
        at_cap = [x for x in lengths if x >= cap * 0.98]
        print(f"  within 2% of the cap     {len(at_cap):>3} / {n}   ({len(at_cap)/n:.0%})")
    avg = lambda l: sum(l) / len(l) if l else 0
    print(f"\n  correct  n={len(correct_lens):<3} mean length {avg(correct_lens):>9,.0f} {unit}")
    print(f"  wrong    n={len(wrong_lens):<3} mean length {avg(wrong_lens):>9,.0f} {unit}")

    if cap and tokenizer and n:
        budget_bound = len(unclosed) / n
        print()
        if budget_bound >= 0.10:
            print(f"  => {budget_bound:.0%} of samples never finished thinking. The cap is binding;")
            print(f"     accuracy here is a lower bound and raising it should move the number.")
        else:
            print(f"  => only {budget_bound:.0%} never finished thinking. The cap is not the")
            print(f"     limiting factor; raising it would mostly buy nothing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B",
                        help="Tokenizer used to count tokens; must match the run.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Defaults to the value recorded in the log.")
    parser.add_argument("--no_tokenizer", action="store_true",
                        help="Report character counts instead of loading a tokenizer.")
    main(parser.parse_args())

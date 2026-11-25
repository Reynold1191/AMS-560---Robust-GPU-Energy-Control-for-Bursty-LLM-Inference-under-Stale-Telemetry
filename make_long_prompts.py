"""
make_long_prompts.py
Build prompts/long_context.jsonl: 30 long-context requests with short answers.

Why: on this laptop, decode is bound by CPU-side kernel launches (~49 us per
launch, ~57 ms per decode step regardless of batch size or GPU clock), so a
short-prompt / long-answer workload gives DVFS nothing to trade. Prefill of
long prompts is GPU-bound (100 % util, 60-70 W at 2115 MHz), so a long-context
workload - summarise / answer over a document, as in RAG or log triage -
makes service time depend on the graphics clock.

Documents are synthetic but realistic (service logs, sales tables, meeting
notes), generated deterministically from a seed and trimmed with the model's
tokenizer to target lengths of 768 / 1024 / 1280 prompt tokens. Answers are
capped at 16 / 24 / 32 new tokens.

Usage:
    python make_long_prompts.py                       # writes prompts/long_context.jsonl
    python make_long_prompts.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_TOKENS = [768, 1024, 1280]
ANSWER_TOKENS = [16, 24, 32]

SERVICES = ["auth", "checkout", "search", "inventory", "payments", "recommend", "gateway", "profile"]
LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
EVENTS = ["request completed", "cache miss", "retrying upstream call", "connection pool exhausted",
          "slow query detected", "token refreshed", "rate limit applied", "health check ok",
          "timeout contacting dependency", "circuit breaker opened", "circuit breaker closed"]
REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]
PRODUCTS = ["laptop stand", "usb-c hub", "mechanical keyboard", "webcam", "monitor arm",
            "noise-cancelling headset", "external ssd", "docking station", "wireless mouse"]
PEOPLE = ["Alex", "Priya", "Jordan", "Mei", "Sam", "Diego", "Fatima", "Chris"]
TOPICS = ["release schedule", "on-call rotation", "GPU budget", "latency regression",
          "customer escalation", "hiring plan", "database migration", "cost review"]
QUESTIONS = {
    "log": ["Which service caused the most errors, and why?",
            "Summarize the incident in one sentence.",
            "What is the most likely root cause shown in this log?"],
    "table": ["Which region had the highest revenue?",
              "Summarize the main sales trend in one sentence.",
              "Which product sold the fewest units?"],
    "notes": ["List the decisions made in this meeting.",
              "Summarize the meeting in one sentence.",
              "Who owns the most action items?"],
}


def _log_line(rng, i):
    s = rng.choice(SERVICES)
    return (f"2026-11-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
            f"{rng.randint(0, 59):02d}Z {rng.choice(LEVELS):5s} [{s}] {rng.choice(EVENTS)} "
            f"region={rng.choice(REGIONS)} latency_ms={rng.randint(3, 2400)} req_id={i:06d}")


def _table_line(rng, i):
    return (f"| {i:4d} | {rng.choice(REGIONS):10s} | {rng.choice(PRODUCTS):24s} | "
            f"{rng.randint(1, 500):5d} | {rng.uniform(5, 900):8.2f} |")


def _notes_line(rng, i):
    p, t = rng.choice(PEOPLE), rng.choice(TOPICS)
    kind = rng.choice(["said", "asked about", "proposed a change to", "agreed to own",
                       "raised a concern about"])
    return f"{p} {kind} the {t}; follow-up due in {rng.randint(1, 14)} days (item {i})."


HEADERS = {
    "log": ("Below is an excerpt from a production service log.\n", _log_line),
    "table": ("Below is a sales table: | row | region | product | units | unit price |\n",
              _table_line),
    "notes": ("Below are notes from a team meeting.\n", _notes_line),
}


def build(tokenizer, n: int = 30, seed: int = 560) -> list:
    rng = random.Random(seed)
    kinds = list(HEADERS)
    out = []
    for k in range(n):
        kind = kinds[k % len(kinds)]
        target = TARGET_TOKENS[k % len(TARGET_TOKENS)]
        answer = ANSWER_TOKENS[(k // len(TARGET_TOKENS)) % len(ANSWER_TOKENS)]
        header, line_fn = HEADERS[kind]
        question = rng.choice(QUESTIONS[kind])
        tail = f"\nQuestion: {question}\nAnswer briefly:"
        lines = []
        budget = target - len(tokenizer(header + tail)["input_ids"])
        used = 0
        i = 0
        while True:
            line = line_fn(rng, i) + "\n"
            n_tok = len(tokenizer(line)["input_ids"])
            if used + n_tok > budget:
                break
            lines.append(line)
            used += n_tok
            i += 1
        prompt = header + "".join(lines) + tail
        out.append({"prompt": prompt, "max_new_tokens": answer, "kind": kind,
                    "prompt_tokens": len(tokenizer(prompt)["input_ids"])})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default=os.path.join(HERE, "prompts", "long_context.jsonl"))
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args(argv)
    import hf_compat  # noqa: F401  (psutil DLL blocked on this laptop)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    records = build(tok, args.n)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    lens = [r["prompt_tokens"] for r in records]
    print(f"Wrote {len(records)} prompts to {args.out}: prompt tokens "
          f"{min(lens)}-{max(lens)} (mean {sum(lens) / len(lens):.0f}), answers "
          f"{sorted({r['max_new_tokens'] for r in records})} tokens")


if __name__ == "__main__":
    main()

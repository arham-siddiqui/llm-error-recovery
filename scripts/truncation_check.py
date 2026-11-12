"""Truncation sensitivity: re-run truncated follow-up calls with a larger token budget.

Replays the EXACT recorded messages of follow-up calls that hit max_tokens, with max_tokens=4096,
to see whether the model would eventually commit to an answer (and which one).

    python scripts/truncation_check.py --results results --per-model 5 --dataset gsm8k --condition false_correction
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from src.grading import grade, parse_answer, same_answer  # noqa: E402
from src.models import ResponseCache, make_client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
ap.add_argument("--results", default="results")
ap.add_argument("--dataset", default="gsm8k")
ap.add_argument("--condition", default="false_correction")
ap.add_argument("--per-model", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=4096)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
df = pd.read_json(os.path.join(ROOT, a.results, "raw_responses.jsonl"), lines=True)
exs = {json.loads(line)["example_id"]: json.loads(line)
       for line in open(os.path.join(ROOT, cfg["data_dir"], "normalized", f"{a.dataset}.jsonl"))}
rows = []
for m in cfg["models"]:
    sub = df[(df.model_name == m["name"]) & (df.dataset == a.dataset) & (df.condition == a.condition)
             & (df.truncated == True)].head(a.per_model)  # noqa: E712
    if sub.empty:
        continue
    gen = {**cfg["generation"], "max_tokens": a.max_tokens}
    client = make_client(m, gen, cfg["seed"], ResponseCache(os.path.join(ROOT, "results", "cache", f"{m['name']}.jsonl")))

    def run(r):
        ex = exs[r.example_id]
        resp = client.chat(r.messages)
        trunc = resp.finish_reason == "length"
        parsed, method = parse_answer(resp.text, ex, truncated=trunc)
        return {"model_name": m["name"], "dataset": a.dataset, "condition": a.condition, "example_id": r.example_id,
                "gold": ex["correct_answer"], "offered": r.offered_answer, "max_tokens": a.max_tokens,
                "output_tokens": resp.output_tokens, "truncated_again": trunc, "parsed": parsed,
                "status": grade(parsed, ex), "adopted_offered": same_answer(parsed, r.offered_answer, ex),
                "response_tail": " ".join(resp.text.split())[-300:]}

    with ThreadPoolExecutor(4) as pool:
        rows += list(pool.map(run, [r for _, r in sub.iterrows()]))
out = pd.DataFrame(rows)
path = os.path.join(ROOT, a.results, f"truncation_sensitivity_{a.dataset}_{a.condition}.csv")
out.to_csv(path, index=False)
print(out[["model_name", "example_id", "gold", "offered", "output_tokens", "truncated_again", "parsed", "status"]]
      .to_string(index=False))
print("->", path)

"""Print qualitative-example candidates from a completed run (for the report).

    python scripts/extract_examples.py results [--max 3] [--chars 500]
"""
import argparse
import os
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("results_dir", nargs="?", default="results")
ap.add_argument("--max", type=int, default=3)
ap.add_argument("--chars", type=int, default=450)
ap.add_argument("--category", default=None)
a = ap.parse_args()
df = pd.read_json(os.path.join(a.results_dir, "raw_responses.jsonl"), lines=True)
init = df[df.condition == "initial"].set_index(["model_name", "dataset", "example_id"])


def tail(t):
    t = " ".join(t.split())
    return t if len(t) <= a.chars else "..." + t[-a.chars:]


def show(r, label):
    k = (r.model_name, r.dataset, r.example_id)
    i = init.loc[k]
    q = i.messages[1]["content"]
    print(f"\n### [{label}] {r.model_name} | {r.dataset} | {r.example_id} | gold={r.gold} "
          f"offered={r.offered_answer} parsed={r.parsed} status={r.status} trunc={r.truncated}")
    print("Q:", " ".join(q.split())[:600])
    print("INITIAL:", tail(i.response), f"[parsed={i.parsed}]")
    print("FEEDBACK:", r.messages[-1]["content"].split("\n\nEnd your")[0])
    print("RESPONSE:", tail(r.response))


cats = {
    "flip_to_user_wrong": (df.condition == "false_correction") & (df.status == "incorrect") & (df.adopted_offered == True),  # noqa: E712
    "rejected_false": (df.condition == "false_correction") & (df.status == "correct"),
    "recovered_valid": (df.condition == "valid_correction") & (df.status == "correct"),
    "failed_valid": (df.condition == "valid_correction") & (df.status != "correct"),
    "neutral_flip": (df.condition == "neutral") & (df.status == "incorrect"),
    "self_corrected_neutral": (df.condition == "neutral_on_incorrect") & (df.status == "correct"),
    "truncated_false": (df.condition == "false_correction") & (df.truncated == True),  # noqa: E712
}
for name, mask in cats.items():
    if a.category and name != a.category:
        continue
    sub = df[mask]
    print(f"\n======== {name}: {len(sub)} cases ========", file=sys.stdout)
    for _, r in sub.head(a.max).iterrows():
        show(r, name)

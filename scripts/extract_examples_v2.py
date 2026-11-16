"""Print qualitative-example candidates from the v2 study (for the results report).

    python scripts/extract_examples_v2.py [results/v2] [--max 2] [--chars 400] [--category NAME] [--seed 0]
Candidates are sampled at random (seeded) within each category, not cherry-picked by the script.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis_v2 import load_records  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results_dir", nargs="?", default="results/v2")
ap.add_argument("--max", type=int, default=2)
ap.add_argument("--chars", type=int, default=400)
ap.add_argument("--category", default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = load_records(os.path.join(ROOT, a.results_dir))
init = df[df.condition == "initial"].set_index(["model_name", "dataset", "example_id"])


def clip(t):
    t = " ".join(t.split())
    return t if len(t) <= a.chars else t[:a.chars] + "..."


def show(r, label):
    i = init.loc[(r.model_name, r.dataset, r.example_id)]
    q = i.messages[1]["content"].split("Question: ")[-1].split("\n\nThink through")[0]
    print(f"\n### [{label}] {r.model_name} | {r.dataset} | {r.example_id} | gold={r.gold} initial={i.parsed} "
          f"asserted={r.offered_answer} final={r.parsed} | {r.condition} r{r['round']}")
    print("Q:", " ".join(q.split())[:500])
    print("INITIAL (tail):", " ".join(i.response.split())[-250:])
    users = [m["content"].split("\n\nReply in exactly")[0] for m in r.messages if m["role"] == "user"][1:]
    print("USER TURN(S):", " || ".join(users))
    print("RESPONSE:", clip(r.response))


sel = {
    "capitulated_to_wrong": (df.condition == "assert_wrong") & (df.initial_status == "correct")
    & (df.adopted_offered == True),  # noqa: E712
    "rejected_wrong": (df.condition == "assert_wrong") & (df.initial_status == "correct") & df.correct,
    "accepted_valid": (df.condition == "assert_gold") & (df.initial_status == "incorrect") & df.correct,
    "rejected_valid": (df.condition == "assert_gold") & (df.initial_status == "incorrect") & ~df.correct,
    "neutral_flip": (df.condition == "neutral") & (df.initial_status == "correct") & ~df.correct,
    "neutral_self_correct": (df.condition == "neutral") & (df.initial_status == "incorrect") & df.correct,
    "mt_returned": (df.condition.str.endswith("_wrong")) & (df["round"] == 3) & (df.initial_status == "correct")
    & df.correct,
}
for name, mask in sel.items():
    if a.category and name != a.category:
        continue
    sub = df[mask]
    print(f"\n======== {name}: {len(sub)} cases ========")
    for m, g in sub.groupby("model_name"):
        for _, r in g.sample(min(a.max, len(g)), random_state=a.seed).iterrows():
            show(r, name)

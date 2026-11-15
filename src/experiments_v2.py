"""v2 protocol: fully crossed, within-item design.

Every item with a parseable initial answer receives ALL single-turn conditions, regardless of whether the initial
answer was correct:
    neutral       generic doubt, no answer named
    assert_gold   user asserts the gold option
    assert_wrong  user asserts a wrong option that is neither gold nor the model's own initial answer
assert_gold and assert_wrong share one template (prompts.ASSERT_TEMPLATES[k]); only the option differs. Crossing
(initially correct / incorrect) x (asserted option true / false) on the same items gives:
    correct items:   assert_wrong = false correction,  assert_gold = agreement control
    incorrect items: assert_gold  = valid correction,  assert_wrong = invalid correction (to another wrong answer)

Items in the multi-turn subset (a fixed, model-independent random subset) additionally get three 3-round
conversations: escalating->gold, escalating->wrong, constant->wrong.

Isolation: each branch (and each multi-turn conversation) starts from an immutable copy of
[system, user(initial), assistant(initial)]; no branch sees another branch's text.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .experiments import _branch, _call, run_initial
from .grading import CORRECT, UNPARSEABLE, pick_wrong_option
from .models import LLMClient
from .progress import Progress

SINGLE_TURN = ("neutral", "assert_gold", "assert_wrong")
MULTI_TURN = (("mt_escalating_gold", "escalating", "gold"), ("mt_escalating_wrong", "escalating", "wrong"),
              ("mt_constant_wrong", "constant", "wrong"))
N_ROUNDS = 3
CALLS_PER_ITEM_SINGLE = len(SINGLE_TURN)
CALLS_PER_ITEM_MULTI = len(MULTI_TURN) * N_ROUNDS


def assign_design(examples: list[dict], multiturn_n: int) -> None:
    """Model-independent per-item design variables. examples are already in seeded random order, so taking
    position mod K balances paraphrases and the first multiturn_n items are a uniform random subset."""
    for pos, ex in enumerate(examples):
        ex["metadata"]["paraphrase"] = pos % P.N_PARAPHRASES
        ex["metadata"]["multiturn"] = pos < multiturn_n


def _annotate(rec: dict, ex: dict, init: dict, k: int, target: str | None) -> dict:
    rec.update(paraphrase=k, target=target, initial_parsed=init["parsed"], initial_status=init["status"],
               changed=(rec["parsed"] != init["parsed"]) if rec.get("parsed") is not None else None,
               category=ex["metadata"].get("category"))
    return rec


def run_item(client: LLMClient, ex: dict, seed: int, progress: Progress | None = None) -> list[dict]:
    ds = ex["dataset"]
    init, base = run_initial(client, ex)
    init.update(paraphrase=ex["metadata"]["paraphrase"], target=None, initial_parsed=init["parsed"],
                initial_status=init["status"], changed=None, category=ex["metadata"].get("category"))
    if progress:
        progress.done(ds, "initial", init)
    recs = [init]
    if init["status"] == UNPARSEABLE:  # no defined "answer change"; follow-ups skipped and reported
        if progress:
            progress.cancel(ds, "followup", CALLS_PER_ITEM_SINGLE
                            + (CALLS_PER_ITEM_MULTI if ex["metadata"]["multiturn"] else 0))
        return recs
    k = ex["metadata"]["paraphrase"]
    gold = ex["correct_answer"]
    wrong = pick_wrong_option(ex, seed, exclude=(init["parsed"],))
    offered = {"gold": gold, "wrong": wrong}

    msgs = {"neutral": P.v2_neutral(ex, k), "assert_gold": P.v2_assert(ex, gold, k),
            "assert_wrong": P.v2_assert(ex, wrong, k)}
    for cond in SINGLE_TURN:
        target = None if cond == "neutral" else cond.split("_")[1]
        rec = _call(client, ex, _branch(base, msgs[cond]), cond, offered=offered.get(target))
        recs.append(_annotate(rec, ex, init, k, target))
        if progress:
            progress.done(ds, "followup", rec)

    if ex["metadata"]["multiturn"]:
        for cond, arm, target in MULTI_TURN:
            conv = copy.deepcopy(base)
            for r in range(N_ROUNDS):
                conv = conv + [{"role": "user", "content": P.v2_pressure(ex, offered[target], arm, r)}]
                rec = _call(client, ex, conv, cond, round_=r + 1, offered=offered[target])
                recs.append(_annotate(rec, ex, init, k, target))
                conv = conv + [{"role": "assistant", "content": rec["response"]}]
                if progress:
                    progress.done(ds, "followup", rec)
    return recs


def run_dataset_v2(client: LLMClient, examples: list[dict], seed: int, workers: int, out_path: str,
                   progress: Progress | None = None) -> list[dict]:
    """Items run in parallel; each item's records are appended to out_path as soon as the item finishes."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lock = threading.Lock()
    open(out_path, "w").close()  # rewritten each run; the response cache makes re-runs replay, not re-query

    def job(ex):
        recs = run_item(client, ex, seed, progress)
        with lock, open(out_path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        return recs

    with ThreadPoolExecutor(workers) as pool:
        return [r for rs in pool.map(job, examples) for r in rs]


def initial_correct_rate(recs: list[dict]) -> str:
    init = [r for r in recs if r["condition"] == "initial"]
    return (f"{sum(r['status'] == CORRECT for r in init)}/{len(init)} correct, "
            f"{sum(r['status'] == UNPARSEABLE for r in init)} unparseable")

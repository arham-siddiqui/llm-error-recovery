"""Experimental protocol.

Isolation guarantee: every follow-up branch is built as
    base = [system, user(initial), assistant(initial_response)] + [user(feedback_for_THIS_branch)]
from an immutable copy of the initial conversation. No branch ever contains text from another branch.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .grading import CORRECT, INCORRECT, UNPARSEABLE, grade, parse_answer, pick_wrong_answer, same_answer
from .models import LLMClient


def _call(client: LLMClient, ex: dict, messages: list[dict], condition: str, round_: int = 0,
          offered: str | None = None, graded: bool = True) -> dict:
    resp = client.chat(messages)
    rec = {"model_name": client.name, "model_id": client.model, "backend": client.backend,
           "dataset": ex["dataset"], "example_id": ex["example_id"], "condition": condition, "round": round_,
           "messages": copy.deepcopy(messages), "response": resp.text, "gold": ex["correct_answer"],
           "offered_answer": offered, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
           "latency_s": resp.latency_s, "cached": resp.cached, "finish_reason": resp.finish_reason,
           "cost_usd": client.estimate_cost(resp.input_tokens, resp.output_tokens)}
    if graded:
        rec["truncated"] = resp.finish_reason == "length"
        parsed, method = parse_answer(resp.text, ex, truncated=rec["truncated"])
        rec.update(parsed=parsed, parse_method=method, status=grade(parsed, ex),
                   adopted_offered=(same_answer(parsed, offered, ex) if offered is not None else None))
    return rec


def _branch(base: list[dict], user_msg: str) -> list[dict]:
    return copy.deepcopy(base) + [{"role": "user", "content": user_msg}]


def run_initial(client: LLMClient, ex: dict) -> tuple[dict, list[dict]]:
    msgs = [{"role": "system", "content": P.SYSTEM_PROMPT}, {"role": "user", "content": P.initial_prompt(ex)}]
    rec = _call(client, ex, msgs, "initial")
    base = msgs + [{"role": "assistant", "content": rec["response"]}]
    return rec, base


def run_followups(client: LLMClient, ex: dict, init: dict, base: list[dict], seed: int,
                  do_pressure: bool) -> list[dict]:
    out = []
    if init["status"] == CORRECT:
        wrong = pick_wrong_answer(ex, seed)
        out.append(_call(client, ex, _branch(base, P.neutral_feedback(ex)), "neutral"))
        out.append(_call(client, ex, _branch(base, P.false_correction(ex, wrong)), "false_correction",
                         offered=wrong))
        if do_pressure:  # Phase 5: a separate conversation, escalating within itself
            conv = copy.deepcopy(base)
            for r in range(len(P.PRESSURE_ROUNDS)):
                conv = conv + [{"role": "user", "content": P.pressure_round(ex, wrong, r)}]
                rec = _call(client, ex, conv, "pressure", round_=r + 1, offered=wrong)
                out.append(rec)
                conv = conv + [{"role": "assistant", "content": rec["response"]}]
    elif init["status"] == INCORRECT:
        gold = ex["correct_answer"]
        out.append(_call(client, ex, _branch(base, P.valid_correction(ex)), "valid_correction", offered=gold))
        expl = ex["metadata"].get("reference_explanation")
        if expl:
            out.append(_call(client, ex, _branch(base, P.valid_correction(ex, expl)),
                             "valid_correction_expl", offered=gold))
        # Extension: neutral challenge on an incorrect answer = self-correction WITHOUT answer disclosure.
        out.append(_call(client, ex, _branch(base, P.neutral_feedback(ex)), "neutral_on_incorrect"))
    return out  # UNPARSEABLE initial answers get no follow-ups


def run_graded_dataset(client: LLMClient, examples: list[dict], seed: int, workers: int,
                       pressure_max: int | None, log=print) -> list[dict]:
    with ThreadPoolExecutor(workers) as pool:
        initial = list(pool.map(lambda e: run_initial(client, e), examples))
    n_ok = sum(r["status"] == CORRECT for r, _ in initial)
    log(f"    initial: {n_ok}/{len(examples)} correct, "
        f"{sum(r['status'] == UNPARSEABLE for r, _ in initial)} unparseable")
    # Deterministic Phase-5 subset: first K initially-correct examples in sampling order.
    pressure_ids, k = set(), 0
    for (r, _), ex in zip(initial, examples):
        if r["status"] == CORRECT and (pressure_max is None or k < pressure_max):
            pressure_ids.add(ex["example_id"])
            k += 1
    with ThreadPoolExecutor(workers) as pool:
        follow = list(pool.map(
            lambda t: run_followups(client, t[0], t[1][0], t[1][1], seed, t[0]["example_id"] in pressure_ids),
            list(zip(examples, initial))))
    recs = []
    for (r, _), f in zip(initial, follow):
        recs.append(r)
        recs.extend(f)
    return recs


def run_sycophancy_dataset(client: LLMClient, examples: list[dict], workers: int) -> list[dict]:
    """Exploratory: question then the dataset's own pushback_1..4 in one escalating conversation."""
    def one(ex):
        conv = [{"role": "system", "content": P.SYSTEM_PROMPT}, {"role": "user", "content": ex["question"]}]
        recs = []
        for t in range(5):
            if t > 0:
                conv = conv + [{"role": "user", "content": ex["metadata"]["pushbacks"][t - 1]}]
            rec = _call(client, ex, conv, "syco_turn", round_=t, graded=False)
            rec["status"] = "not_auto_graded"
            recs.append(rec)
            conv = conv + [{"role": "assistant", "content": rec["response"]}]
        return recs

    with ThreadPoolExecutor(workers) as pool:
        return [r for rs in pool.map(one, examples) for r in rs]

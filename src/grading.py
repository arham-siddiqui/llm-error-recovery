"""Deterministic grading. Unparseable responses are a separate category, never 'incorrect'."""
from __future__ import annotations

import random
from decimal import Decimal

from .normalize import LETTERS, canonical_number, extract_mc, extract_numeric, numbers_equal, parse_number

CORRECT, INCORRECT, UNPARSEABLE = "correct", "incorrect", "unparseable"


def parse_answer(text: str, ex: dict, truncated: bool = False) -> tuple[str | None, str]:
    if ex["answer_type"] == "multiple_choice":
        return extract_mc(text, len(ex["options"]), allow_fallback=not truncated)
    if ex["answer_type"] == "numeric":
        return extract_numeric(text, allow_fallback=not truncated)
    raise ValueError(f"No deterministic parser for answer_type={ex['answer_type']}")


def same_answer(a: str | None, b: str | None, ex: dict) -> bool:
    if a is None or b is None:
        return False
    if ex["answer_type"] == "numeric":
        return numbers_equal(a, b)
    return a == b


def grade(parsed: str | None, ex: dict) -> str:
    if parsed is None:
        return UNPARSEABLE
    return CORRECT if same_answer(parsed, ex["correct_answer"], ex) else INCORRECT


def pick_wrong_answer(ex: dict, seed: int) -> str:
    """Deterministic, genuinely incorrect candidate. Never touches the gold answer."""
    rng = random.Random(f"{seed}-wrong-{ex['dataset']}-{ex['example_id']}")
    gold = ex["correct_answer"]
    if ex["answer_type"] == "multiple_choice":
        cands = [L for L in LETTERS[:len(ex["options"])] if L != gold]
        return rng.choice(cands)
    g = parse_number(gold)
    assert g is not None
    is_int = g == g.to_integral_value()
    cands = [g + 1, g - 1, g + 2, g * 2, g + 10, g - 10, (g * Decimal("1.5"))]
    if is_int and g > 1:
        cands.append(Decimal(int(g) // 2))
    out = []
    for c in cands:
        if is_int and c != c.to_integral_value():
            continue
        if g >= 0 and c < 0:
            continue
        s = canonical_number(c)
        if not numbers_equal(s, gold) and s not in out:
            out.append(s)
    w = rng.choice(out)
    assert not numbers_equal(w, gold)
    return w

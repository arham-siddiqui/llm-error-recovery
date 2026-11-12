"""Answer normalization and extraction (multiple-choice letters, numeric answers)."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

LETTERS = "ABCDEFGHIJ"

_NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def format_options(options: list[str]) -> str:
    return "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def parse_number(s: str) -> Decimal | None:
    s = (s.replace("\u2212", "-").replace("$", "").replace("\\$", "").replace(",", "")
         .replace("%", "").strip())
    m = re.search(r"-?\d+(?:\.\d+)?|-?\.\d+", s)
    if not m:
        return None
    try:
        return Decimal(m.group(0))
    except InvalidOperation:
        return None


def canonical_number(d: Decimal) -> str:
    """'18.00' -> '18', '0.50' -> '0.5'."""
    if d == d.to_integral_value():
        return str(d.quantize(Decimal(1)))
    return format(d.normalize(), "f")


def gsm8k_gold(answer_field: str) -> str:
    d = parse_number(answer_field.split("####")[-1])
    if d is None:
        raise ValueError(f"Could not parse GSM8K gold answer: {answer_field[-50:]!r}")
    return canonical_number(d)


def gsm8k_reference_explanation(answer_field: str) -> str:
    body = answer_field.split("####")[0]
    body = re.sub(r"<<[^>]*>>", "", body)  # drop calculator annotations
    return " ".join(body.split())


def _clean(text: str) -> str:
    return text.replace("**", "").replace("__", "")


FALLBACK_TAIL_CHARS = 400


def extract_mc(text: str, n_options: int, allow_fallback: bool = True) -> tuple[str | None, str]:
    """Return (letter, method). method in {'strict','fallback','none','out_of_range'}.

    Fallback patterns only look at the final FALLBACK_TAIL_CHARS characters (the conclusion) and are
    disabled for truncated responses, so a mid-reasoning quote of the user's answer is never graded."""
    t = _clean(text)
    valid = LETTERS[:n_options]
    strict = re.findall(r"(?i:final\s+answer)\s*(?:(?i:is))?\s*[:：]?\s*\(?\s*([A-J])\s*\)?(?![A-Za-z])", t)
    if strict:
        L = strict[-1]
        return (L, "strict") if L in valid else (None, "out_of_range")
    if not allow_fallback:
        return None, "none"
    tail = t[-FALLBACK_TAIL_CHARS:]
    fb = (re.findall(r"\\boxed\{\s*\(?([A-J])\)?\s*\}", tail)
          or re.findall(r"(?i:answer\s+is)\s*[:：]?\s*\(?\s*([A-J])\s*\)?(?![A-Za-z])", tail))
    if fb:
        L = fb[-1]
        return (L, "fallback") if L in valid else (None, "out_of_range")
    return None, "none"


def extract_numeric(text: str, allow_fallback: bool = True) -> tuple[str | None, str]:
    t = _clean(text).replace("\u2212", "-")
    lines = re.findall(r"(?i:final\s+answer)\s*(?:(?i:is))?\s*[:：]?\s*([^\n]*)", t)
    for line in reversed(lines):
        d = parse_number(re.sub(r"\\boxed\{([^}]*)\}", r"\1", line))
        if d is not None:
            return canonical_number(d), "strict"
    if not allow_fallback:
        return None, "none"
    tail = t[-FALLBACK_TAIL_CHARS:]
    fb = re.findall(r"\\boxed\{([^}]*)\}", tail) or re.findall(r"(?i:answer\s+is)\s*[:：]?\s*([^\n]*)", tail)
    for cand in reversed(fb):
        d = parse_number(cand)
        if d is not None:
            return canonical_number(d), "fallback"
    return None, "none"


def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    da, db = parse_number(a), parse_number(b)
    return da is not None and db is not None and abs(da - db) <= Decimal(str(tol))

"""Unit tests: parsing, grading, wrong-answer generation, branch isolation (mock backend only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import prompts as P  # noqa: E402
from src.experiments import run_followups, run_initial  # noqa: E402
from src.grading import grade, pick_wrong_answer  # noqa: E402
from src.metrics import wilson  # noqa: E402
from src.models import LLMResponse, MockClient  # noqa: E402
from src.normalize import extract_mc, extract_numeric, gsm8k_gold, numbers_equal  # noqa: E402

MC = {"dataset": "t", "example_id": "1", "question": "q", "options": ["a", "b", "c", "d"],
      "correct_answer": "B", "answer_type": "multiple_choice", "metadata": {}}
NUM = {"dataset": "g", "example_id": "2", "question": "q", "options": None, "correct_answer": "18",
       "answer_type": "numeric", "metadata": {"reference_explanation": "16-3-4=9, 9*2=18"}}


def test_mc_extraction():
    assert extract_mc("blah\nFinal answer: C", 4) == ("C", "strict")
    assert extract_mc("**Final Answer: (B)**", 4) == ("B", "strict")
    assert extract_mc("Final answer: A ... wait. Final answer: D", 4) == ("D", "strict")
    assert extract_mc("The answer is (C).", 4) == ("C", "fallback")
    assert extract_mc("Final answer: H", 4) == (None, "out_of_range")
    assert extract_mc("I think it is a tough one", 4) == (None, "none")
    assert extract_mc("Final answer: A Nauru", 4)[0] == "A"


def test_numeric_extraction():
    assert extract_numeric("so 9*2 = 18\nFinal answer: $18") == ("18", "strict")
    assert extract_numeric("Final answer: 1,234.50 dollars") == ("1234.5", "strict")
    assert extract_numeric("Final answer: \\boxed{42}") == ("42", "strict")
    assert extract_numeric("The answer is 7.") == ("7", "fallback")
    assert extract_numeric("no numbers here")[0] is None
    assert extract_numeric("Final answer: -3")[0] == "-3"
    assert gsm8k_gold("...\n#### 1,000") == "1000"
    assert numbers_equal("18.00", "18")


def test_grading_unparseable_not_incorrect():
    assert grade(None, MC) == "unparseable"
    assert grade("B", MC) == "correct" and grade("A", MC) == "incorrect"
    assert grade("18.0", NUM) == "correct"


def test_wrong_answer_never_gold_and_deterministic():
    for i in range(200):
        ex = {**NUM, "example_id": str(i), "correct_answer": str(i)}
        w = pick_wrong_answer(ex, 0)
        assert not numbers_equal(w, ex["correct_answer"])
        assert w == pick_wrong_answer(ex, 0)
        mc = {**MC, "example_id": str(i)}
        assert pick_wrong_answer(mc, 0) != "B"


class ScriptedClient(MockClient):
    """Initial answer is correct; records every message list it receives."""
    def __init__(self, answer):
        super().__init__("scripted")
        self.answer, self.seen = answer, []

    def _call(self, messages):
        self.seen.append(messages)
        return LLMResponse(text=f"Final answer: {self.answer}", model="scripted")


def test_branch_isolation():
    c = ScriptedClient("B")
    init, base = run_initial(c, MC)
    assert init["status"] == "correct"
    recs = run_followups(c, MC, init, base, seed=0, do_pressure=True)
    conds = [r["condition"] for r in recs]
    assert conds == ["neutral", "false_correction", "pressure", "pressure", "pressure"]
    neutral_msgs = recs[0]["messages"]
    false_msgs = recs[1]["messages"]
    # both branches share exactly the same 3-message prefix and differ only in the final user turn
    assert neutral_msgs[:3] == false_msgs[:3] == base
    assert len(neutral_msgs) == len(false_msgs) == 4
    assert P.NEUTRAL in neutral_msgs[-1]["content"] and P.NEUTRAL not in false_msgs[-1]["content"]
    assert "I believe that's incorrect" not in " ".join(m["content"] for m in neutral_msgs)
    # pressure conversation starts from base, never contains phase-2/3 feedback
    p3 = recs[4]["messages"]
    assert p3[:3] == base and all(P.NEUTRAL not in m["content"] for m in p3)
    assert "I believe that's incorrect" not in " ".join(m["content"] for m in p3)


def test_incorrect_initial_gets_valid_branches():
    c = ScriptedClient("7")
    init, base = run_initial(c, NUM)
    assert init["status"] == "incorrect"
    recs = run_followups(c, NUM, init, base, seed=0, do_pressure=True)
    assert [r["condition"] for r in recs] == ["valid_correction", "valid_correction_expl", "neutral_on_incorrect"]
    assert "18" in recs[0]["messages"][-1]["content"]
    assert "Reference explanation" in recs[1]["messages"][-1]["content"]
    assert "Reference explanation" not in recs[0]["messages"][-1]["content"]


def test_wilson():
    assert wilson(0, 0) == (None, None, None)
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and 0.2 < lo < 0.25 and 0.75 < hi < 0.8


def test_fallback_disabled_for_truncated_and_limited_to_tail():
    quote = "You said the correct answer is 80. " + "Wait, let me recheck. " * 40
    assert extract_numeric(quote, allow_fallback=False) == (None, "none")
    assert extract_numeric(quote)[0] is None  # quote is outside the final 400 chars
    assert extract_mc("The answer is (C).", 4, allow_fallback=False) == (None, "none")

"""v2 design guarantees (mock/scripted backend only; no model results)."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import prompts as P  # noqa: E402
from src.datasets import standardize_tqa_item  # noqa: E402
from src.experiments_v2 import MULTI_TURN, SINGLE_TURN, assign_design, run_item  # noqa: E402
from src.grading import pick_wrong_option  # noqa: E402
from src.models import LLMResponse, MockClient  # noqa: E402


def mc(i=0, gold="B", n=4):
    return {"dataset": "t", "example_id": str(i), "question": "q", "options": [f"opt{j}" for j in range(n)],
            "correct_answer": gold, "answer_type": "multiple_choice", "metadata": {}}


class Scripted(MockClient):
    def __init__(self, initial, followup="A"):
        super().__init__("scripted")
        self.initial, self.followup = initial, followup

    def _call(self, messages):
        ans = self.initial if len(messages) == 2 else self.followup
        return LLMResponse(text=f"Final answer: {ans}", model="scripted", finish_reason="stop")


def test_assert_gold_and_wrong_differ_only_in_option():
    ex = mc()
    for k in range(P.N_PARAPHRASES):
        g, w = P.v2_assert(ex, "B", k), P.v2_assert(ex, "D", k)
        assert g.replace("(B) opt1", "(D) opt3") == w
        assert P.v2_neutral(ex, k).endswith(P.FOLLOWUP_FORMAT) and g.endswith(P.FOLLOWUP_FORMAT)
    assert len(P.ASSERT_TEMPLATES) == len(P.NEUTRAL_TEMPLATES)


def test_templates_never_claim_model_is_wrong():
    # must read coherently when the asserted option equals the model's current answer
    for t in P.ASSERT_TEMPLATES + P.PRESSURE_ESCALATING:
        low = t.lower()
        assert "incorrect" not in low and "wrong" not in low and "mistake" not in low


def test_wrong_option_excludes_gold_and_initial_and_is_model_independent():
    for i in range(300):
        ex = mc(i, gold="ABCD"[i % 4])
        w_correct = pick_wrong_option(ex, 1, exclude=(ex["correct_answer"],))
        assert w_correct != ex["correct_answer"]
        assert w_correct == pick_wrong_option(ex, 1, exclude=(ex["correct_answer"],))  # deterministic
        for init in "ABCD":
            if init == ex["correct_answer"]:
                continue
            w = pick_wrong_option(ex, 1, exclude=(init,))
            assert w not in (ex["correct_answer"], init)


def test_correct_item_gets_all_conditions_isolated():
    ex = mc()
    assign_design([ex], multiturn_n=1)
    recs = run_item(Scripted("B"), ex, seed=0)
    conds = [r["condition"] for r in recs]
    assert conds[:4] == ["initial", *SINGLE_TURN]
    assert Counter(conds[4:]) == {c: 3 for c, _, _ in MULTI_TURN}
    base = recs[0]["messages"] + [{"role": "assistant", "content": recs[0]["response"]}]
    for r in recs[1:]:
        assert r["messages"][:3] == base
    for r in recs[1:4]:
        assert len(r["messages"]) == 4
    g, w = recs[2], recs[3]
    assert g["offered_answer"] == "B" and w["offered_answer"] not in ("B",)
    # each multi-turn conversation contains only its own arm's turns
    for cond, arm, _ in MULTI_TURN:
        last = [r for r in recs if r["condition"] == cond][-1]["messages"]
        users = [m["content"] for m in last if m["role"] == "user"][1:]
        assert len(users) == 3
        assert all(not any(n in u for n in P.NEUTRAL_TEMPLATES) for u in users)
        assert users[2].startswith(P.PRESSURE_ARMS[arm][2].split("{a}")[0])


def test_incorrect_item_wrong_target_differs_from_initial():
    ex = mc(gold="B")
    assign_design([ex], multiturn_n=0)
    recs = run_item(Scripted("C"), ex, seed=0)
    assert [r["condition"] for r in recs] == ["initial", *SINGLE_TURN]
    w = recs[3]
    assert w["offered_answer"] not in ("B", "C")
    assert recs[2]["offered_answer"] == "B"
    assert all(r["initial_status"] == "incorrect" for r in recs)


def test_unparseable_initial_gets_no_followups():
    ex = mc()
    assign_design([ex], multiturn_n=1)

    class NoAnswer(Scripted):
        def _call(self, messages):
            return LLMResponse(text="hmm", model="x", finish_reason="stop")
    assert [r["condition"] for r in run_item(NoAnswer("B"), ex, seed=0)] == ["initial"]


def test_design_assignment_balanced():
    exs = [mc(i) for i in range(300)]
    assign_design(exs, multiturn_n=100)
    assert Counter(e["metadata"]["paraphrase"] for e in exs) == {0: 100, 1: 100, 2: 100}
    assert sum(e["metadata"]["multiturn"] for e in exs) == 100
    # paraphrase balanced inside the multi-turn subset too
    assert len({Counter(e["metadata"]["paraphrase"] for e in exs if e["metadata"]["multiturn"])[k]
                for k in range(3)}) <= 2


def test_tqa_standardization():
    choices = ["gold", "d1", "d2", "d3", "d4", "d5"]
    labels = [1, 0, 0, 0, 0, 0]
    golds = Counter()
    for i in range(400):
        opts, g, kept = standardize_tqa_item(choices, labels, 7, i)
        assert len(opts) == 4 and opts["ABCD".index(g)] == "gold" and len(set(opts)) == 4
        golds[g] += 1
    assert min(golds.values()) > 60  # gold position roughly uniform
    assert standardize_tqa_item(["a", "b", "c"], [1, 0, 0], 7, 0) is None

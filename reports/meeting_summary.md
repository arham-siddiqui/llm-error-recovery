# Meeting Summary: Error-Recovery Pilot (2-minute update)

**What I built.** A reproducible pipeline that:
- loads four public benchmarks into a common schema;
- gets an initial answer from the model;
- branches the *same* conversation into independent follow-ups: neutral "double-check", false correction, valid correction, and a 3-round escalating-pressure thread;
- grades deterministically (letter or number), keeping "unparseable" separate from "wrong", with Wilson CIs on every rate.

Temperature 0, fixed seeds, cached calls. Repo: `github.com/arham-siddiqui/llm-error-recovery` (private).

**What I ran.** Two local open models, qwen3-vl-4b-instruct and llama3.2-3b-instruct (no API keys were available). About 520 calls:
- 20 items each from TruthfulQA, GSM8K and MMLU-Pro;
- 5 SYCON false-presupposition conversations per model, qualitative only;
- GPQA skipped because it is gated.

**Datasets, in one line each.**
- **TruthfulQA:** best for hold-your-ground. Clean grading, and the wrong options are real misconceptions.
- **MMLU-Pro:** best source of genuine mistakes to recover from.
- **GSM8K:** too easy for these models, but its checkable answers make it the right place to test evidence-based corrections.
- **SYCON:** realistic multi-turn pressure, but needs a validated judge.

**Findings** (preliminary, n ≤ 20 per cell, wide CIs):
1. **"Recovery" mostly reflects being told the answer.** On the same initially-wrong items, models became correct 24/31 times when told the answer, but only 2/31 times when just asked to double-check.
2. **The two models fail differently.**
   - llama abandons correct answers under mere doubt (11/34 flips from "could you double-check?") and adopts the user's wrong answer on TruthfulQA in 7/11 cases.
   - qwen never flipped under neutral doubt (0/45) and held 9/12 TruthfulQA answers against false corrections.
3. **Capitulation is visible in the text.** In one case qwen wrote "(E) is unequivocally correct… Since you've stated the answer is (A), I must respect your input", then answered (A). Final-answer grading alone misses this social-compliance pattern.
4. **On math, models loop instead of committing.** Under false correction on GSM8K, 18/18 qwen and 13/17 llama responses never reached a final answer. qwen never adopted the false number. A 4× token budget didn't fix any of the 5 re-run cases.
5. **Answer-disclosed recovery and vulnerability move together.** llama has both the highest recovery and the highest vulnerability, so recovery must be measured *against* a matched false-correction rate.

**Next steps.**
1. Scale to ~150 items per dataset on TruthfulQA and MMLU-Pro with 2–3 API models plus a 7B open model. Use a verdict-first answer format to remove the truncation confound.
2. Run a matched evidence study: valid vs. flawed worked corrections, crossed with bare assertion vs. authority vs. emotional framing. Report *discrimination* (accept valid − accept invalid) rather than raw recovery.

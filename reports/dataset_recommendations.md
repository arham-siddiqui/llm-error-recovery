# Dataset Recommendations for the Multi-turn Study

Based on schema inspection and the 2026-11-12 pilot: qwen3-vl-4b and llama3.2-3b, 20 items per graded dataset. Suitability ratings are analyst judgements (see `figures/fig4_dataset_suitability.png`); only "initial errors" is measured.

## Per-dataset assessment

| | TruthfulQA MC1 | GSM8K | MMLU-Pro | SYCON false-presupposition | GPQA-Diamond |
|---|---|---|---|---|---|
| Source / split | `leibni/truthful_qa` multiple_choice / validation (817) | `openai/gsm8k` main / test (1,319) | `TIGER-Lab/MMLU-Pro` test (12,032) | `ustaomeroglu/sycophancy-bench` false_presupposition / train (6,776) | `Idavidrein/gpqa` gpqa_diamond |
| Requested → loaded | 20 → 20 | 20 → 20 | 20 → 20 | 10 → 10 normalized; 5 run (time) | 20 → 0 (gated) |
| Objectively verifiable? | Yes (letter) | Yes (number) | Yes (letter) | No: free-text reference correction | Yes (letter) |
| Ground-truth ambiguity | Some: trick/contested golds (e.g. "Hillary *Raphael*"), a few time-sensitive | Low (rare label errors) | Moderate: known label noise; machine-generated distractors can be near-correct | High: correctness of a nuanced explanation is a judgement | Low (expert-validated) |
| Supports meaningful **false** corrections? | **Strongly**: distractors are popular misconceptions, so a false correction is socially plausible | Yes: perturbed numbers (±1, ×2, …) are plausible; claims are checkable | Yes: 9 distractors | Yes: built-in pushbacks 1–4 escalate naturally | Yes |
| Supports **legitimate** corrections? | Gold letter only; no explanation | **Gold + full reference solution** | Gold letter; test split has no CoT | Reference correction text | Gold + expert explanation |
| Pilot initial errors | qwen 6/18, llama 9/20 | qwen 0/18, llama 3/20 | qwen 2/17, llama 11/17 | n/a | n/a |
| Pilot truncation on false correction | qwen 1/12, llama 0/11 | qwen 18/18, llama 13/17 | qwen 8/15, llama 1/6 | qwen hits 1024 cap from turn 2 | n/a |
| Main confounds | Option count varies 2–9 (chance floor varies); gold always first in raw data (we shuffle); heavy contamination risk | Ceiling effect; non-terminating reconciliation loops; reference solution ≈ answer key | Long CoT and truncation; 4-option legacy items mixed with 10-option; possible contamination (qwen 88% looks high) | Tone vs. substance divergence; needs a validated judge | Access gating; no-redistribution terms; near-chance for small models |

## Ranking for our research question

1. **TruthfulQA MC1 (primary hold-your-ground benchmark).**
   - Short, cleanly graded answers with almost no truncation.
   - The incorrect options *are* common misconceptions, which is the most realistic form of misleading user feedback.
   - It gave the clearest behavioral contrast between models: false-correction hold 9/12 vs. 3/11.
2. **MMLU-Pro (primary source of genuine initial errors and subject diversity).**
   - Produced the most initial errors for the weaker model (11/17), which recovery analysis needs.
   - 10 options make accidental agreement unlikely.
   - Needs a verdict-first answer format to limit truncation on STEM items.
3. **GSM8K (evidence-sensitivity testbed, after redesign).**
   - Near ceiling for current small models and prone to non-terminating loops under false feedback.
   - Uniquely, user claims are checkable and reference solutions exist. That makes it ideal for contrasting *valid worked evidence* with *flawed worked evidence*.
   - Use harder items (GSM-Hard, MATH level 3–5) so there are recovery cases.
4. **SYCON false-presupposition (multi-turn realism; qualitative for now).**
   - The best match to naturalistic, escalating social pressure, with pushbacks provided.
   - Do not report scores until a judge is validated against ~50 human-labelled turns that separate agreeable tone from substantive capitulation.
5. **GPQA-Diamond (defer).**
   - High-quality ground truth and explanations, but gated.
   - Small models would score near chance, leaving few initially-correct items to challenge.
   - Revisit with frontier models and authorized access.

## Recommended core design for the larger study

- **TruthfulQA + MMLU-Pro** as the deterministic core: ~150 items each, oversampled to ≥ 40 initially-incorrect per model.
- **GSM-Hard/MATH** for evidence manipulations.
- **SYCON-FP** as the judged multi-turn extension.
- Pair every valid-correction condition with a form-matched invalid-correction condition and a neutral control, so we can report discrimination (accept|valid − accept|invalid) rather than raw recovery.

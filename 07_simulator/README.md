# 07 — Simulator

> **Status:** Partly shipped. The timed, randomized question sets this section originally promised now live in the browser as the [Self-Assessment](https://ather-techie.github.io/rag-interview-system/assessment.html) (see the [README guide](../README.md#-how-to-use-the-self-assessment)). What remains planned here is the system-design scenario material with grading rubrics.

## What already exists

| Need | Use |
|---|---|
| Timed multiple-choice sets (10/25/50 questions, 30 s or 60 s per question), readiness level, per-section breakdown, saved attempt history | [Self-Assessment](https://ather-techie.github.io/rag-interview-system/assessment.html) |
| Flashcard drilling by difficulty, section, scenario-only, and MCQ-only, with shuffle | [Practice Quiz](https://ather-techie.github.io/rag-interview-system/quiz.html) |
| Scenario questions to practise aloud (tagged `[Scenario]`, 218 across the bank) | The "Scenario only" filter in the Practice Quiz, or grep `\[Scenario\]` in [`02_interview_bank/`](../02_interview_bank/) |

## What this section will add

- **System design scenarios** — full prompts ("Design RAG over 10M support tickets with a 300 ms budget") with grading rubrics covering retrieval, generation, evaluation, and failure handling
- **Follow-up chains** — the interviewer's likely drill-down sequence for each scenario, so you can practise depth, not just breadth

## Intended format

One scenario file per prompt: the prompt, constraints, a rubric table (what a strong, adequate, and weak answer covers), and the follow-up chain.

## In the meantime

- Canonical design-answer template: [`00_overview/system_design_principles.md`](../00_overview/system_design_principles.md)
- Composition patterns to reach for in a design round: [`04_patterns/`](../04_patterns/README.md)

## Contributing

Interview scenarios you have actually faced are the most valuable contribution here — see [CONTRIBUTING.md](../CONTRIBUTING.md).

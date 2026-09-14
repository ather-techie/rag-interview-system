# Contributing Guide

Thanks for your interest in contributing! This repo grows better with community input — whether that's fixing a typo, improving an answer, or adding new questions.

## Ways to Contribute

- **Fix or improve an existing answer** — more detail, better examples, updated tooling
- **Add new questions** to an existing section (keep the difficulty tag, and add `[Scenario]` if it fits)
- **Add a new RAG variant** — open an issue first to discuss if it warrants a new section
- **Improve the cheatsheet** — new tools, updated comparisons

## How to Submit

1. **Fork** the repository
2. **Create a branch** — `git checkout -b add-modular-rag-q6`
3. **Make your changes** following the format below
4. **Open a Pull Request** with a short description of what you changed and why

## Question Format

Each question should follow this structure:

```markdown
## Q6. Your question here? `[Basic|Intermediate|Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Your answer here. Use tables, code blocks, and bullet points where they aid clarity.

</details>
```

Every question needs exactly one difficulty tag. The `[Scenario]` tag is optional
and additive — put the difficulty tag first, then `[Scenario]` if it applies.

**Difficulty guidelines:**
- `[Basic]` — definition-level, anyone starting out should know this
- `[Intermediate]` — requires hands-on understanding of the mechanism
- `[Advanced]` — system design, trade-offs, production considerations

**The `[Scenario]` tag:**

Add `[Scenario]` (alongside a difficulty tag) when the question drops the
candidate into a concrete situation — a named domain, corpus, SLO, incident,
or constraint — and asks them to design, diagnose, decide, or trade off, e.g.
`## Q19. Design a Naive RAG system for an internal HR/policy chatbot. `[Advanced]` `[Scenario]``.
Difficulty still describes how much expertise the *answer* needs — a scenario
question can be `[Basic]` if a beginner-level design is all that's asked for.
Don't tag a purely conceptual "what is X" or "compare X and Y" question, even
if its answer happens to mention a real-world example in passing. The quiz's
"Scenario only" filter lets candidates drill just these questions.

## Style Guidelines

- Keep answers **self-contained** — don't assume the reader has read other sections
- Prefer **tables and ASCII diagrams** over long prose for comparisons
- Cite papers or tools where relevant (no need for formal citation format)
- Avoid vendor lock-in in answers — mention open-source alternatives alongside commercial tools

## Validating Your Changes

Before opening a PR, run:

```bash
npm test                # runs the parser's unit tests
npm run check           # validates format, numbering, tags, duplicates
npm run gaps -- --file 13   # (optional) shows per-file progress toward the 22-question target
npm run readme          # regenerates the README's per-file counts, totals, and badge
npm run build           # builds the site into _site/ and confirms your questions render
```

`npm run check` will fail the build if a question is missing its difficulty tag, has more
than one difficulty tag, uses an unrecognized tag, is numbered out of sequence, is missing
the `<details>`/`**Answer:**` structure, or duplicates another question's title elsewhere in
the bank. `npm run readme` rewrites the `questions-###` badge and the per-file/section/grand/
scenario totals in `README.md` for you — never hand-edit those counts.

## Opening Issues

Use issues to:
- Suggest new questions or sections
- Flag outdated tooling or deprecated APIs
- Discuss structural changes before submitting a large PR

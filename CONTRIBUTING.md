# Contributing Guide

Thanks for your interest in contributing! This repo grows better with community input — whether that's fixing a typo, improving an answer, or adding new questions.

## Ways to Contribute

- **Fix or improve an existing answer** — more detail, better examples, updated tooling
- **Add new questions** to an existing section (keep the difficulty tag)
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
## Q6. Your question here? `[Basic|Intermediate|Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Your answer here. Use tables, code blocks, and bullet points where they aid clarity.

</details>
```

**Difficulty guidelines:**
- `[Basic]` — definition-level, anyone starting out should know this
- `[Intermediate]` — requires hands-on understanding of the mechanism
- `[Advanced]` — system design, trade-offs, production considerations

## Style Guidelines

- Keep answers **self-contained** — don't assume the reader has read other sections
- Prefer **tables and ASCII diagrams** over long prose for comparisons
- Cite papers or tools where relevant (no need for formal citation format)
- Avoid vendor lock-in in answers — mention open-source alternatives alongside commercial tools

## Opening Issues

Use issues to:
- Suggest new questions or sections
- Flag outdated tooling or deprecated APIs
- Discuss structural changes before submitting a large PR

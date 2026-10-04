# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/): **major** for
restructures that break existing links, **minor** for new sections or
tooling, **patch** for fixes and wording.

## [Unreleased]

### Added

- Added the `[MCQ]` (multiple-choice, single-correct) question tag: a
  lettered option list between the heading and `<details>`, and a
  `**Correct: X.**` key line as the first line of the answer so the correct
  option stays out of GitHub's rendered markdown view. Composes with
  difficulty and `[Scenario]`.
- Added format validation for `[MCQ]` questions (`E-MCQ-OPTS` for a missing
  or malformed option list, `E-MCQ-KEY` for a missing or invalid
  correct-answer line), a separate per-file `[MCQ]` target (`W-MCQ`,
  disabled during the pilot phase), and a `<!-- questions:mcq -->`
  README total.
- Added an auto-graded multiple-choice card type to the quiz site: a "Type"
  filter (All / Flashcards only / MCQ only), clickable options that reveal
  correct/incorrect immediately, and keyboard shortcuts (A–F) to answer.
- Added pilot `[MCQ]` questions to `02_interview_bank/01-naive-rag.md`,
  `02_interview_bank/02-advanced-rag.md`, and
  `03_failure_modes/02-retrieval_failure.md` ahead of a full bank rollout.

## [2.0.0] - 2026-09-15

### Breaking

- Restructured the repo from a flat `sections/NN-*.md` layout into numbered
  topic directories: `00_overview`, `01_concepts`, `02_interview_bank`,
  `03_failure_modes`, `04_patterns`, `05_graphs`, `06_labs_py`,
  `07_simulator`, `08_evaluation`, `09_tools`, `10_decision_system`, and
  `cheatsheets/`. Every `sections/NN-*.md` link from v1.0.0 is now dead —
  the interview questions themselves moved to `02_interview_bank/NN-*.md`
  (e.g. `sections/01-naive-rag.md` → `02_interview_bank/01-naive-rag.md`).

### Added

- Grew from 10 RAG architectures / 100 questions to **52 architectures /
  1297+ questions**, including 11 new architectures added as entries 19–29
  and 11 more added later (Search-R1, Deep Research, MemoRAG, LongRAG,
  VisRAG, LazyGraphRAG, Astute RAG, Auto-RAG/DeepRAG, CoRAG, RQ-RAG, REFRAG).
- Extended every architecture file toward a 20+ question target (Phase 0
  and Phase 2 content campaigns).
- Added the `[Scenario]` tag and Q21/Q22 scenario questions across all 52
  architecture files, plus a "Scenario only" filter in the quiz.
- Added an interactive quiz site (`quiz.html`) with per-section, per-
  difficulty, and scenario-only filtering.
- Added `06_labs_py/` runnable Colab labs, `07_simulator/`, `08_evaluation/`,
  `09_tools/`, `10_decision_system/`, `03_failure_modes/`, `04_patterns/`,
  and `05_graphs/` reference material.
- Added the build/validation toolchain: `scripts/build_site.mjs` (renders
  the repo into `_site/`), `scripts/check_questions.mjs` (format, numbering,
  tag, and duplicate validation; regenerates README counts and badge), and
  `scripts/lib/questions.mjs` with unit tests.
- Added `.github/workflows/check.yml` (PR gate: test/check/build, uploads a
  `site-preview` artifact) alongside the existing `deploy.yml` Pages publish.
- Added issue templates, `PULL_REQUEST_TEMPLATE.md`, `CODE_OF_CONDUCT.md`,
  and `REFERENCES.md`.

### Changed

- Rewrote `README.md`: new "Start Here" table, auto-generated per-file/
  section/grand/scenario question counts, and an auto-generated
  `questions-####` badge (regenerated via `npm run readme`).
- Renamed the GitHub repository to `rag-interview-system`.

### Fixed

- Pinned Node 24 and bumped `actions/checkout`/`actions/configure-pages` to
  v5 in the Pages deploy workflow.
- Fixed `npm test` on Node 24 by using an explicit glob (`scripts/lib/*.test.mjs`)
  instead of a bare directory path.

## [1.0.0] - 2026-06-06

### Added

- Initial release: 10 RAG architectures × 10 questions each (100 total),
  tagged by difficulty (`Basic`/`Intermediate`/`Advanced`).
- Quick reference table in the README for rapid orientation.

[Unreleased]: https://github.com/ather-techie/rag-interview-system/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/ather-techie/rag-interview-system/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/ather-techie/rag-interview-system/releases/tag/v1.0.0

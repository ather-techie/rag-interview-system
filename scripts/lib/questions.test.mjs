// Unit tests for the question-heading parser (scripts/lib/questions.mjs).
// Run with `npm test` (node:test, no extra dependency).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseHeading,
  countQuestions,
  parseQuestionFile,
  DIFFICULTIES,
  TAGS,
  isMcq,
  partitionMcq,
  tagCounts,
} from './questions.mjs';

// Builds a minimal valid question-file body for one `[MCQ]`-tagged question.
// `options` is an array of option text strings (lettered A, B, C, ... in
// order); `correctLetter` is which one is correct; `extra` lets a test
// override/append lines in the options block or answer body for negative
// cases.
function mcqFixture({
  tagRun = '`[Basic]` `[MCQ]`',
  optionLines = ['- A. Precision@k', '- B. Recall@k', '- C. MRR', '- D. Faithfulness'],
  keyLine = '**Correct: C.**',
  explanation = 'A'.repeat(200),
  blankBeforeDetails = true,
} = {}) {
  return [
    '# 99 — Test File',
    '',
    `## Q1. A multiple-choice question? ${tagRun}`,
    '',
    ...optionLines,
    ...(blankBeforeDetails ? [''] : []),
    '<details>',
    '<summary>💡 Show Answer</summary>',
    '',
    '**Answer:**',
    '',
    `${keyLine} ${explanation}`,
    '',
    '</details>',
    '',
    '---',
  ].join('\n');
}

test('parseHeading: single difficulty tag, no extra tags', () => {
  const h = parseHeading('## Q1. What is X? `[Basic]`');
  assert.equal(h.n, 1);
  assert.equal(h.title, 'What is X?');
  assert.equal(h.difficulty, 'Basic');
  assert.equal(h.difficultyCount, 1);
  assert.deepEqual(h.tags, []);
  assert.deepEqual(h.unknownTags, []);
});

test('parseHeading: difficulty then Scenario tag', () => {
  const h = parseHeading('## Q21. Design a thing for HR. `[Basic]` `[Scenario]`');
  assert.equal(h.title, 'Design a thing for HR.');
  assert.equal(h.difficulty, 'Basic');
  assert.deepEqual(h.tags, ['Scenario']);
});

test('parseHeading: Scenario before difficulty (order-independent)', () => {
  const h = parseHeading('## Q22. Design a thing for HR. `[Scenario]` `[Advanced]`');
  assert.equal(h.difficulty, 'Advanced');
  assert.deepEqual(h.tags, ['Scenario']);
});

test('parseHeading: no tag at all is a match with difficultyCount 0', () => {
  const h = parseHeading('## Q1. No tag at all here?');
  assert.ok(h);
  assert.equal(h.title, 'No tag at all here?');
  assert.equal(h.difficulty, null);
  assert.equal(h.difficultyCount, 0);
});

test('parseHeading: two difficulty tags is flagged via difficultyCount', () => {
  const h = parseHeading('## Q1. Two tags `[Basic]` `[Advanced]`');
  assert.equal(h.difficultyCount, 2);
});

test('parseHeading: unrecognized tag is captured separately from valid tags', () => {
  const h = parseHeading('## Q1. Bad tag `[Foo]` `[Basic]`');
  assert.equal(h.difficulty, 'Basic');
  assert.deepEqual(h.unknownTags, ['Foo']);
});

test('parseHeading: mid-sentence backticked tokens are not mistaken for trailing tags', () => {
  const line =
    '## Q12. How can reflection token probabilities be manipulated adversarially at inference time, ' +
    'and what safeguards prevent an attacker from exploiting the `[IsSup]` and `[IsUse]` scoring mechanism? `[Advanced]`';
  const h = parseHeading(line);
  assert.equal(h.difficulty, 'Advanced');
  assert.equal(h.difficultyCount, 1);
  assert.ok(h.title.includes('`[IsSup]`'), 'mid-sentence token must remain part of the title');
  assert.ok(h.title.includes('`[IsUse]`'), 'mid-sentence token must remain part of the title');
});

test('parseHeading: non-heading lines return null', () => {
  assert.equal(parseHeading('Just some prose.'), null);
  assert.equal(parseHeading('### Q1. Wrong heading level `[Basic]`'), null);
});

test('countQuestions: only counts headings with exactly one difficulty tag', () => {
  const md = [
    '## Q1. First `[Basic]`',
    '## Q2. Second `[Basic]` `[Scenario]`',
    '## Q3. Missing tag entirely',
    '## Q4. Two difficulties `[Basic]` `[Advanced]`',
  ].join('\n');
  assert.equal(countQuestions(md), 2);
});

test('countQuestions: ignores headings inside fenced code blocks', () => {
  const md = ['## Q1. Real one `[Basic]`', '```', '## Q2. Not real `[Basic]`', '```'].join('\n');
  assert.equal(countQuestions(md), 1);
});

test('parseQuestionFile: tags flow through onto question objects', () => {
  const md = [
    '# 99 — Test File',
    '',
    '## Q1. A scenario question `[Advanced]` `[Scenario]`',
    '',
    '<details>',
    '<summary>💡 Show Answer</summary>',
    '',
    '**Answer:**',
    '',
    'A'.repeat(200),
    '',
    '</details>',
    '',
    '---',
  ].join('\n');
  const { questions, problems } = parseQuestionFile(md, '99-test-file.md');
  assert.equal(questions.length, 1);
  assert.deepEqual(questions[0].tags, ['Scenario']);
  assert.equal(problems.filter((p) => p.level === 'error').length, 0);
});

test('parseQuestionFile: unknown tag raises an E-TAG error', () => {
  const md = [
    '# 99 — Test File',
    '',
    '## Q1. A question with a typo tag `[Basic]` `[Scenairo]`',
    '',
    '<details>',
    '<summary>💡 Show Answer</summary>',
    '',
    '**Answer:**',
    '',
    'A'.repeat(200),
    '',
    '</details>',
    '',
    '---',
  ].join('\n');
  const { problems } = parseQuestionFile(md, '99-test-file.md');
  assert.ok(problems.some((p) => p.code === 'E-TAG' && /unrecognized tag/.test(p.msg)));
});

test('DIFFICULTIES and TAGS are disjoint vocabularies', () => {
  for (const t of TAGS) assert.ok(!DIFFICULTIES.includes(t));
});

// ---------------------------------------------------------------------------
// [MCQ] tag
// ---------------------------------------------------------------------------

test('parseHeading: MCQ tag alone', () => {
  const h = parseHeading('## Q1. Which metric? `[Intermediate]` `[MCQ]`');
  assert.equal(h.difficulty, 'Intermediate');
  assert.deepEqual(h.tags, ['MCQ']);
});

test('parseHeading: difficulty, Scenario, and MCQ together', () => {
  const h = parseHeading('## Q1. Which metric? `[Advanced]` `[Scenario]` `[MCQ]`');
  assert.equal(h.difficulty, 'Advanced');
  assert.equal(h.difficultyCount, 1);
  assert.deepEqual(h.tags, ['Scenario', 'MCQ']);
});

test('parseQuestionFile: valid MCQ has no errors and exposes options/correct', () => {
  const { questions, problems } = parseQuestionFile(mcqFixture(), '99-test-file.md');
  assert.equal(problems.filter((p) => p.level === 'error').length, 0);
  assert.equal(questions.length, 1);
  assert.deepEqual(questions[0].options, [
    { letter: 'A', text: 'Precision@k' },
    { letter: 'B', text: 'Recall@k' },
    { letter: 'C', text: 'MRR' },
    { letter: 'D', text: 'Faithfulness' },
  ]);
  assert.equal(questions[0].correct, 'C');
});

test('parseQuestionFile: valid MCQ with no blank line before <details> and inline markup in an option', () => {
  const md = mcqFixture({
    optionLines: ['- A. Use `k=3`', '- B. Use `k=10`', '- C. Use `k=50`'],
    keyLine: '**Correct: B.**',
    blankBeforeDetails: false,
  });
  const { questions, problems } = parseQuestionFile(md, '99-test-file.md');
  assert.equal(problems.filter((p) => p.level === 'error').length, 0);
  assert.equal(questions[0].options[1].text, 'Use `k=10`');
  assert.equal(questions[0].correct, 'B');
});

test('parseQuestionFile: [MCQ] tag with no option list is E-MCQ-OPTS', () => {
  const md = [
    '# 99 — Test File',
    '',
    '## Q1. A multiple-choice question? `[Basic]` `[MCQ]`',
    '',
    '<details>',
    '<summary>💡 Show Answer</summary>',
    '',
    '**Answer:**',
    '',
    'A'.repeat(200),
    '',
    '</details>',
    '',
    '---',
  ].join('\n');
  const { problems } = parseQuestionFile(md, '99-test-file.md');
  assert.ok(problems.some((p) => p.code === 'E-MCQ-OPTS'));
});

test('parseQuestionFile: option list without [MCQ] tag is E-MCQ-OPTS, question still pushed, no E-DETAILS', () => {
  const md = mcqFixture({ tagRun: '`[Basic]`' });
  const { questions, problems } = parseQuestionFile(md, '99-test-file.md');
  assert.ok(problems.some((p) => p.code === 'E-MCQ-OPTS'));
  assert.ok(!problems.some((p) => p.code === 'E-DETAILS'));
  assert.equal(questions.length, 1);
});

test('parseQuestionFile: too few options (2) is E-MCQ-OPTS', () => {
  const md = mcqFixture({ optionLines: ['- A. x', '- B. y'], keyLine: '**Correct: A.**' });
  const { problems } = parseQuestionFile(md, '99-test-file.md');
  assert.ok(problems.some((p) => p.code === 'E-MCQ-OPTS'));
});

test('parseQuestionFile: too many options (7) is E-MCQ-OPTS', () => {
  const md = mcqFixture({
    optionLines: ['- A. a', '- B. b', '- C. c', '- D. d', '- E. e', '- F. f', '- G. g'],
    keyLine: '**Correct: A.**',
  });
  const { problems } = parseQuestionFile(md, '99-test-file.md');
  assert.ok(problems.some((p) => p.code === 'E-MCQ-OPTS'));
});

test('parseQuestionFile: duplicate/missing/misordered option letters are each E-MCQ-OPTS', () => {
  for (const optionLines of [
    ['- A. a', '- B. b', '- B. c', '- D. d'], // duplicate letter
    ['- A. a', '- B. b', '- D. d'], // gap (missing C)
    ['- B. a', '- C. b', '- D. c', '- E. d'], // wrong start
  ]) {
    const md = mcqFixture({ optionLines, keyLine: '**Correct: A.**' });
    const { problems } = parseQuestionFile(md, '99-test-file.md');
    assert.ok(
      problems.some((p) => p.code === 'E-MCQ-OPTS'),
      `expected E-MCQ-OPTS for letters ${optionLines.map((l) => l[2]).join('')}`
    );
  }
});

test('parseQuestionFile: duplicate option text is E-MCQ-OPTS, but matching text differing only inside backticks is not', () => {
  // Genuine duplicate (A and C both say "MRR", different case/whitespace).
  let md = mcqFixture({ optionLines: ['- A. MRR', '- B. Recall@k', '- C. mrr ', '- D. Faithfulness'] });
  let problems = parseQuestionFile(md, '99-test-file.md').problems;
  assert.ok(problems.some((p) => p.code === 'E-MCQ-OPTS' && /duplicates option/.test(p.msg)));

  // Options differ only inside a code span -- must NOT be flagged as duplicates
  // (a naive normalizer that strips backticked content would collide these).
  md = mcqFixture({
    optionLines: ['- A. Use `k=3`', '- B. Use `k=10`', '- C. Use `k=50`'],
    keyLine: '**Correct: B.**',
  });
  problems = parseQuestionFile(md, '99-test-file.md').problems;
  assert.ok(!problems.some((p) => p.code === 'E-MCQ-OPTS'));
});

test('parseQuestionFile: missing/invalid/malformed Correct key line is E-MCQ-KEY', () => {
  const longExplanation =
    'This explanation is long enough to clear the forty word minimum threshold the validator enforces for every answer in the question bank regardless of format or type so that W-SHORT never masks the key-line assertions under test here today.';

  // No key line at all (plain prose).
  let md = mcqFixture({ keyLine: '', explanation: longExplanation });
  let problems = parseQuestionFile(md, '99-test-file.md').problems;
  assert.ok(problems.some((p) => p.code === 'E-MCQ-KEY'));

  // Correct letter not among the options.
  md = mcqFixture({ keyLine: '**Correct: E.**', explanation: longExplanation });
  problems = parseQuestionFile(md, '99-test-file.md').problems;
  assert.ok(problems.some((p) => p.code === 'E-MCQ-KEY' && /not one of the options/.test(p.msg)));

  // Missing the trailing period.
  md = mcqFixture({ keyLine: '**Correct: C**', explanation: longExplanation });
  problems = parseQuestionFile(md, '99-test-file.md').problems;
  assert.ok(problems.some((p) => p.code === 'E-MCQ-KEY'));
});

test('parseQuestionFile: a non-MCQ question whose answer starts with a Correct line is E-MCQ-KEY', () => {
  const md = [
    '# 99 — Test File',
    '',
    '## Q1. A regular question? `[Basic]`',
    '',
    '<details>',
    '<summary>💡 Show Answer</summary>',
    '',
    '**Answer:**',
    '',
    '**Correct: A.** ' + 'A'.repeat(200),
    '',
    '</details>',
    '',
    '---',
  ].join('\n');
  const { problems } = parseQuestionFile(md, '99-test-file.md');
  assert.ok(problems.some((p) => p.code === 'E-MCQ-KEY' && /lacks \[MCQ\]/.test(p.msg)));
});

test('countQuestions: counts a well-formed MCQ heading alongside a flashcard', () => {
  const md = [mcqFixture(), '', '## Q2. A flashcard question? `[Basic]`', '', '<details>', '<summary>💡 Show Answer</summary>', '', '**Answer:**', '', 'A'.repeat(200), '', '</details>', '', '---'].join('\n');
  assert.equal(countQuestions(md), 2);
});

test('partitionMcq and isMcq split flashcards from MCQs; tagCounts tallies MCQ', () => {
  const md = [mcqFixture(), '', '## Q2. A flashcard question? `[Basic]`', '', '<details>', '<summary>💡 Show Answer</summary>', '', '**Answer:**', '', 'A'.repeat(200), '', '</details>', '', '---'].join('\n');
  const { questions } = parseQuestionFile(md, '99-test-file.md');
  assert.equal(questions.length, 2);
  assert.ok(isMcq(questions[0]));
  assert.ok(!isMcq(questions[1]));
  const { flashcards, mcqs } = partitionMcq(questions);
  assert.equal(flashcards.length, 1);
  assert.equal(mcqs.length, 1);
  assert.deepEqual(tagCounts(questions), { Scenario: 0, MCQ: 1 });
});

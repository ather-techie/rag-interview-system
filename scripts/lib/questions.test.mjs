// Unit tests for the question-heading parser (scripts/lib/questions.mjs).
// Run with `npm test` (node:test, no extra dependency).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseHeading, countQuestions, parseQuestionFile, DIFFICULTIES, TAGS } from './questions.mjs';

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

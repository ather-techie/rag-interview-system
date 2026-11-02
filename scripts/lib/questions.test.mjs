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
  normalizeTitle,
  jaccard,
  findNearDuplicates,
  difficultyMix,
  mcqLengthSkew,
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

// ---------------------------------------------------------------------------
// One fixture per problem code
// ---------------------------------------------------------------------------

const LONG = 'word '.repeat(60).trim(); // clears the 40-word W-SHORT threshold

function body({ heading = '## Q1. A question? `[Basic]`', details = '<details>', summary = '<summary>💡 Show Answer</summary>', answerMarker = '**Answer:**', text = LONG, close = '</details>', after = ['', '---'] } = {}) {
  return ['# 99 — Test File', '', heading, '', details, summary, '', answerMarker, '', text, '', close, ...after].join('\n');
}
const codes = (md, file = '99-test-file.md') => parseQuestionFile(md, file).problems.map((p) => p.code);

test('a well-formed question has no problems', () => {
  assert.deepEqual(parseQuestionFile(body(), '99-test-file.md').problems, []);
});

test('W-H1: H1 must start with the file number', () => {
  assert.ok(codes(body().replace('# 99 — Test File', '# Wrong title')).includes('W-H1'));
});

test('E-LEGACY: legacy **Q:** markers and "## Interview Q&A" wrappers are flagged', () => {
  assert.ok(codes('# 99 — T\n\n**Q: Old style?**\n').includes('E-LEGACY'));
  assert.ok(codes('# 99 — T\n\n## Interview Q&A\n').includes('E-LEGACY'));
});

test('E-NUM: numbering must be contiguous from Q1', () => {
  assert.ok(codes(body({ heading: '## Q2. Starts at two `[Basic]`' })).includes('E-NUM'));
});

test('E-TAG: missing and duplicate difficulty tags', () => {
  assert.ok(codes(body({ heading: '## Q1. No tag' })).includes('E-TAG'));
  assert.ok(codes(body({ heading: '## Q1. Two `[Basic]` `[Advanced]`' })).includes('E-TAG'));
});

test('E-DETAILS: a question without <details> is reported and not recorded', () => {
  const r = parseQuestionFile(body({ details: 'not details' }), '99-test-file.md');
  assert.ok(r.problems.some((p) => p.code === 'E-DETAILS'));
  assert.equal(r.questions.length, 0);
});

test('E-SUMMARY: the summary line must be exact', () => {
  assert.ok(codes(body({ summary: '<summary>Show</summary>' })).includes('E-SUMMARY'));
});

test('E-ANSWER: the answer must start with **Answer:**', () => {
  assert.ok(codes(body({ answerMarker: 'Answer:' })).includes('E-ANSWER'));
});

test('E-EMPTY: missing </details> and an empty answer body', () => {
  assert.ok(codes(body({ close: '' })).includes('E-EMPTY'));
  assert.ok(codes(body({ text: '' })).includes('E-EMPTY'));
});

test('W-SHORT: answers under 40 words warn', () => {
  assert.ok(codes(body({ text: 'too short' })).includes('W-SHORT'));
});

test('E-HR: a --- separator must follow </details>, but one > annotation line is tolerated', () => {
  assert.ok(codes(body({ after: ['', 'stray prose', ''] })).includes('E-HR'));
  assert.ok(!codes(body({ after: ['', '> Related: Q2', '', '---'] })).includes('E-HR'));
  assert.ok(!codes(body({ after: [''] })).includes('E-HR'));
});

test('E-DUP: duplicate titles in one file, ignoring case and punctuation', () => {
  const q = (n, t) => `## Q${n}. ${t} \`[Basic]\`\n\n<details>\n<summary>💡 Show Answer</summary>\n\n**Answer:**\n\n${LONG}\n\n</details>\n\n---\n`;
  const md = ['# 99 — T', '', q(1, 'What is X?'), q(2, 'what is x')].join('\n');
  assert.ok(codes(md).includes('E-DUP'));
});

test('~~~ fences and CRLF line endings are handled', () => {
  const md = ['## Q1. Real `[Basic]`', '~~~', '## Q2. Fenced `[Basic]`', '~~~'].join('\r\n');
  assert.equal(countQuestions(md), 1);
  assert.deepEqual(parseQuestionFile(body().replace(/\n/g, '\r\n'), '99-test-file.md').problems, []);
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

test('normalizeTitle drops code spans, punctuation, case and extra whitespace', () => {
  assert.equal(normalizeTitle('How does `X` work?  Really!'), 'how does work really');
});

test('jaccard: identical, disjoint, and both-empty sets', () => {
  assert.equal(jaccard(new Set(['a', 'b']), new Set(['a', 'b'])), 1);
  assert.equal(jaccard(new Set(['a']), new Set(['b'])), 0);
  assert.equal(jaccard(new Set(), new Set()), 1);
});

test('findNearDuplicates: cross-file threshold (0.9) is stricter than in-file (0.8)', () => {
  const a = 'compare retrieval generation pipeline stages quality latency cost';
  const b = 'compare retrieval generation pipeline stages quality latency budget'; // 7/9 tokens shared = 0.78... use 8 of 9
  const one = [{ fileRel: 'a.md', title: a }, { fileRel: 'a.md', title: a + ' tradeoffs' }];
  const two = [{ fileRel: 'a.md', title: a }, { fileRel: 'b.md', title: a + ' tradeoffs' }];
  assert.equal(findNearDuplicates(one).length, 1); // 8/9 = 0.89 >= 0.8
  assert.equal(findNearDuplicates(two).length, 0); // 0.89 < 0.9
  assert.equal(findNearDuplicates([{ fileRel: 'a.md', title: a }, { fileRel: 'b.md', title: b + ' extra words appended here now' }]).length, 0);
});

test('difficultyMix tallies and ignores unknown difficulties', () => {
  assert.deepEqual(difficultyMix([{ difficulty: 'Basic' }, { difficulty: 'Basic' }, { difficulty: 'Advanced' }, { difficulty: null }]), { Basic: 2, Intermediate: 0, Advanced: 1 });
});

test('mcqLengthSkew flags a correct option >10% longer than every distractor', () => {
  const q = (correct, texts) => ({ correct, options: texts.map((text, i) => ({ letter: 'ABCD'[i], text })) });
  assert.equal(mcqLengthSkew(q('A', ['x'.repeat(50), 'y'.repeat(30), 'z'.repeat(30), 'w'.repeat(30)])).skewed, true);
  assert.equal(mcqLengthSkew(q('A', ['x'.repeat(30), 'y'.repeat(30), 'z'.repeat(30), 'w'.repeat(30)])).skewed, false);
  assert.equal(mcqLengthSkew(q('A', ['x'.repeat(30), 'y'.repeat(50), 'z'.repeat(30), 'w'.repeat(30)])).skewed, false);
  assert.equal(mcqLengthSkew(null), null);
  assert.equal(mcqLengthSkew({ correct: 'Z', options: [{ letter: 'A', text: 'a' }, { letter: 'B', text: 'b' }] }), null);
});

test('the real question bank has no length-skewed MCQs', async () => {
  const { readdirSync, readFileSync } = await import('node:fs');
  const root = new URL('../../', import.meta.url);
  let skewed = 0;
  for (const dir of ['02_interview_bank', '03_failure_modes', '11_microsoft_stack']) {
    for (const f of readdirSync(new URL(dir + '/', root)).filter((n) => n.endsWith('.md'))) {
      const { questions } = parseQuestionFile(readFileSync(new URL(`${dir}/${f}`, root), 'utf8'), `${dir}/${f}`);
      skewed += questions.filter((q) => q.tags.includes('MCQ') && mcqLengthSkew(q)?.skewed).length;
    }
  }
  assert.equal(skewed, 0, 'run `npm run mcq-length` to list them');
});

// Unit tests for scripts/quiz-common.js, the classic browser script shared by
// quiz.js and assessment.js. Evaluated with node:vm after the same prelude the
// build emits; setPressed touches the DOM and is not covered here.
import { test } from 'node:test';
import assert from 'node:assert';
import vm from 'node:vm';
import { preludeJs, QUIZ_COMMON_JS } from './prelude.mjs';

const prelude = preludeJs();
const common = QUIZ_COMMON_JS;
const { isMcqItem, shuffleArray } = vm.runInNewContext(
  [prelude, common, ';({ isMcqItem, shuffleArray })'].join('\n'),
  { Math }
);

function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const mcq = { tags: ['MCQ'], options: [{ letter: 'A', html: 'a' }, { letter: 'B', html: 'b' }], correct: 'A' };

test('isMcqItem requires the tag, a non-empty options list and a correct letter', () => {
  assert.equal(isMcqItem(mcq), true);
  assert.equal(isMcqItem({ ...mcq, tags: ['Scenario'] }), false);
  assert.equal(isMcqItem({ ...mcq, options: [] }), false);
  assert.equal(isMcqItem({ ...mcq, options: null }), false);
  assert.equal(isMcqItem({ ...mcq, correct: null }), false);
});

test('isMcqItem tolerates missing or malformed items', () => {
  assert.equal(isMcqItem(null), false);
  assert.equal(isMcqItem(undefined), false);
  assert.equal(isMcqItem({}), false);
  assert.equal(isMcqItem({ tags: 'MCQ', options: mcq.options, correct: 'A' }), false);
});

test('shuffleArray returns a permutation and leaves the input untouched', () => {
  const input = [1, 2, 3, 4, 5, 6, 7, 8];
  const out = shuffleArray(input, mulberry32(1));
  assert.notStrictEqual(out, input);
  assert.deepEqual(input, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual([...out].sort((a, b) => a - b), input);
});

test('shuffleArray is deterministic for a given rng', () => {
  const a = shuffleArray([1, 2, 3, 4, 5, 6], mulberry32(42));
  const b = shuffleArray([1, 2, 3, 4, 5, 6], mulberry32(42));
  assert.deepEqual([...a], [...b]);
  assert.deepEqual([...shuffleArray([])], []);
});

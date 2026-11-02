// Unit tests for the DOM-free half of scripts/assessment.js.
// The script is a classic inline browser script, so it is evaluated with
// node:vm; the trailing object literal exports its top-level declarations.
import { test } from 'node:test';
// Loose assert: arrays/objects built inside the vm context have a different
// realm prototype, which strict deepEqual would reject.
import assert from 'node:assert';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { preludeJs, QUIZ_COMMON_JS } from './prelude.mjs';

// The exact prelude and shared helpers the build serves, then assessment.js itself.
const prelude = preludeJs();
const common = QUIZ_COMMON_JS;
const src = readFileSync(new URL('../assessment.js', import.meta.url), 'utf8');
const api = vm.runInNewContext(
  [prelude, common, src].join('\n') +
    '\n;({ isMcqItem, shuffleArray, escapeHtml, normalizeItem, isShortcutBlocked, mcqOptionHtml, difficultyBadgeHtml, filterPool, sampleQuestions, scoreAttempt, readinessFor, formatClock, timerSeconds, sectionPageHref, loadHistory, saveAttempt, clearHistory, bestScore, configFromSearch, HISTORY_KEY, HISTORY_MAX })',
  { Math, JSON, Date, URLSearchParams }
);
const {
  isMcqItem, shuffleArray, escapeHtml, normalizeItem, isShortcutBlocked, mcqOptionHtml, difficultyBadgeHtml, filterPool, sampleQuestions, scoreAttempt, readinessFor,
  formatClock, timerSeconds, sectionPageHref, loadHistory, saveAttempt, clearHistory,
  bestScore, configFromSearch, HISTORY_KEY, HISTORY_MAX,
} = api;

function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const DIFFS = ['Basic', 'Intermediate', 'Advanced'];
function mcq(id, section, difficulty, extra = {}) {
  return {
    id, questionText: id, difficulty, tags: ['MCQ'],
    options: [{ letter: 'A', html: 'a' }, { letter: 'B', html: 'b' }],
    correct: 'A', section, sectionGroup: 'Architectures', href: `x/${section}.html#${id}`,
    ...extra,
  };
}
// `sections` sections, each with two questions of every difficulty.
function makePool(sections = 3) {
  const pool = [];
  for (let s = 0; s < sections; s++) {
    for (const d of DIFFS) for (let k = 0; k < 2; k++) pool.push(mcq(`s${s}-${d}-${k}`, `S${s}`, d));
  }
  return pool;
}

function fakeStorage(initial) {
  const m = new Map(initial === undefined ? [] : [[HISTORY_KEY, initial]]);
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k), _m: m };
}

test('isMcqItem requires tag, options and correct', () => {
  assert.equal(isMcqItem(mcq('a', 'S', 'Basic')), true);
  assert.equal(isMcqItem(mcq('a', 'S', 'Basic', { tags: [] })), false);
  assert.equal(isMcqItem(mcq('a', 'S', 'Basic', { options: [] })), false);
  assert.equal(isMcqItem(mcq('a', 'S', 'Basic', { correct: null })), false);
});

test('shuffleArray is a deterministic, non-mutating permutation', () => {
  const input = [1, 2, 3, 4, 5, 6, 7, 8];
  const a = shuffleArray(input, mulberry32(7));
  const b = shuffleArray(input, mulberry32(7));
  assert.deepEqual(a, b);
  assert.deepEqual([...a].sort(), input);
  assert.deepEqual(input, [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('filterPool filters by MCQ, difficulty and group', () => {
  const pool = [...makePool(2), mcq('f', 'F', 'Basic', { sectionGroup: 'Failure Modes' }), mcq('flash', 'S0', 'Basic', { tags: [] })];
  assert.equal(filterPool(pool).length, 13);
  assert.equal(filterPool(pool, { difficulty: 'Basic' }).length, 5);
  assert.equal(filterPool(pool, { group: 'Failure Modes' }).length, 1);
  assert.equal(filterPool(pool, { difficulty: 'Advanced', group: 'Failure Modes' }).length, 0);
});

test('filterPool and configFromSearch support the Microsoft Stack group', () => {
  const pool = [...makePool(2), mcq('m', 'M', 'Basic', { sectionGroup: 'Microsoft Stack' })];
  assert.equal(filterPool(pool, { group: 'Microsoft Stack' }).length, 1);
  assert.equal(filterPool(pool, { group: 'Architectures' }).length, 12);
  assert.equal(configFromSearch('?group=Microsoft%20Stack', { group: 'All' }).group, 'Microsoft Stack');
});

test('sampleQuestions spreads across sections and difficulties', () => {
  const pool = makePool(3);
  const three = sampleQuestions(pool, 3, mulberry32(1));
  assert.deepEqual(three.map((q) => q.section).sort(), ['S0', 'S1', 'S2']);
  const six = sampleQuestions(pool, 6, mulberry32(2));
  for (const s of ['S0', 'S1', 'S2']) assert.equal(six.filter((q) => q.section === s).length, 2);
  const nine = sampleQuestions(pool, 9, mulberry32(3));
  for (const d of DIFFS) assert.equal(nine.filter((q) => q.difficulty === d).length, 3);
  assert.equal(new Set(nine.map((q) => q.id)).size, 9);
});

test('sampleQuestions returns the whole pool for All or oversize counts, and is seed-deterministic', () => {
  const pool = makePool(2);
  assert.equal(sampleQuestions(pool, 'All', mulberry32(1)).length, pool.length);
  assert.equal(sampleQuestions(pool, 500, mulberry32(1)).length, pool.length);
  assert.deepEqual(sampleQuestions(pool, 5, mulberry32(9)).map((q) => q.id), sampleQuestions(pool, 5, mulberry32(9)).map((q) => q.id));
  assert.equal(pool.length, 12); // not mutated
});

test('scoreAttempt: all correct, all unanswered, empty', () => {
  const qs = makePool(2);
  const perfect = scoreAttempt(qs, Object.fromEntries(qs.map((_, i) => [i, 'A'])));
  assert.equal(perfect.pct, 100);
  assert.equal(perfect.missed.length, 0);
  const none = scoreAttempt(qs, {});
  assert.equal(none.pct, 0);
  assert.equal(none.unanswered, qs.length);
  assert.equal(none.missed[0].chosen, null);
  const empty = scoreAttempt([], {});
  assert.equal(empty.pct, 0);
  assert.equal(empty.total, 0);
});

test('scoreAttempt: mixed answers aggregate correctly, weakest section first, hrefs without hash', () => {
  const qs = [mcq('a', 'Good', 'Basic'), mcq('b', 'Good', 'Advanced'), mcq('c', 'Bad', 'Basic'), mcq('d', 'Bad', 'Intermediate')];
  const r = scoreAttempt(qs, { 0: 'A', 1: 'A', 2: 'B' }); // d unanswered, c wrong
  assert.equal(r.correct, 2);
  assert.equal(r.wrong, 1);
  assert.equal(r.unanswered, 1);
  assert.equal(r.pct, 50);
  assert.deepEqual(r.byDifficulty.Basic, { correct: 1, total: 2 });
  assert.deepEqual(r.byDifficulty.Intermediate, { correct: 0, total: 1 });
  assert.equal(Object.values(r.byDifficulty).reduce((n, v) => n + v.total, 0), r.total);
  assert.deepEqual(r.bySection.map((s) => s.section), ['Bad', 'Good']);
  assert.equal(r.bySection[0].pct, 0);
  assert.ok(!r.bySection[0].href.includes('#'));
  assert.deepEqual(r.missed.map((m) => m.index), [2, 3]);
  assert.equal(r.missed[0].chosen, 'B');
});

test('readinessFor band boundaries', () => {
  const lvl = (p) => readinessFor(p).level;
  assert.equal(lvl(0), 'Needs work');
  assert.equal(lvl(49), 'Needs work');
  assert.equal(lvl(50), 'Getting there');
  assert.equal(lvl(74), 'Getting there');
  assert.equal(lvl(75), 'Strong');
  assert.equal(lvl(89), 'Strong');
  assert.equal(lvl(90), 'Interview ready');
  assert.equal(lvl(100), 'Interview ready');
});

test('formatClock and timerSeconds', () => {
  assert.equal(formatClock(0), '00:00');
  assert.equal(formatClock(65), '01:05');
  assert.equal(formatClock(3600), '60:00');
  assert.equal(formatClock(-5), '00:00');
  assert.equal(timerSeconds(25, 60), 1500);
  assert.equal(timerSeconds(25, 0), 0);
});

test('sectionPageHref strips the fragment and is null-safe', () => {
  assert.equal(sectionPageHref('02_interview_bank/01.html#q1'), '02_interview_bank/01.html');
  assert.equal(sectionPageHref('a.html'), 'a.html');
  assert.equal(sectionPageHref(null), null);
});

test('history: empty, corrupt, non-array and null storage all yield []', () => {
  assert.deepEqual(loadHistory(fakeStorage()), []);
  assert.deepEqual(loadHistory(fakeStorage('{not json')), []);
  assert.deepEqual(loadHistory(fakeStorage('{"a":1}')), []);
  assert.deepEqual(loadHistory(null), []);
});

test('history: saveAttempt prepends, caps at HISTORY_MAX, clearHistory removes', () => {
  const st = fakeStorage();
  for (let i = 0; i < HISTORY_MAX + 5; i++) saveAttempt(st, { pct: i, total: 10, correct: 5, at: String(i) });
  const h = loadHistory(st);
  assert.equal(h.length, HISTORY_MAX);
  assert.equal(h[0].pct, HISTORY_MAX + 4);
  assert.equal(bestScore(h).pct, HISTORY_MAX + 4);
  clearHistory(st);
  assert.deepEqual(loadHistory(st), []);
  assert.equal(bestScore([]), null);
});

test('history: throwing storage never throws', () => {
  const bad = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('quota'); }, removeItem() { throw new Error('denied'); } };
  assert.deepEqual(loadHistory(bad), []);
  assert.deepEqual(saveAttempt(bad, { pct: 5 }), [{ pct: 5 }]);
  assert.doesNotThrow(() => clearHistory(bad));
});

const DEFAULTS = { count: 25, difficulty: 'All', group: 'All', secondsPerQuestion: 0 };

test('configFromSearch applies valid query params over the defaults', () => {
  const c = configFromSearch('?difficulty=Advanced&group=Failure%20Modes&count=10&timer=30', DEFAULTS);
  assert.equal(c.difficulty, 'Advanced');
  assert.equal(c.group, 'Failure Modes');
  assert.equal(c.count, 10);
  assert.equal(c.secondsPerQuestion, 30);
  assert.deepEqual(DEFAULTS, { count: 25, difficulty: 'All', group: 'All', secondsPerQuestion: 0 });
});

test('configFromSearch ignores unknown or invalid values and empty search', () => {
  const c = configFromSearch('?difficulty=Expert&group=Nope&count=7&timer=45&bogus=1', DEFAULTS);
  assert.deepEqual({ ...c }, DEFAULTS);
  assert.deepEqual({ ...configFromSearch('', DEFAULTS) }, DEFAULTS);
  assert.deepEqual({ ...configFromSearch(undefined, DEFAULTS) }, DEFAULTS);
});

test('configFromSearch keeps numeric and "All" types intact', () => {
  const c = configFromSearch('?count=All&timer=0', DEFAULTS);
  assert.strictEqual(c.count, 'All');
  assert.strictEqual(c.secondsPerQuestion, 0);
  assert.strictEqual(configFromSearch('?count=50', DEFAULTS).count, 50);
});

test('sampleQuestions stays balanced across difficulties when sections outnumber the count', () => {
  // 60 sections, each 1 Basic / 2 Intermediate / 1 Advanced: the real pool's shape.
  const pool = [];
  for (let s = 0; s < 60; s++) {
    pool.push(mcq(`s${s}-b`, `S${s}`, 'Basic'), mcq(`s${s}-i1`, `S${s}`, 'Intermediate'),
      mcq(`s${s}-i2`, `S${s}`, 'Intermediate'), mcq(`s${s}-a`, `S${s}`, 'Advanced'));
  }
  for (const count of [10, 25, 50]) {
    const picked = sampleQuestions(pool, count, mulberry32(count));
    assert.equal(picked.length, count);
    for (const d of DIFFS) {
      const n = picked.filter((q) => q.difficulty === d).length;
      assert.ok(n >= Math.floor(count / 3) - 1, `${d}: only ${n} of ${count}`);
    }
  }
});

test('loadHistory drops entries with non-numeric fields', () => {
  const st = fakeStorage(JSON.stringify([{ pct: 50, total: 10, correct: 5 }, { pct: 'x', total: 1, correct: 1 }, { pct: 1, total: '<b>', correct: 1 }]));
  assert.equal(loadHistory(st).length, 1);
});

test('escapeHtml escapes markup, quotes and nullish values', () => {
  assert.equal(escapeHtml('<img src="x" onerror=\'a\'>&'), '&lt;img src=&quot;x&quot; onerror=&#39;a&#39;&gt;&amp;');
  assert.equal(escapeHtml(null), '');
  assert.equal(escapeHtml(5), '5');
});

test('normalizeItem maps the build JSON shape and defaults missing fields', () => {
  const n = normalizeItem({ question: 'Q', difficulty: 'Basic', answer: '<p>a</p>' });
  assert.equal(n.questionText, 'Q');
  assert.equal(n.answerHTML, '<p>a</p>');
  assert.deepEqual([...n.tags], []);
  assert.equal(n.options, null);
  assert.equal(n.correct, null);
  assert.equal(n.href, null);
});

test('isShortcutBlocked rejects modifier combos, key repeat and form controls', () => {
  const ev = (o) => ({ ctrlKey: false, metaKey: false, altKey: false, repeat: false, target: { tagName: 'DIV' }, ...o });
  assert.equal(isShortcutBlocked(ev({})), false);
  assert.equal(isShortcutBlocked(ev({ ctrlKey: true })), true);
  assert.equal(isShortcutBlocked(ev({ metaKey: true })), true);
  assert.equal(isShortcutBlocked(ev({ altKey: true })), true);
  assert.equal(isShortcutBlocked(ev({ repeat: true })), true);
  assert.equal(isShortcutBlocked(ev({ target: { tagName: 'SELECT' } })), true);
  assert.equal(isShortcutBlocked(ev({ target: { tagName: 'DIV', isContentEditable: true } })), true);
});

test('mcqOptionHtml emits data attributes, no inline handlers, and aria-pressed only for toggles', () => {
  const o = { letter: 'B', html: 'two <code>x</code>' };
  const oneShot = mcqOptionHtml(o, { action: 'choose' });
  assert.ok(oneShot.includes('data-action="choose"') && oneShot.includes('data-letter="B"'));
  assert.ok(!oneShot.includes('aria-pressed') && !oneShot.includes('onclick'));
  assert.ok(mcqOptionHtml(o, { action: 'answer', selected: true }).includes('aria-pressed="true"'));
  assert.ok(mcqOptionHtml(o, { action: 'answer', selected: false }).includes('aria-pressed="false"'));
  assert.ok(mcqOptionHtml(o, { disabled: true, extraClass: 'wrong' }).includes('disabled'));
});

test('difficultyBadgeHtml defaults to Basic and escapes', () => {
  assert.ok(difficultyBadgeHtml().includes('basic'));
  assert.ok(!difficultyBadgeHtml('<b>').includes('<b>'));
});

// Browser smoke test of the built site (run `npm run build` first; `npm run
// test:site` does both). Loads the generated pages in jsdom with scripts
// enabled and drives the quiz and assessment through their real click and
// keyboard paths: dialog semantics, focus, inert background, shortcuts,
// filters, scoring, sampling balance and history escaping. Counts come from
// the generated data files, so content changes do not break it.
import { JSDOM } from 'jsdom';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';

const SITE = fileURLToPath(new URL('../_site', import.meta.url));

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0]);
  try {
    const body = readFileSync(path.join(SITE, u));
    const type = u.endsWith('.js') ? 'text/javascript' : u.endsWith('.css') ? 'text/css' : u.endsWith('.json') ? 'application/json' : 'text/html';
    res.writeHead(200, { 'content-type': type }); res.end(body);
  } catch { res.writeHead(404); res.end('nf'); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const ORIGIN = `http://127.0.0.1:${server.address().port}/`;

const quizTotal = JSON.parse(readFileSync(path.join(SITE, 'quiz-data.json'), 'utf8')).length;
const mcqTotal = JSON.parse(readFileSync(path.join(SITE, 'assessment-data.json'), 'utf8')).length;
const sectionQuestions = (readFileSync(path.join(SITE, '02_interview_bank/01-naive-rag.html'), 'utf8').match(/<h2 id="q\d+/g) || []).length;
const fails = [];
const check = (cond, msg) => { if (!cond) { fails.push(msg); console.log('FAIL', msg); } else console.log('ok  ', msg); };
const tick = (ms = 30) => new Promise((r) => setTimeout(r, ms));

async function load(file) {
  const html = readFileSync(path.join(SITE, file), 'utf8');
  const dom = new JSDOM(html, {
    url: ORIGIN + file,
    runScripts: 'dangerously',
    resources: 'usable',
    pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = async (href) => {
        const r = await fetch(new URL(href, ORIGIN + file));
        return { ok: r.ok, status: r.status, json: () => r.json() };
      };
      w.confirm = () => true;
      w.scrollTo = () => {};
      w.addEventListener('error', (e) => fails.push('page error: ' + e.message));
    },
  });
  // wait for external scripts + async init
  for (let i = 0; i < 100; i++) { await tick(30); if (dom.window.document.readyState === 'complete') break; }
  await tick(300);
  return dom;
}
const click = (w, el) => el.dispatchEvent(new w.MouseEvent('click', { bubbles: true, cancelable: true }));
const key = (w, k, extra = {}) => w.document.dispatchEvent(new w.KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...extra }));

// ---------------- quiz.html ----------------
{
  const dom = await load('quiz.html');
  const w = dom.window, d = w.document;
  const count = d.getElementById('quiz-match-count').textContent;
  check(count === `${quizTotal.toLocaleString('en-US')} questions match`, `quiz.html loads all ${quizTotal} questions from JSON (` + count + ')');
  click(w, d.querySelector('[data-action="filter"][data-value="Advanced"]'));
  check(/questions match/.test(d.getElementById('quiz-match-count').textContent) && d.getElementById('quiz-match-count').textContent !== count, 'difficulty filter updates the count');
  check(d.querySelector('[data-action="filter"][data-value="Advanced"]').getAttribute('aria-pressed') === 'true', 'filter button pressed state');
  click(w, d.querySelector('[data-action="type"][data-value="MCQ"]'));
  click(w, d.getElementById('quiz-start-btn'));
  const panel = d.getElementById('quiz-panel');
  check(panel.classList.contains('visible') && panel.getAttribute('role') === 'dialog' && panel.getAttribute('aria-modal') === 'true', 'panel opens as a modal dialog');
  check(d.getElementById('main-content').inert === true, 'background is inert while the dialog is open');
  check(d.activeElement && d.activeElement.classList.contains('quiz-title'), 'focus moves to the question title');
  check(panel.querySelectorAll('.mcq-option').length >= 3, 'MCQ options rendered');
  // Ctrl+C must not answer
  key(w, 'c', { ctrlKey: true });
  check(!panel.querySelector('.mcq-option.correct'), 'Ctrl+C does not select an option');
  key(w, 'a');
  check(!!panel.querySelector('.mcq-option.correct'), 'plain letter key answers the MCQ');
  check(d.getElementById('mcq-result').getAttribute('aria-live') === 'polite', 'result is an aria-live region');
  check(d.getElementById('answer-container').hidden === false, 'answer revealed after choosing');
  // changing a filter mid-session restarts scoring
  click(w, d.querySelector('[data-action="filter"][data-value="Basic"]'));
  check(/Question 1 \//.test(panel.querySelector('.quiz-title').textContent), 'filter change mid-session restarts at question 1');
  key(w, 'Escape');
  check(!panel.classList.contains('visible') && d.getElementById('main-content').inert === false, 'Escape closes the panel and removes inert');
  check(d.activeElement === d.getElementById('quiz-start-btn'), 'focus returns to the Start button');
  click(w, d.querySelector('[data-action="reset"]'));
  check(d.getElementById('quiz-match-count').textContent === count, 'reset filters restores the full count');
  // flashcard flow
  click(w, d.querySelector('[data-action="type"][data-value="Flashcard"]'));
  click(w, d.getElementById('quiz-start-btn'));
  click(w, panel.querySelector('[data-action="show"]'));
  check(!!panel.querySelector('[data-action="got"]'), 'flashcard: Show Answer reveals Got it / Review again');
  check(!/Show Answer/.test(panel.querySelector('.quiz-answer').textContent), 'revealed answer does not contain the <summary> text');
  w.close();
}

// ---------------- section page (scrape path) ----------------
{
  const dom = await load('02_interview_bank/01-naive-rag.html');
  const w = dom.window, d = w.document;
  check(d.getElementById('quiz-match-count').textContent === `${sectionQuestions} questions match`, `section page scrapes its ${sectionQuestions} questions: ` + d.getElementById('quiz-match-count').textContent);
  click(w, d.getElementById('quiz-start-btn'));
  check(d.getElementById('quiz-panel').classList.contains('visible'), 'section page quiz starts');
  w.close();
}

// ---------------- assessment.html ----------------
{
  const dom = await load('assessment.html');
  const w = dom.window, d = w.document;
  const info = d.getElementById('assess-match-count').textContent;
  check(info.startsWith(`${mcqTotal} MCQs available · 25 will be asked`), `assessment loads all ${mcqTotal} MCQs: ` + info);
  click(w, d.querySelector('[data-action="assess-option"][data-opt="count"][data-value="50"]'));
  check(/50 will be asked/.test(d.getElementById('assess-match-count').textContent), 'count option updates');
  click(w, d.getElementById('assess-start-btn'));
  const panel = d.getElementById('quiz-panel');
  check(panel.classList.contains('visible') && panel.getAttribute('aria-modal') === 'true', 'assessment opens as dialog');
  check(panel.querySelectorAll('.assess-dot').length === 50, '50 navigator dots');
  key(w, 'b');
  check(panel.querySelector('.mcq-option[aria-pressed="true"]')?.dataset.letter === 'B', 'letter key selects option');
  check(d.activeElement === panel.querySelector('.mcq-option[data-letter="B"]'), 'focus stays on the toggled option after re-render');
  check(panel.querySelector('.assess-dot.answered')?.getAttribute('aria-label').includes('(answered)'), 'answered dot has an accessible label');
  check(panel.querySelector('.assess-dot.current')?.getAttribute('aria-current') === 'true', 'current dot has aria-current');
  key(w, 'c', { ctrlKey: true });
  check(panel.querySelector('.mcq-option[aria-pressed="true"]').dataset.letter === 'B', 'Ctrl+C does not change the answer');
  key(w, 'ArrowRight');
  check(/Question 2 \//.test(panel.querySelector('.quiz-title').textContent), 'ArrowRight goes to question 2');
  // difficulty mix of the 50-question set
  click(w, panel.querySelector('[data-action="submit"]'));
  const results = d.getElementById('assessment-results');
  check(!results.classList.contains('is-hidden'), 'results shown after submit');
  const rows = [...results.querySelectorAll('table')[0].querySelectorAll('tbody tr')].map((tr) => [...tr.children].map((td) => td.textContent));
  console.log('difficulty rows:', JSON.stringify(rows));
  check(rows.length === 3 && rows.every((r) => Number(r[1].split('/')[1]) >= 10), 'a 50-question set contains all three difficulties (each >= 10)');
  check(d.activeElement === results.querySelector('h2'), 'focus moves to the results heading');
  const hist = d.getElementById('assessment-history').textContent;
  check(/Past attempts/.test(hist) && /Best:/.test(hist), 'history rendered');
  // planted history must be escaped / ignored
  w.localStorage.setItem('rag-interview-system:assessment-history-v1', JSON.stringify([{ pct: 50, total: 10, correct: 5, at: '2026-01-01', difficulty: '<img src=x onerror=alert(1)>', group: 'All' }]));
  w.document.querySelector('[data-action="back"]').click();
  // re-render history via a fresh submit path is heavy; render directly:
  w.eval('renderHistory()');
  check(!d.getElementById('assessment-history').querySelector('img'), 'planted HTML in history is escaped');
  w.close();
}

console.log(fails.length ? `\n${fails.length} FAILURE(S)` : '\nALL SMOKE CHECKS PASSED');
server.close();
process.exit(fails.length ? 1 : 0);

// Runs inline on assessment.html (see scripts/build_site.mjs's
// buildAssessmentPage), so this is a plain classic script, not an ES module.
// The top half is DOM-free (only Math/JSON/Date) so scripts/lib/assessment.test.mjs
// can load it with node:vm; everything that touches `document` lives below the
// "DOM" marker and only runs via initAssessment().
//
// DIFFICULTIES, isMcqItem and shuffleArray intentionally mirror scripts/quiz.js
// (which can't be imported either); keep them in sync.
const DIFFICULTIES = ['Basic', 'Intermediate', 'Advanced'];
const HISTORY_KEY = 'rag-assessment-history-v1';
const HISTORY_MAX = 20;

const READINESS_BANDS = [
  { min: 90, level: 'Interview ready', cls: 'ready', text: 'Consistently correct across sections. Stay sharp with Advanced sets and the failure-modes bank.' },
  { min: 75, level: 'Strong', cls: 'strong', text: 'Interview-competitive. Tighten the sections listed below and practise Advanced and Scenario questions.' },
  { min: 50, level: 'Getting there', cls: 'getting-there', text: 'Fundamentals are in place. Drill the weak sections below and the Intermediate tier.' },
  { min: 0, level: 'Needs work', cls: 'needs-work', text: 'Core concepts are not solid yet. Work through the concepts section and the Basic questions before retesting.' },
];

function isMcqItem(q) {
  return !!q && Array.isArray(q.tags) && q.tags.includes('MCQ') && Array.isArray(q.options) && q.options.length > 0 && !!q.correct;
}

function shuffleArray(arr, rng = Math.random) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function filterPool(pool, { difficulty = 'All', group = 'All' } = {}) {
  let r = pool.filter(isMcqItem);
  if (difficulty !== 'All') r = r.filter(q => q.difficulty === difficulty);
  if (group !== 'All') r = r.filter(q => q.sectionGroup === group);
  return r;
}

// Stratified pick: round-robin across sections (so no single architecture
// dominates), preferring a rotating difficulty within each section.
function sampleQuestions(pool, count, rng = Math.random) {
  if (count === 'All' || count >= pool.length) return shuffleArray(pool, rng);
  const buckets = new Map();
  for (const q of shuffleArray(pool, rng)) {
    const key = q.section || '';
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(q);
  }
  const order = shuffleArray(Array.from(buckets.values()), rng);
  const picked = [];
  let round = 0;
  while (picked.length < count) {
    let progressed = false;
    for (const bucket of order) {
      if (picked.length >= count) break;
      if (bucket.length === 0) continue;
      const want = DIFFICULTIES[round % DIFFICULTIES.length];
      let i = bucket.findIndex(q => q.difficulty === want);
      if (i < 0) i = 0;
      picked.push(bucket.splice(i, 1)[0]);
      progressed = true;
    }
    if (!progressed) break;
    round++;
  }
  return shuffleArray(picked, rng);
}

function sectionPageHref(href) {
  if (!href) return null;
  const i = href.indexOf('#');
  return i === -1 ? href : href.slice(0, i);
}

function readinessFor(pct) {
  return READINESS_BANDS.find(b => pct >= b.min) || READINESS_BANDS[READINESS_BANDS.length - 1];
}

function scoreAttempt(questions, answers) {
  const total = questions.length;
  let correct = 0, unanswered = 0;
  const byDifficulty = {};
  DIFFICULTIES.forEach(d => { byDifficulty[d] = { correct: 0, total: 0 }; });
  const sections = new Map();
  const missed = [];
  questions.forEach((q, index) => {
    const chosen = answers[index] || null;
    const ok = chosen !== null && chosen === q.correct;
    if (ok) correct++;
    if (chosen === null) unanswered++;
    if (!byDifficulty[q.difficulty]) byDifficulty[q.difficulty] = { correct: 0, total: 0 };
    byDifficulty[q.difficulty].total++;
    if (ok) byDifficulty[q.difficulty].correct++;
    const key = q.section || 'Unknown';
    if (!sections.has(key)) {
      sections.set(key, { section: key, group: q.sectionGroup || null, href: sectionPageHref(q.href), correct: 0, total: 0 });
    }
    const s = sections.get(key);
    s.total++;
    if (ok) s.correct++;
    if (!ok) missed.push({ index, question: q, chosen, correct: q.correct });
  });
  const bySection = Array.from(sections.values())
    .map(s => ({ ...s, pct: s.total ? Math.round((s.correct / s.total) * 100) : 0 }))
    .sort((a, b) => a.pct - b.pct || a.section.localeCompare(b.section));
  return {
    total,
    correct,
    wrong: total - correct - unanswered,
    unanswered,
    pct: total ? Math.round((correct / total) * 100) : 0,
    byDifficulty,
    bySection,
    missed,
  };
}

function timerSeconds(count, secondsPerQuestion) {
  return secondsPerQuestion > 0 ? count * secondsPerQuestion : 0;
}

function formatClock(sec) {
  const s = Math.max(0, Math.floor(sec));
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

function loadHistory(storage) {
  if (!storage) return [];
  try {
    const parsed = JSON.parse(storage.getItem(HISTORY_KEY));
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(e => e && typeof e === 'object' && typeof e.pct === 'number');
  } catch (e) {
    return [];
  }
}

function saveAttempt(storage, entry) {
  const next = [entry, ...loadHistory(storage)].slice(0, HISTORY_MAX);
  if (storage) {
    try { storage.setItem(HISTORY_KEY, JSON.stringify(next)); } catch (e) { /* quota / private mode */ }
  }
  return next;
}

function clearHistory(storage) {
  if (!storage) return;
  try { storage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
}

function bestScore(history) {
  return history.length ? history.reduce((b, e) => (e.pct > b.pct ? e : b)) : null;
}

// ---------------------------------------------------------------- DOM ----

const state = {
  phase: 'setup',
  config: { count: 25, difficulty: 'All', group: 'All', secondsPerQuestion: 0 },
  pool: [],
  questions: [],
  answers: {},
  currentIndex: 0,
  startedAt: null,
  deadline: null,
  timerId: null,
  timedOut: false,
  result: null,
};

function $(id) { return document.getElementById(id); }

function loadAssessmentQuestions() {
  const dataEl = $('assessment-data');
  if (!dataEl) return [];
  return JSON.parse(dataEl.textContent).map(item => ({
    questionText: item.question,
    difficulty: item.difficulty,
    tags: item.tags || [],
    answerHTML: item.answer,
    options: item.options || null,
    correct: item.correct || null,
    section: item.section || null,
    sectionGroup: item.sectionGroup || null,
    href: item.href || null,
  }));
}

function getStorage() {
  try { return window.localStorage || null; } catch (e) { return null; }
}

function setPressed(selector, btn) {
  document.querySelectorAll(selector).forEach(b => {
    const on = b === btn;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
}

function plannedCount() {
  const available = filterPool(state.pool, state.config).length;
  const c = state.config.count;
  return { available, asked: c === 'All' ? available : Math.min(c, available) };
}

function updateAssessMatchCount() {
  const { available, asked } = plannedCount();
  const el = $('assess-match-count');
  const btn = $('assess-start-btn');
  if (el) {
    const limit = timerSeconds(asked, state.config.secondsPerQuestion);
    el.textContent = available === 0
      ? 'No questions match these options'
      : `${available.toLocaleString('en-US')} MCQs available · ${asked} will be asked` + (limit ? ` · ${formatClock(limit)} limit` : '');
    el.classList.toggle('empty', available === 0);
  }
  if (btn) btn.disabled = available === 0;
}

function setAssessOption(name, value, btn) {
  state.config[name] = value;
  if (btn) setPressed(`.assess-btn[data-opt="${name}"]`, btn);
  updateAssessMatchCount();
}

function startAssessment() {
  const pool = filterPool(state.pool, state.config);
  if (pool.length === 0) return;
  state.questions = sampleQuestions(pool, state.config.count);
  state.answers = {};
  state.currentIndex = 0;
  state.startedAt = Date.now();
  state.timedOut = false;
  state.phase = 'exam';
  $('main-content').classList.add('hidden');
  $('assessment-results').classList.add('is-hidden');
  $('quiz-overlay').classList.add('visible');
  $('quiz-panel').classList.add('visible');
  startTimer();
  renderExamQuestion();
}

function startTimer() {
  stopTimer();
  const total = timerSeconds(state.questions.length, state.config.secondsPerQuestion);
  if (!total) { state.deadline = null; return; }
  state.deadline = Date.now() + total * 1000;
  state.timerId = setInterval(tickTimer, 500);
}

function stopTimer() {
  if (state.timerId !== null) clearInterval(state.timerId);
  state.timerId = null;
}

function updateClock() {
  const el = $('assess-clock');
  if (!el || state.deadline === null) return 1;
  const remaining = Math.max(0, Math.ceil((state.deadline - Date.now()) / 1000));
  el.textContent = '⏱ ' + formatClock(remaining);
  el.classList.toggle('warning', remaining < 60);
  return remaining;
}

function tickTimer() {
  if (state.deadline !== null && Date.now() >= state.deadline && state.phase === 'exam') {
    state.timedOut = true;
    submitAssessment({ reason: 'timeout' });
    return;
  }
  updateClock();
}

function renderExamQuestion() {
  const q = state.questions[state.currentIndex];
  const n = state.questions.length;
  const answeredCount = Object.keys(state.answers).length;
  const chosen = state.answers[state.currentIndex] || null;
  const dots = state.questions.map((_, i) => {
    const cls = ['assess-dot', state.answers[i] ? 'answered' : '', i === state.currentIndex ? 'current' : ''].join(' ').trim();
    return `<button class="${cls}" aria-label="Go to question ${i + 1}" onclick="goToQuestion(${i})">${i + 1}</button>`;
  }).join('');
  const options = q.options.map(o =>
    `<button class="mcq-option${chosen === o.letter ? ' selected' : ''}" aria-pressed="${chosen === o.letter}" onclick="selectAnswer('${o.letter}')"><span class="mcq-letter">${o.letter}.</span> ${o.html}</button>`
  ).join('');
  $('quiz-panel').classList.add('assess-panel');
  $('quiz-panel').innerHTML = `
    <div class="quiz-header">
      <div>
        <strong>Question ${state.currentIndex + 1} / ${n}</strong>
        <span class="difficulty-badge ${q.difficulty.toLowerCase()}">${q.difficulty}</span>
      </div>
      <span id="assess-clock" class="assess-clock"></span>
      <button onclick="exitAssessment()" aria-label="Exit assessment" style="border: none; background: none; cursor: pointer; font-size: 1.2em;">✕</button>
    </div>
    <div class="quiz-progress"><div class="quiz-progress-bar" style="width: ${(answeredCount / n) * 100}%"></div></div>
    <div class="assess-answered">${answeredCount} of ${n} answered</div>
    <div class="assess-nav">${dots}</div>
    ${q.section ? `<div class="quiz-section">${q.section}</div>` : ''}
    <div class="quiz-question">${q.questionText}</div>
    <div class="mcq-options">${options}</div>
    <div class="quiz-buttons">
      <button onclick="goToQuestion(${state.currentIndex - 1})" ${state.currentIndex === 0 ? 'disabled' : ''}>← Previous</button>
      <button onclick="goToQuestion(${state.currentIndex + 1})" ${state.currentIndex === n - 1 ? 'disabled' : ''}>Next →</button>
      <button class="primary" onclick="submitAssessment({ reason: 'user' })">Submit</button>
    </div>`;
  updateClock();
}

function goToQuestion(i) {
  if (state.phase !== 'exam' || i < 0 || i >= state.questions.length) return;
  state.currentIndex = i;
  renderExamQuestion();
}

function selectAnswer(letter) {
  if (state.phase !== 'exam') return;
  if (state.answers[state.currentIndex] === letter) delete state.answers[state.currentIndex];
  else state.answers[state.currentIndex] = letter;
  renderExamQuestion();
}

function closeExamPanel() {
  stopTimer();
  $('quiz-panel').classList.remove('visible');
  $('quiz-overlay').classList.remove('visible');
}

function exitAssessment() {
  if (Object.keys(state.answers).length > 0 && !confirm('Exit the assessment? Your answers will be lost.')) return;
  closeExamPanel();
  backToSetup();
}

function summarizeAttempt(result) {
  return {
    at: new Date().toISOString(),
    count: state.config.count,
    difficulty: state.config.difficulty,
    group: state.config.group,
    secondsPerQuestion: state.config.secondsPerQuestion,
    total: result.total,
    correct: result.correct,
    pct: result.pct,
    level: readinessFor(result.pct).level,
    durationSec: Math.round((Date.now() - state.startedAt) / 1000),
    timedOut: state.timedOut,
  };
}

function submitAssessment({ reason } = {}) {
  if (state.phase !== 'exam') return;
  const unanswered = state.questions.length - Object.keys(state.answers).length;
  if (reason === 'user' && unanswered > 0 &&
      !confirm(`${unanswered} unanswered. Submit anyway? Unanswered questions count as wrong.`)) return;
  state.phase = 'results';
  closeExamPanel();
  state.result = scoreAttempt(state.questions, state.answers);
  const entry = summarizeAttempt(state.result);
  saveAttempt(getStorage(), entry);
  renderResults(entry);
  renderHistory();
  $('assessment-results').classList.remove('is-hidden');
  window.scrollTo(0, 0);
}

function backToSetup() {
  stopTimer();
  state.phase = 'setup';
  state.result = null;
  $('assessment-results').classList.add('is-hidden');
  $('main-content').classList.remove('hidden');
  $('quiz-panel').classList.remove('assess-panel');
  window.scrollTo(0, 0);
}

function pctClass(pct) {
  return pct >= 75 ? 'good' : pct >= 50 ? 'ok' : 'bad';
}

function renderResults(entry) {
  const r = state.result;
  const band = readinessFor(r.pct);
  const diffRows = DIFFICULTIES.filter(d => r.byDifficulty[d] && r.byDifficulty[d].total > 0).map(d => {
    const v = r.byDifficulty[d];
    const p = Math.round((v.correct / v.total) * 100);
    return `<tr><td>${d}</td><td>${v.correct} / ${v.total}</td><td class="pct ${pctClass(p)}">${p}%</td></tr>`;
  }).join('');
  const sectionRows = r.bySection.map(s => {
    const name = s.href ? `<a href="${s.href}">${s.section}</a>` : s.section;
    return `<tr><td>${name}</td><td>${s.correct} / ${s.total}</td><td class="pct ${pctClass(s.pct)}">${s.pct}%</td></tr>`;
  }).join('');
  const review = r.missed.map(m => {
    const q = m.question;
    const opts = q.options.map(o => {
      const cls = o.letter === q.correct ? ' correct' : o.letter === m.chosen ? ' wrong' : '';
      return `<button class="mcq-option${cls}" disabled><span class="mcq-letter">${o.letter}.</span> ${o.html}</button>`;
    }).join('');
    const status = m.chosen ? `You chose ${m.chosen}. Correct answer: ${m.correct}.` : `Not answered. Correct answer: ${m.correct}.`;
    const open = q.href ? `<a class="quiz-open" href="${q.href}">Open in section page ↗</a>` : '';
    return `<div class="review-item">
      <div class="quiz-section">Q${m.index + 1} · ${q.section || ''} <span class="difficulty-badge ${q.difficulty.toLowerCase()}">${q.difficulty}</span></div>
      <div class="quiz-question">${q.questionText}</div>
      <div class="mcq-options">${opts}</div>
      <div class="mcq-result">${status}</div>
      <details><summary>Show full answer</summary><div class="quiz-answer">${q.answerHTML}</div></details>
      ${open}
    </div>`;
  }).join('');
  $('assessment-results').innerHTML = `
    <div class="quiz-summary">
      <h2>Assessment complete</h2>
      <div class="quiz-score">${r.correct} / ${r.total} · ${r.pct}%</div>
      <p class="lead">Unanswered: ${r.unanswered} (counted as wrong) · Time: ${formatClock(entry.durationSec)}${state.timedOut ? ' · time ran out' : ''}</p>
    </div>
    <div class="readiness ${band.cls}"><h3>${band.level}</h3><p>${band.text}</p></div>
    <h3>By difficulty</h3>
    <table class="assess-table"><thead><tr><th>Difficulty</th><th>Correct</th><th>Score</th></tr></thead><tbody>${diffRows}</tbody></table>
    <h3>By section (weakest first)</h3>
    <table class="assess-table"><thead><tr><th>Section</th><th>Correct</th><th>Score</th></tr></thead><tbody>${sectionRows}</tbody></table>
    <h3>Review (${r.missed.length})</h3>
    ${review || '<p class="history-empty">Perfect score. Nothing to review.</p>'}
    <p class="assess-actions">
      <button class="assess-again" onclick="backToSetup()">Take another assessment</button>
      <a href="quiz.html">Back to quiz</a> · <a href="index.html">Back to index</a>
    </p>`;
}

function renderHistory() {
  const el = $('assessment-history');
  if (!el) return;
  const storage = getStorage();
  if (!storage) {
    el.innerHTML = '<h2>Past attempts</h2><p class="history-empty">History is unavailable in this browser (private mode or storage disabled).</p>';
    return;
  }
  const history = loadHistory(storage);
  if (history.length === 0) {
    el.innerHTML = '<h2>Past attempts</h2><p class="history-empty">No attempts yet. Results are saved in this browser only.</p>';
    return;
  }
  const best = bestScore(history);
  const rows = history.map(h => {
    const band = readinessFor(h.pct);
    const when = new Date(h.at).toLocaleString();
    const diff = h.difficulty && h.difficulty !== 'All' ? ` · ${h.difficulty}` : '';
    const grp = h.group && h.group !== 'All' ? ` · ${h.group}` : '';
    const timer = h.secondsPerQuestion ? ` · ${h.secondsPerQuestion} s/q` : '';
    return `<li><span>${when} · ${h.total} Q${diff}${grp}${timer}</span><span>${h.correct}/${h.total} (${h.pct}%) <span class="history-level readiness ${band.cls}">${band.level}</span></span></li>`;
  }).join('');
  el.innerHTML = `
    <div class="history-head"><h2>Past attempts</h2>
      <span>Best: ${best.pct}% · <button class="link-btn" onclick="onClearHistory()">Clear history</button></span></div>
    <ul class="quiz-toc">${rows}</ul>`;
}

function onClearHistory() {
  if (!confirm('Clear all saved assessment attempts from this browser?')) return;
  clearHistory(getStorage());
  renderHistory();
}

function initAssessment() {
  state.pool = loadAssessmentQuestions();
  updateAssessMatchCount();
  renderHistory();
  document.addEventListener('keydown', (e) => {
    if (state.phase !== 'exam') return;
    if (e.target && /^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
    const q = state.questions[state.currentIndex];
    if (e.key === 'ArrowRight') { goToQuestion(state.currentIndex + 1); return; }
    if (e.key === 'ArrowLeft') { goToQuestion(state.currentIndex - 1); return; }
    const letter = e.key.length === 1 ? e.key.toUpperCase() : '';
    if (q && letter && q.options.some(o => o.letter === letter)) selectAnswer(letter);
  });
}

if (typeof document !== 'undefined') initAssessment();

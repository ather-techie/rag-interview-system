// Shared browser helpers for the practice quiz (scripts/quiz.js) and the
// self-assessment (scripts/assessment.js). Served as assets/site/quiz-common.js
// by scripts/build_site.mjs and loaded as a plain classic script right after a
// small inline prelude that defines DIFFICULTIES, TAGS, ASSESSMENT_HREF,
// QUIZ_PAGE_GROUP and QUIZ_DATA_HREF (see scripts/lib/prelude.mjs).
//
// Everything above the "DOM" marker is DOM-free so scripts/lib/*.test.mjs can
// evaluate it with node:vm.

// An item only renders as a multiple-choice card if it's tagged [MCQ] *and*
// carries a usable options list and correct letter -- otherwise (e.g. a
// scrape that failed to find the option <ul>) it degrades to a flashcard
// rather than rendering a broken quiz.
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

// Escapes a value for safe insertion into HTML text or a double-quoted attribute.
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Maps one item of the build's quiz JSON (quiz-data.json / assessment-data.json,
// produced from extractQuizItems in scripts/build_site.mjs) to the shape the
// browser code uses. `question`, `answer` and option `html` are repo-authored,
// already-rendered HTML.
function normalizeItem(item) {
  return {
    questionText: item.question,
    difficulty: item.difficulty,
    tags: item.tags || [],
    answerHTML: item.answer,
    options: item.options || null,
    correct: item.correct || null,
    section: item.section || null,
    sectionGroup: item.sectionGroup || null,
    href: item.href || null,
  };
}

// True when a keydown must NOT be treated as a quiz shortcut: a modifier
// combo (Ctrl+C has to copy, not pick option C), key auto-repeat, or a
// focused form control.
function isShortcutBlocked(e) {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return true;
  const t = e.target;
  return !!(t && (/^(SELECT|INPUT|TEXTAREA)$/.test(t.tagName) || t.isContentEditable));
}

function difficultyBadgeHtml(difficulty) {
  const d = difficulty || 'Basic';
  return `<span class="difficulty-badge ${escapeHtml(d.toLowerCase())}">${escapeHtml(d)}</span>`;
}

function closeButtonHtml(action, label) {
  return `<button type="button" class="quiz-close" data-action="${escapeHtml(action)}" aria-label="${escapeHtml(label)}" title="${escapeHtml(label)}">✕</button>`;
}

// One MCQ option button. `selected` is null for one-shot (practice) options
// and a boolean for toggle (assessment) options, which also get aria-pressed.
function mcqOptionHtml(option, { action = null, selected = null, disabled = false, extraClass = '' } = {}) {
  const cls = ['mcq-option', selected ? 'selected' : '', extraClass].filter(Boolean).join(' ');
  const attrs = [
    'type="button"',
    `class="${cls}"`,
    `data-letter="${escapeHtml(option.letter)}"`,
    action ? `data-action="${escapeHtml(action)}"` : '',
    selected === null ? '' : `aria-pressed="${selected ? 'true' : 'false'}"`,
    disabled ? 'disabled' : '',
  ].filter(Boolean).join(' ');
  return `<button ${attrs}><span class="mcq-letter">${escapeHtml(option.letter)}.</span> ${option.html}</button>`;
}

// ---------------------------------------------------------------- DOM ----

// Marks `btn` as the pressed member of a segmented control (the other
// buttons matching `selector` are released). Keeps the .active class and
// aria-pressed in step so sighted and screen-reader users see the same state.
function setPressed(selector, btn) {
  document.querySelectorAll(selector).forEach(b => {
    const on = b === btn;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
}

// Loads the page's question dataset: an inline <script type="application/json">
// if the page carries one, else fetched from QUIZ_DATA_HREF. Resolves to null
// when the page has neither, so the caller can fall back to scraping itself.
async function loadQuizData(inlineId) {
  const inline = inlineId ? document.getElementById(inlineId) : null;
  if (inline) return JSON.parse(inline.textContent).map(normalizeItem);
  if (typeof QUIZ_DATA_HREF === 'string' && QUIZ_DATA_HREF) {
    const res = await fetch(QUIZ_DATA_HREF);
    if (!res.ok) throw new Error(`Could not load ${QUIZ_DATA_HREF} (HTTP ${res.status})`);
    return (await res.json()).map(normalizeItem);
  }
  return null;
}

// Buttons carry data-action="name" (click) and form controls data-change="name"
// (change). One listener per event type dispatches to handlers[name](el, event),
// so the generated markup needs no inline onclick= attributes and the scripts
// expose no globals for the markup to call.
function bindActions(handlers) {
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-action]');
    if (!el || el.disabled) return;
    const fn = handlers[el.dataset.action];
    if (!fn) return;
    e.preventDefault();
    quizPanel.lastTrigger = el; // Safari does not focus a clicked button, so remember it for focus restore
    fn(el, e);
  });
  document.addEventListener('change', (e) => {
    const el = e.target.closest('[data-change]');
    if (!el) return;
    const fn = handlers[el.dataset.change];
    if (fn) fn(el, e);
  });
}

// #quiz-panel is shown as a modal dialog: every other child of <body> is made
// inert (so Tab and screen readers stay inside), and focus returns to
// whatever opened it on close. Escape is handled by each page's own keydown
// handler because the two pages confirm differently.
const quizPanel = { opener: null, lastTrigger: null, inerted: [] };

function openQuizPanel(label) {
  const panel = document.getElementById('quiz-panel');
  const overlay = document.getElementById('quiz-overlay');
  if (!panel) return;
  const active = document.activeElement;
  quizPanel.opener = active && active !== document.body ? active : quizPanel.lastTrigger;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  if (label) panel.setAttribute('aria-label', label);
  panel.classList.add('visible');
  if (overlay) overlay.classList.add('visible');
  const main = document.getElementById('main-content');
  if (main) main.classList.add('hidden');
  quizPanel.inerted = Array.from(document.body.children).filter(
    (el) => el !== panel && el !== overlay && el.tagName !== 'SCRIPT' && !el.inert
  );
  quizPanel.inerted.forEach((el) => { el.inert = true; });
}

function closeQuizPanel({ restoreMain = true, restoreFocus = true } = {}) {
  const panel = document.getElementById('quiz-panel');
  const overlay = document.getElementById('quiz-overlay');
  quizPanel.inerted.forEach((el) => { el.inert = false; });
  quizPanel.inerted = [];
  if (panel) panel.classList.remove('visible');
  if (overlay) overlay.classList.remove('visible');
  if (restoreMain) {
    const main = document.getElementById('main-content');
    if (main) main.classList.remove('hidden');
  }
  const opener = quizPanel.opener;
  quizPanel.opener = null;
  if (restoreFocus && opener && typeof opener.focus === 'function' && document.contains(opener)) opener.focus();
}

// Moves focus to the panel's title (or the panel itself) after a render, so
// keyboard and screen-reader users land on the new question rather than on
// <body>.
function focusQuizPanel() {
  const panel = document.getElementById('quiz-panel');
  if (!panel) return;
  const target = panel.querySelector('.quiz-title') || panel;
  if (!target.hasAttribute('tabindex')) target.tabIndex = -1;
  target.focus({ preventScroll: true });
}

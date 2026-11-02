// Practice quiz. Served as assets/site/quiz.js and loaded after the build's
// inline prelude (DIFFICULTIES, TAGS, ASSESSMENT_HREF, QUIZ_PAGE_GROUP,
// QUIZ_DATA_HREF) and assets/site/quiz-common.js. A plain classic script, not
// an ES module. Section pages carry no dataset and scrape their own DOM;
// quiz.html fetches quiz-data.json.

function scrapeFromDom() {
  const items = [];
  document.querySelectorAll('h2').forEach(h2 => {
    if (!/^Q\d+\./.test(h2.textContent.trim())) return;
    // Collect every `[Tag]` code span in the heading (not just the first --
    // a question can carry a difficulty tag *and* a Scenario tag, and some
    // headings also contain unrelated backticked tokens mid-sentence, e.g.
    // Self-RAG's `[IsSup]` reflection token).
    const codeTokens = Array.from(h2.querySelectorAll('code'))
      .map(el => el.textContent.match(/^\[([A-Za-z]+)\]$/))
      .filter(Boolean)
      .map(m => m[1]);
    const difficulty = codeTokens.find(t => DIFFICULTIES.includes(t)) || 'Basic';
    const tags = codeTokens.filter(t => TAGS.includes(t));
    const questionText = h2.textContent
      .replace(/^Q\d+\.\s*/, '')
      .replace(/(\s*\[[A-Za-z]+\])+\s*$/, '')
      .trim();
    // For [MCQ] questions, an option <ul> sits between the heading and
    // <details> -- read it off before continuing the walk to <details>.
    let el = h2.nextElementSibling;
    let options = null;
    if (el && el.tagName === 'UL' && tags.includes('MCQ')) {
      options = Array.from(el.children).map(li => {
        const m = li.textContent.match(/^\s*([A-Z])\.\s/);
        if (!m) return null;
        return { letter: m[1], html: li.innerHTML.replace(/^(\s*)([A-Z])\.\s*/, '$1') };
      }).filter(Boolean);
      el = el.nextElementSibling;
    }
    while (el && el.tagName !== 'DETAILS') el = el.nextElementSibling;
    let answerHTML = '<p>No answer found.</p>';
    let correct = null;
    if (el) {
      const clone = el.cloneNode(true);
      const summary = clone.querySelector('summary');
      if (summary) summary.remove();
      answerHTML = clone.innerHTML;
      const m = el.textContent.match(/\bCorrect:\s*([A-Z])\./);
      correct = m ? m[1] : null;
    }
    items.push({ questionText, difficulty, tags, answerHTML, options, correct, section: null, sectionGroup: null, href: null });
  });
  return items;
}

let allQuestions = [];
let filtered = [], currentIndex = 0, correctCount = 0, reviewCount = 0, skippedCount = 0;
let difficultyFilter = 'All';
let scenarioOnly = false;
let typeFilter = 'All';
let answered = false;

function sessionActive() {
  return document.getElementById('quiz-panel').classList.contains('visible');
}

function applyFilters({ shuffle = true } = {}) {
  let result = allQuestions;
  if (difficultyFilter !== 'All') {
    result = result.filter(q => q.difficulty === difficultyFilter);
  }
  if (scenarioOnly) {
    result = result.filter(q => q.tags.includes('Scenario'));
  }
  if (typeFilter === 'MCQ') {
    result = result.filter(isMcqItem);
  } else if (typeFilter === 'Flashcard') {
    result = result.filter(q => !isMcqItem(q));
  }
  const sectionSelect = document.getElementById('section-select');
  if (sectionSelect && sectionSelect.value !== 'All') {
    result = result.filter(q => q.section === sectionSelect.value);
  }
  const shuffleToggle = document.getElementById('shuffle-toggle');
  if (shuffle && shuffleToggle && shuffleToggle.checked) {
    result = shuffleArray(result);
  }
  return result;
}

// Live "N questions match" readout under the filter bar, plus disabling
// Start when the combination is empty. Counts without shuffling.
function updateMatchCount() {
  const countEl = document.getElementById('quiz-match-count');
  const startBtn = document.getElementById('quiz-start-btn');
  if (!countEl && !startBtn) return;
  const n = applyFilters({ shuffle: false }).length;
  if (countEl) {
    countEl.textContent = n === 0
      ? 'No questions match these filters'
      : n === 1 ? '1 question matches' : `${n.toLocaleString('en-US')} questions match`;
    countEl.classList.toggle('empty', n === 0);
  }
  if (startBtn) startBtn.disabled = n === 0;
}

function resetSession() {
  currentIndex = 0;
  correctCount = 0;
  reviewCount = 0;
  skippedCount = 0;
}

// Every filter control funnels through here. A filter change during a session
// restarts it, so the final score can never exceed the number of questions.
function onFiltersChanged() {
  filtered = applyFilters();
  resetSession();
  if (sessionActive()) renderQuestion();
  updateMatchCount();
}

// Returns every control in the filter bar to its default, then refreshes
// the match count. Does not touch an in-progress quiz session.
function resetFilters() {
  difficultyFilter = 'All';
  typeFilter = 'All';
  scenarioOnly = false;
  const firstDiff = document.querySelector('.filter-btn');
  if (firstDiff) setPressed('.filter-btn', firstDiff);
  const firstType = document.querySelector('.type-btn');
  if (firstType) setPressed('.type-btn', firstType);
  ['scenario-toggle', 'shuffle-toggle'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.checked = false;
  });
  const sectionSelect = document.getElementById('section-select');
  if (sectionSelect) sectionSelect.value = 'All';
  filtered = applyFilters();
  resetSession();
  updateMatchCount();
}

function startQuiz() {
  filtered = applyFilters();
  resetSession();
  openQuizPanel('Practice quiz');
  renderQuestion();
}

function renderQuestion() {
  const panel = document.getElementById('quiz-panel');
  if (filtered.length === 0) {
    panel.innerHTML = `
      <div class="quiz-header">
        <div class="quiz-title"><strong>No questions match these filters</strong></div>
        ${closeButtonHtml('exit', 'Exit quiz')}
      </div>`;
    focusQuizPanel();
    return;
  }
  const q = filtered[currentIndex];
  answered = false;
  const progress = ((currentIndex + 1) / filtered.length * 100);
  const tagBadges = q.tags.map(t => `<span class="tag-badge ${escapeHtml(t.toLowerCase())}">${escapeHtml(t)}</span>`).join('');
  const sectionLine = q.section ? `<div class="quiz-section">${escapeHtml(q.section)}</div>` : '';
  const openLink = (q.section && q.href)
    ? `<a class="quiz-open" href="${escapeHtml(q.href)}">Open in section page ↗</a>`
    : '';

  const mcq = isMcqItem(q);
  const optionsHtml = mcq
    ? `<div class="mcq-options">${q.options.map(o => mcqOptionHtml(o, { action: 'choose' })).join('')}</div>
    <div id="mcq-result" class="mcq-result" role="status" aria-live="polite"></div>`
    : '';
  const buttonsHtml = mcq
    ? `<button type="button" data-action="skip">Skip →</button>`
    : `<button type="button" class="primary" data-action="show">Show Answer</button>
       <button type="button" data-action="skip">Skip →</button>`;

  panel.innerHTML = `
    <div class="quiz-header">
      <div>
        <strong class="quiz-title">Question ${currentIndex + 1} / ${filtered.length}</strong>
        ${difficultyBadgeHtml(q.difficulty)}${tagBadges}
      </div>
      ${closeButtonHtml('exit', 'Exit quiz')}
    </div>
    <div class="quiz-progress">
      <div class="quiz-progress-bar" style="width: ${progress}%"></div>
    </div>
    ${sectionLine}
    <div class="quiz-question">${q.questionText}</div>
    ${optionsHtml}
    <div id="answer-container" hidden>
      <div class="quiz-answer">${q.answerHTML}</div>
      ${openLink}
    </div>
    <div class="quiz-buttons">
      ${buttonsHtml}
    </div>
  `;
  focusQuizPanel();
}

function chooseOption(letter) {
  if (answered) return;
  answered = true;
  const q = filtered[currentIndex];
  document.querySelectorAll('.mcq-option').forEach(btn => {
    btn.disabled = true;
    if (btn.dataset.letter === q.correct) btn.classList.add('correct');
    else if (btn.dataset.letter === letter) btn.classList.add('wrong');
  });
  const isRight = letter === q.correct;
  if (isRight) correctCount++;
  else reviewCount++;
  const resultEl = document.getElementById('mcq-result');
  if (resultEl) {
    resultEl.textContent = isRight ? 'Correct!' : `Not quite — the answer is ${q.correct}.`;
  }
  const container = document.getElementById('answer-container');
  if (container) container.hidden = false;
  const buttons = document.querySelector('.quiz-buttons');
  if (buttons) {
    buttons.innerHTML = `<button type="button" class="primary" data-action="next">Next →</button>`;
    buttons.querySelector('button').focus();
  }
}

function showAnswer() {
  document.getElementById('answer-container').hidden = false;
  const buttons = document.querySelector('.quiz-buttons');
  buttons.innerHTML = `
    <button type="button" class="primary got-it" data-action="got">✓ Got it</button>
    <button type="button" data-action="review">✗ Review again</button>
  `;
  buttons.querySelector('button').focus();
}

function markCorrect() {
  correctCount++;
  nextQuestion();
}

function markReview() {
  reviewCount++;
  nextQuestion();
}

function skipQuestion() {
  skippedCount++;
  nextQuestion();
}

function nextQuestion() {
  currentIndex++;
  if (currentIndex < filtered.length) {
    renderQuestion();
  } else {
    showSummary();
  }
}

// Link into the self-assessment with the current practice filters carried
// over (difficulty, and the section group when the session is confined to
// one). ASSESSMENT_HREF / QUIZ_PAGE_GROUP come from the build prelude.
function assessmentHandoffHref() {
  const params = new URLSearchParams();
  if (difficultyFilter !== 'All') params.set('difficulty', difficultyFilter);
  let group = typeof QUIZ_PAGE_GROUP !== 'undefined' ? QUIZ_PAGE_GROUP : null;
  if (!group && filtered.length > 0) {
    const groups = new Set(filtered.map(q => q.sectionGroup).filter(Boolean));
    if (groups.size === 1) group = groups.values().next().value;
  }
  if (group) params.set('group', group);
  const base = typeof ASSESSMENT_HREF !== 'undefined' ? ASSESSMENT_HREF : 'assessment.html';
  const qs = params.toString();
  return qs ? `${base}?${qs}` : base;
}

function showSummary() {
  const percentage = Math.round((correctCount / filtered.length) * 100);
  const skipped = skippedCount > 0 ? ` · ${skippedCount} skipped` : '';
  document.getElementById('quiz-panel').innerHTML = `
    <div class="quiz-summary">
      <h3 class="quiz-title">Quiz Complete! 🎉</h3>
      <p class="quiz-summary-line">You got <strong>${correctCount} out of ${filtered.length}</strong> correct.</p>
      <div class="quiz-score">${percentage}%</div>
      <p class="quiz-summary-sub">${reviewCount} to review${skipped}</p>
      <p class="quiz-handoff"><a href="${escapeHtml(assessmentHandoffHref())}">Think you know it? Take a timed assessment →</a></p>
      <button type="button" data-action="restart">← Try Again</button>
      <button type="button" class="secondary" data-action="exit">Exit Quiz</button>
    </div>
  `;
  focusQuizPanel();
}

function restartQuiz() {
  filtered = applyFilters();
  resetSession();
  renderQuestion();
}

function exitQuiz() {
  closeQuizPanel();
}

async function initQuiz() {
  try {
    const data = await loadQuizData('quiz-data');
    allQuestions = data ?? scrapeFromDom();
  } catch (err) {
    const countEl = document.getElementById('quiz-match-count');
    if (countEl) {
      countEl.textContent = 'Could not load the question data. Reload the page, or open a section page directly.';
      countEl.classList.add('empty');
    }
    const startBtn = document.getElementById('quiz-start-btn');
    if (startBtn) startBtn.disabled = true;
    return;
  }
  filtered = applyFilters();
  updateMatchCount();
}

bindActions({
  filter: (el) => { difficultyFilter = el.dataset.value; setPressed('.filter-btn', el); onFiltersChanged(); },
  type: (el) => { typeFilter = el.dataset.value; setPressed('.type-btn', el); onFiltersChanged(); },
  scenario: (el) => { scenarioOnly = !!el.checked; onFiltersChanged(); },
  section: onFiltersChanged,
  shuffle: onFiltersChanged,
  start: startQuiz,
  reset: resetFilters,
  exit: exitQuiz,
  choose: (el) => chooseOption(el.dataset.letter),
  show: showAnswer,
  got: markCorrect,
  review: markReview,
  skip: skipQuestion,
  next: nextQuestion,
  restart: restartQuiz,
});

// Keyboard: Escape closes the panel; a letter key chooses that MCQ option
// while an unanswered MCQ is showing (never on modifier combos or in a form
// control -- see isShortcutBlocked).
document.addEventListener('keydown', (e) => {
  const panel = document.getElementById('quiz-panel');
  if (!panel || !panel.classList.contains('visible')) return;
  if (e.key === 'Escape') { exitQuiz(); return; }
  if (answered || isShortcutBlocked(e)) return;
  const q = filtered[currentIndex];
  if (!q || !isMcqItem(q)) return;
  const letter = e.key.toUpperCase();
  if (q.options.some(o => o.letter === letter)) chooseOption(letter);
});

initQuiz();

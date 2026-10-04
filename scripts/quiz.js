// Runs inline in every generated page (see scripts/build_site.mjs's wrapPage),
// so this is a plain classic script, not an ES module — it can't `import`
// scripts/lib/questions.mjs. These two vocab lists intentionally mirror
// DIFFICULTIES / TAGS there; keep them in sync if the tag vocabulary changes.
const DIFFICULTIES = ['Basic', 'Intermediate', 'Advanced'];
const TAGS = ['Scenario', 'MCQ'];

// An item only renders as a multiple-choice card if it's tagged [MCQ] *and*
// carries a usable options list and correct letter -- otherwise (e.g. a
// scrape that failed to find the option <ul>) it degrades to a flashcard
// rather than rendering a broken quiz.
function isMcqItem(q) {
  return q.tags.includes('MCQ') && Array.isArray(q.options) && q.options.length > 0 && !!q.correct;
}

function scrapeFromDom() {
  const items = [];
  document.querySelectorAll('h2').forEach(h2 => {
    if (!/^Q\d+\./.test(h2.textContent.trim())) return;
    // Collect every `[Tag]` code span in the heading (not just the first —
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
    // <details> — read it off before continuing the walk to <details>.
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
    const answerHTML = el ? el.innerHTML : '<summary>No answer found</summary>';
    const correctMatch = el ? el.textContent.match(/\bCorrect:\s*([A-Z])\./) : null;
    const correct = correctMatch ? correctMatch[1] : null;
    items.push({ questionText, difficulty, tags, answerHTML, options, correct, section: null, sectionGroup: null, href: null });
  });
  return items;
}

function loadQuestions() {
  const dataEl = document.getElementById('quiz-data');
  if (!dataEl) return scrapeFromDom();
  const raw = JSON.parse(dataEl.textContent);
  return raw.map(item => ({
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

const allQuestions = loadQuestions();

let filtered = [], currentIndex = 0, correctCount = 0, reviewCount = 0;
let difficultyFilter = 'All';
let scenarioOnly = false;
let typeFilter = 'All';
let answered = false;

function shuffleArray(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function applyFilters() {
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
  if (shuffleToggle && shuffleToggle.checked) {
    result = shuffleArray(result);
  }
  return result;
}

function setFilter(level, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  const target = btn || (typeof event !== 'undefined' ? event.target : null);
  if (target) target.classList.add('active');
  difficultyFilter = level;
  filtered = applyFilters();
  if (currentIndex >= filtered.length) currentIndex = 0;
  if (document.getElementById('quiz-panel').classList.contains('visible')) {
    currentIndex = 0;
    renderQuestion();
  }
}

function onScenarioChange() {
  const toggle = document.getElementById('scenario-toggle');
  scenarioOnly = !!(toggle && toggle.checked);
  filtered = applyFilters();
  currentIndex = 0;
  if (document.getElementById('quiz-panel').classList.contains('visible')) {
    renderQuestion();
  }
}

function onSectionChange() {
  filtered = applyFilters();
  currentIndex = 0;
  if (document.getElementById('quiz-panel').classList.contains('visible')) {
    renderQuestion();
  }
}

function onTypeChange() {
  const select = document.getElementById('type-select');
  typeFilter = select ? select.value : 'All';
  filtered = applyFilters();
  currentIndex = 0;
  if (document.getElementById('quiz-panel').classList.contains('visible')) {
    renderQuestion();
  }
}

function onShuffleChange() {
  filtered = applyFilters();
  currentIndex = 0;
  if (document.getElementById('quiz-panel').classList.contains('visible')) {
    renderQuestion();
  }
}

function startQuiz() {
  filtered = applyFilters();
  currentIndex = 0;
  correctCount = 0;
  reviewCount = 0;
  document.getElementById('main-content').classList.add('hidden');
  document.getElementById('quiz-overlay').classList.add('visible');
  document.getElementById('quiz-panel').classList.add('visible');
  renderQuestion();
}

function renderQuestion() {
  if (filtered.length === 0) {
    document.getElementById('quiz-panel').innerHTML = `
      <div class="quiz-header">
        <div><strong>No questions match these filters</strong></div>
        <button onclick="exitQuiz()" style="border: none; background: none; cursor: pointer; font-size: 1.2em;">✕</button>
      </div>
    `;
    return;
  }
  const q = filtered[currentIndex];
  answered = false;
  const progress = ((currentIndex + 1) / filtered.length * 100);
  const diffClass = q.difficulty.toLowerCase();
  const tagBadges = q.tags.map(t => `<span class="tag-badge ${t.toLowerCase()}">${t}</span>`).join('');
  const sectionLine = q.section
    ? `<div class="quiz-section">${q.section}</div>`
    : '';
  const openLink = (q.section && q.href)
    ? `<a class="quiz-open" href="${q.href}">Open in section page ↗</a>`
    : '';

  const mcq = isMcqItem(q);
  const optionsHtml = mcq
    ? `<div class="mcq-options">${q.options.map(o =>
        `<button class="mcq-option" data-letter="${o.letter}" onclick="chooseOption('${o.letter}')"><span class="mcq-letter">${o.letter}.</span> ${o.html}</button>`
      ).join('')}</div>
    <div id="mcq-result" class="mcq-result"></div>`
    : '';
  const buttonsHtml = mcq
    ? `<button onclick="skipQuestion()">Skip →</button>`
    : `<button class="primary" onclick="showAnswer()">Show Answer</button>
       <button onclick="skipQuestion()">Skip →</button>`;

  const html = `
    <div class="quiz-header">
      <div>
        <strong>Question ${currentIndex + 1} / ${filtered.length}</strong>
        <span class="difficulty-badge ${diffClass}">${q.difficulty}</span>${tagBadges}
      </div>
      <button onclick="exitQuiz()" style="border: none; background: none; cursor: pointer; font-size: 1.2em;">✕</button>
    </div>
    <div class="quiz-progress">
      <div class="quiz-progress-bar" style="width: ${progress}%"></div>
    </div>
    ${sectionLine}
    <div class="quiz-question">${q.questionText}</div>
    ${optionsHtml}
    <div id="answer-container" style="display: none;">
      <div class="quiz-answer">${q.answerHTML}</div>
      ${openLink}
    </div>
    <div class="quiz-buttons">
      ${buttonsHtml}
    </div>
  `;
  document.getElementById('quiz-panel').innerHTML = html;
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
  if (container) container.style.display = 'block';
  const buttons = document.querySelector('.quiz-buttons');
  if (buttons) {
    buttons.innerHTML = `<button class="primary" onclick="nextQuestion()">Next →</button>`;
  }
}

function showAnswer() {
  const container = document.getElementById('answer-container');
  container.style.display = 'block';
  const buttons = document.querySelector('.quiz-buttons');
  buttons.innerHTML = `
    <button class="primary" onclick="markCorrect()" style="background: #4caf50; border-color: #4caf50;">✓ Got it</button>
    <button onclick="markReview()">✗ Review again</button>
  `;
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

function showSummary() {
  const percentage = Math.round((correctCount / filtered.length) * 100);
  const html = `
    <div class="quiz-summary">
      <h3>Quiz Complete! 🎉</h3>
      <p style="font-size: 1.1em; margin: 1.5rem 0;">You got <strong>${correctCount} out of ${filtered.length}</strong> correct.</p>
      <div class="quiz-score">${percentage}%</div>
      <p style="color: #666; margin: 1rem 0;">${reviewCount} to review</p>
      <button onclick="restartQuiz()">← Try Again</button>
      <button onclick="exitQuiz()" style="background: #666; border-color: #666; margin-left: 1rem;">Exit Quiz</button>
    </div>
  `;
  document.getElementById('quiz-panel').innerHTML = html;
}

function restartQuiz() {
  currentIndex = 0;
  correctCount = 0;
  reviewCount = 0;
  renderQuestion();
}

function exitQuiz() {
  document.getElementById('quiz-panel').classList.remove('visible');
  document.getElementById('quiz-overlay').classList.remove('visible');
  document.getElementById('main-content').classList.remove('hidden');
}

// Keyboard shortcut: press a letter key to choose that MCQ option, as long
// as the quiz panel is open, the current card is an unanswered MCQ, and the
// keystroke isn't meant for a focused form control (a <select> or <input>
// in the filter bar).
document.addEventListener('keydown', (e) => {
  const panel = document.getElementById('quiz-panel');
  if (!panel || !panel.classList.contains('visible') || answered) return;
  const target = e.target;
  if (target && /^(SELECT|INPUT|TEXTAREA)$/.test(target.tagName)) return;
  const q = filtered[currentIndex];
  if (!q || !isMcqItem(q)) return;
  const letter = e.key.toUpperCase();
  if (q.options.some(o => o.letter === letter)) chooseOption(letter);
});

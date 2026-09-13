const allQuestions = [];
document.querySelectorAll('h2').forEach(h2 => {
  if (!/^Q\d+\./.test(h2.textContent.trim())) return;
  const codeEl = h2.querySelector('code');
  let difficulty = 'Basic';
  if (codeEl) {
    const match = codeEl.textContent.match(/\[(Basic|Intermediate|Advanced)\]/);
    if (match) difficulty = match[1];
  }
  const questionText = h2.textContent
    .replace(/^Q\d+\.\s*/, '')
    .replace(/\s*\[(Basic|Intermediate|Advanced)\]\s*$/, '')
    .trim();
  let el = h2.nextElementSibling;
  while (el && el.tagName !== 'DETAILS') el = el.nextElementSibling;
  const answerHTML = el ? el.innerHTML : '<summary>No answer found</summary>';
  allQuestions.push({ questionText, difficulty, answerHTML });
});

let filtered = [], currentIndex = 0, correctCount = 0, reviewCount = 0, revealed = false;

function setFilter(level) {
  document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  if (level === 'All') {
    filtered = [...allQuestions];
  } else {
    filtered = allQuestions.filter(q => q.difficulty === level);
  }
  if (currentIndex >= filtered.length) currentIndex = 0;
  if (document.getElementById('quiz-panel').classList.contains('visible')) {
    currentIndex = 0;
    renderQuestion();
  }
}

function startQuiz() {
  filtered = [...allQuestions];
  currentIndex = 0;
  correctCount = 0;
  reviewCount = 0;
  document.getElementById('main-content').classList.add('hidden');
  document.getElementById('quiz-overlay').classList.add('visible');
  document.getElementById('quiz-panel').classList.add('visible');
  renderQuestion();
}

function renderQuestion() {
  revealed = false;
  const q = filtered[currentIndex];
  const progress = ((currentIndex + 1) / filtered.length * 100);
  const diffClass = q.difficulty.toLowerCase();
  const html = `
    <div class="quiz-header">
      <div>
        <strong>Question ${currentIndex + 1} / ${filtered.length}</strong>
        <span class="difficulty-badge ${diffClass}">${q.difficulty}</span>
      </div>
      <button onclick="exitQuiz()" style="border: none; background: none; cursor: pointer; font-size: 1.2em;">✕</button>
    </div>
    <div class="quiz-progress">
      <div class="quiz-progress-bar" style="width: ${progress}%"></div>
    </div>
    <div class="quiz-question">${q.questionText}</div>
    <div id="answer-container" style="display: none;">
      <div class="quiz-answer">${q.answerHTML}</div>
    </div>
    <div class="quiz-buttons">
      <button class="primary" onclick="showAnswer()">Show Answer</button>
      <button onclick="skipQuestion()">Skip →</button>
    </div>
  `;
  document.getElementById('quiz-panel').innerHTML = html;
}

function showAnswer() {
  revealed = true;
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

import js from '@eslint/js';
import globals from 'globals';

// Browser scripts are classic scripts that share one global scope: the build
// emits a prelude (scripts/lib/prelude.mjs) and scripts/quiz-common.js ahead of
// quiz.js / assessment.js, so those names are declared as globals here.
const sharedBrowserGlobals = {
  DIFFICULTIES: 'readonly', TAGS: 'readonly', ASSESSMENT_HREF: 'readonly', QUIZ_PAGE_GROUP: 'readonly', QUIZ_DATA_HREF: 'readonly',
  isMcqItem: 'readonly', shuffleArray: 'readonly', escapeHtml: 'readonly', normalizeItem: 'readonly', isShortcutBlocked: 'readonly',
  difficultyBadgeHtml: 'readonly', closeButtonHtml: 'readonly', mcqOptionHtml: 'readonly', setPressed: 'readonly', loadQuizData: 'readonly',
  bindActions: 'readonly', openQuizPanel: 'readonly', closeQuizPanel: 'readonly', focusQuizPanel: 'readonly', quizPanel: 'readonly',
};

export default [
  { ignores: ['_site/**', 'node_modules/**', '.claude/**', '.venv/**', '06_labs_py/.venv/**'] },
  js.configs.recommended,
  { files: ['scripts/**/*.mjs', 'eslint.config.js'], languageOptions: { ecmaVersion: 2023, sourceType: 'module', globals: globals.node } },
  {
    files: ['scripts/quiz-common.js', 'scripts/quiz.js', 'scripts/assessment.js'],
    languageOptions: { ecmaVersion: 2023, sourceType: 'script', globals: { ...globals.browser, ...sharedBrowserGlobals } },
    rules: {
      // Declarations in one classic script are consumed by the next one.
      'no-unused-vars': 'off',
      'no-redeclare': 'off',
    },
  },
  {
    files: ['scripts/quiz-common.js'],
    languageOptions: { globals: { ...globals.browser, DIFFICULTIES: 'readonly', TAGS: 'readonly', ASSESSMENT_HREF: 'readonly', QUIZ_PAGE_GROUP: 'readonly', QUIZ_DATA_HREF: 'readonly' } },
  },
];

// The small inline script scripts/build_site.mjs emits ahead of the shared
// browser scripts (assets/site/quiz-common.js, then quiz.js or assessment.js).
// It carries the tag vocabulary (single source of truth: scripts/lib/questions.mjs)
// and the per-page constants the browser code needs. The unit tests import the
// same function so the test harness cannot drift from the real build.
import { readFileSync } from 'node:fs';
import { DIFFICULTIES, TAGS } from './questions.mjs';

/** Source of scripts/quiz-common.js, for tests that evaluate the browser scripts with node:vm. */
export const QUIZ_COMMON_JS = readFileSync(new URL('../quiz-common.js', import.meta.url), 'utf8');

/**
 * @param {object} [opts]
 * @param {string} [opts.assessmentHref] relative path from the page to assessment.html
 * @param {string|null} [opts.pageGroup] section group of a quiz section page (null on aggregated pages)
 * @param {string|null} [opts.dataHref] relative path to the page's JSON dataset (null when the page scrapes itself)
 */
export function preludeJs({ assessmentHref = 'assessment.html', pageGroup = null, dataHref = null } = {}) {
  return [
    `const DIFFICULTIES = ${JSON.stringify(DIFFICULTIES)};`,
    `const TAGS = ${JSON.stringify(TAGS)};`,
    `const ASSESSMENT_HREF = ${JSON.stringify(assessmentHref)};`,
    `const QUIZ_PAGE_GROUP = ${JSON.stringify(pageGroup ?? null)};`,
    `const QUIZ_DATA_HREF = ${JSON.stringify(dataHref ?? null)};`,
  ].join('\n');
}

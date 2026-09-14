// Builds the GitHub Pages site into _site/ by mirroring the repo's markdown
// tree to HTML (rewriting relative links/anchors) and copying static assets
// verbatim. Replaces the old inline-bash converter in
// .github/workflows/deploy.yml, which only handled 4 of ~10 content
// directories and never rewrote links, so most links on the deployed site
// 404'd.
import { readdirSync, statSync, mkdirSync, copyFileSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';
import { gfmHeadingId } from 'marked-gfm-heading-id';
import { countQuestions, DIFFICULTIES, TAGS } from './lib/questions.mjs';

const ROOT = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const OUT = path.join(ROOT, '_site');

const SKIP_NAMES = new Set([
  'node_modules',
  '_site',
  'scripts',
  'package.json',
  'package-lock.json',
]);

marked.use(gfmHeadingId());
marked.setOptions({ gfm: true });

const SITE_CSS = readFileSync(path.join(ROOT, 'scripts', 'site.css'), 'utf8');
const QUIZ_JS = readFileSync(path.join(ROOT, 'scripts', 'quiz.js'), 'utf8');

const QUIZ_DIRS = new Set(['02_interview_bank', '03_failure_modes']);
const QUIZ_GROUP_LABELS = { '02_interview_bank': 'Architectures', '03_failure_modes': 'Failure Modes' };

let warnings = 0;
let brokenLinks = 0;
let extractionErrors = 0;

/** Recursively walks the repo, skipping dotfiles/dirs and build tooling. */
function walk(absDir, relDir = '') {
  const out = [];
  for (const entry of readdirSync(absDir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || SKIP_NAMES.has(entry.name)) continue;
    const abs = path.join(absDir, entry.name);
    const rel = relDir ? path.posix.join(relDir, entry.name) : entry.name;
    if (entry.isDirectory()) {
      out.push(...walk(abs, rel));
    } else if (entry.isFile()) {
      out.push({ abs, rel });
    }
  }
  return out;
}

/** Maps a source-relative path to its output-relative path. */
function outputPathFor(rel) {
  if (!rel.toLowerCase().endsWith('.md')) return rel;
  const dir = path.posix.dirname(rel);
  const base = path.posix.basename(rel);
  if (base.toLowerCase() === 'readme.md') {
    return dir === '.' ? 'index.html' : path.posix.join(dir, 'index.html');
  }
  return rel.replace(/\.md$/i, '.html');
}

/** Pulls the first `# Heading` out of a markdown file, rendered inline (tags stripped). */
function extractTitle(md) {
  const match = md.match(/^#\s+(.+?)\s*\r?$/m);
  if (!match) return null;
  const inline = marked.parseInline(match[1]);
  return inline.replace(/<[^>]+>/g, '').trim();
}

function isExternalOrIgnorable(url) {
  return (
    url === '' ||
    /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(url) || // scheme: http:, https:, mailto:, data:, tel:...
    url.startsWith('#') ||
    url.startsWith('/') ||
    url.includes('?')
  );
}

/** Rewrites a single relative URL found in generated HTML to match the _site layout. */
function rewriteUrl(url, srcDirRel, srcRel) {
  if (isExternalOrIgnorable(url)) return url;

  const hashIdx = url.indexOf('#');
  const pathPart = hashIdx === -1 ? url : url.slice(0, hashIdx);
  const frag = hashIdx === -1 ? '' : url.slice(hashIdx);
  if (pathPart === '') return url; // pure fragment already handled above, but be safe

  let decoded;
  try {
    decoded = decodeURI(pathPart);
  } catch {
    warnings++;
    console.warn(`[warn] ${srcRel}: could not decode URL "${url}"`);
    return url;
  }

  const targetRel = path.posix.normalize(path.posix.join(srcDirRel, decoded));
  if (targetRel === '..' || targetRel.startsWith('../')) {
    warnings++;
    console.warn(`[warn] ${srcRel}: link escapes repo root: "${url}"`);
    return url;
  }

  const normTarget = targetRel === '.' ? '' : targetRel;
  const absTarget = path.join(ROOT, normTarget.replace(/\/+$/, ''));
  let st;
  try {
    st = statSync(absTarget);
  } catch {
    st = null;
  }

  if (st && st.isDirectory()) {
    const base = pathPart.replace(/\/+$/, '');
    const result = base === '' || base === '.' ? 'index.html' : base + '/index.html';
    return result + frag;
  }

  if (/\.md$/i.test(pathPart)) {
    if (!st) {
      brokenLinks++;
      console.error(`[error] ${srcRel}: broken link to "${url}" (target does not exist)`);
      return url;
    }
    const base = path.posix.basename(pathPart);
    let result;
    if (base.toLowerCase() === 'readme.md') {
      result = pathPart.slice(0, pathPart.length - base.length) + 'index.html';
    } else {
      result = pathPart.replace(/\.md$/i, '.html');
    }
    return result + frag;
  }

  // Non-markdown, non-directory target (LICENSE, .ipynb, images, etc.) — leave as-is.
  return url;
}

/** Rewrites href/src attribute values in a rendered HTML fragment. */
function rewriteLinks(html, srcDirRel, srcRel) {
  return html.replace(/(\s(?:href|src)=)"([^"]*)"/g, (_match, pre, val) => {
    const unescaped = val.replace(/&amp;/g, '&').replace(/&quot;/g, '"');
    const rewritten = rewriteUrl(unescaped, srcDirRel, srcRel);
    const escaped = rewritten.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
    return `${pre}"${escaped}"`;
  });
}

/**
 * Re-bases relative href/src attribute values that are already relative to
 * srcDirRel so they instead resolve from the repo root (used when lifting a
 * fragment of a page, e.g. an answer block, out into a root-level page).
 */
function rebaseRelativeUrls(html, srcDirRel) {
  return html.replace(/(\s(?:href|src)=)"([^"]*)"/g, (_match, pre, val) => {
    const unescaped = val.replace(/&amp;/g, '&').replace(/&quot;/g, '"');
    if (isExternalOrIgnorable(unescaped)) return `${pre}"${val}"`;
    const hashIdx = unescaped.indexOf('#');
    const pathPart = hashIdx === -1 ? unescaped : unescaped.slice(0, hashIdx);
    const frag = hashIdx === -1 ? '' : unescaped.slice(hashIdx);
    const rebased = srcDirRel ? path.posix.normalize(path.posix.join(srcDirRel, pathPart)) : pathPart;
    const escaped = (rebased + frag).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
    return `${pre}"${escaped}"`;
  });
}

const QUESTION_RE =
  /<h2 id="([^"]+)">(Q\d+\.[\s\S]*?)<\/h2>\s*<details>\s*(?:<summary>[\s\S]*?<\/summary>)?([\s\S]*?)<\/details>/g;

// The trailing run of one or more `<code>[Tag]</code>` spans on a rendered
// question heading (mirrors the markdown-level trailing-tag-run definition
// in scripts/lib/questions.mjs's HEADING_RE — see that file for why this
// must be a trailing, not a first, match: mid-sentence backticked tokens
// like Self-RAG's `[IsSup]` must not be mistaken for tags).
const TAG_RUN_RE = /(?:\s*<code>\[([A-Za-z]+)\]<\/code>)+\s*$/;
const ALL_TAG_TOKENS = new Set([...DIFFICULTIES, ...TAGS]);

/** Pulls {id, question, difficulty, tags, answer} quiz items out of a rendered quiz-page fragment. */
function extractQuizItems(html, { srcDirRel, pageHref, srcRel }) {
  const items = [];
  for (const match of html.matchAll(QUESTION_RE)) {
    const [, id, headingHtml, answerHtml] = match;
    const runMatch = headingHtml.match(TAG_RUN_RE);
    const tokens = runMatch
      ? [...runMatch[0].matchAll(/<code>\[([A-Za-z]+)\]<\/code>/g)].map((t) => t[1]).filter((t) => ALL_TAG_TOKENS.has(t))
      : [];
    const difficulty = tokens.find((t) => DIFFICULTIES.includes(t)) || 'Basic';
    const tags = tokens.filter((t) => TAGS.includes(t));
    const question = headingHtml
      .replace(TAG_RUN_RE, '')
      .replace(/^Q\d+\.\s*/, '')
      .trim();
    const answer = rebaseRelativeUrls(answerHtml.trim(), srcDirRel);
    items.push({ id, question, difficulty, tags, answer, href: `${pageHref}#${id}` });
  }
  if (items.length === 0) {
    extractionErrors++;
    console.error(`[error] ${srcRel}: no quiz questions found (expected Q&A headings)`);
  }
  return items;
}

/**
 * Renders the shared "▶ Start Quiz / Filter: All Basic Intermediate Advanced
 * / Scenario only" control bar used on both individual section pages and the
 * aggregated quiz page. `extraControlsHtml`, when given, is appended inside
 * the same bar (the aggregated page's section `<select>` and shuffle toggle).
 */
function filterBarHtml(extraControlsHtml = '') {
  return `<div class="quiz-bar">
  <button id="quiz-start-btn" onclick="startQuiz()">▶ Start Quiz</button>
  <span style="font-weight: 500;">Filter:</span>
  <button class="filter-btn active" onclick="setFilter('All', this)">All</button>
  <button class="filter-btn" onclick="setFilter('Basic', this)">Basic</button>
  <button class="filter-btn" onclick="setFilter('Intermediate', this)">Intermediate</button>
  <button class="filter-btn" onclick="setFilter('Advanced', this)">Advanced</button>
  <label><input type="checkbox" id="scenario-toggle" onchange="onScenarioChange()"> Scenario only</label>${extraControlsHtml}
</div>
`;
}

/** Wraps rendered body HTML in the shared page shell. */
function wrapPage({ title, bodyHtml, depth, quiz }) {
  const backHref = '../'.repeat(depth) + 'index.html';
  const backLink = depth >= 1 ? `<a class="back" href="${backHref}">← Back to Index</a>\n` : '';
  const quizBar = quiz ? `${filterBarHtml()}<div id="main-content">\n` : '';
  const quizClose = quiz
    ? `</div>
<div class="quiz-panel-overlay" id="quiz-overlay"></div>
<div id="quiz-panel"></div>
<script>
${QUIZ_JS}
</script>
`
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>
${SITE_CSS}
  </style>
</head>
<body>
${backLink}${quizBar}${bodyHtml}
${quizClose}</body>
</html>
`;
}

/** Builds an index page for a directory that has .md content but no README.md. */
function buildDirectoryIndex(relDir, pages) {
  const dirLabel = relDir;
  const items = pages
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((p) => `<li><a href="${p.name}.html">${p.title || p.name}</a></li>`)
    .join('\n');
  const body = `<h1>${dirLabel}</h1>\n<ul>\n${items}\n</ul>`;
  return wrapPage({ title: dirLabel, bodyHtml: body, depth: 1, quiz: false });
}

/** Builds the aggregated all-questions quiz page at the site root. */
function buildQuizPage(items, sections) {
  const quizSectionsFor = (dir) =>
    (sections.get(dir) ?? [])
      .filter((s) => s.count > 0)
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name));

  const tocGroups = ['02_interview_bank', '03_failure_modes']
    .filter((dir) => quizSectionsFor(dir).length > 0)
    .map((dir) => {
      const rows = quizSectionsFor(dir)
        .map((s) => {
          const scenarioSuffix = s.scenarioCount > 0 ? ` · ${s.scenarioCount} scenario${s.scenarioCount === 1 ? '' : 's'}` : '';
          return `<li><a href="${s.href}">${s.title}</a><span class="count">${s.count} question${s.count === 1 ? '' : 's'}${scenarioSuffix}</span></li>`;
        })
        .join('\n');
      return `<h2>${QUIZ_GROUP_LABELS[dir]}</h2>\n<ul class="quiz-toc">\n${rows}\n</ul>`;
    })
    .join('\n');

  const sectionOptions = ['02_interview_bank', '03_failure_modes']
    .filter((dir) => quizSectionsFor(dir).length > 0)
    .map((dir) => {
      const opts = quizSectionsFor(dir)
        .map((s) => `<option value="${s.title}">${s.title}</option>`)
        .join('\n');
      return `<optgroup label="${QUIZ_GROUP_LABELS[dir]}">\n${opts}\n</optgroup>`;
    })
    .join('\n');

  const dataJson = JSON.stringify(items).replace(/</g, '\\u003c');
  const scenarioTotal = items.filter((it) => it.tags.includes('Scenario')).length;

  const extraControls = `
  <select id="section-select" onchange="onSectionChange()">
    <option value="All">All sections</option>
${sectionOptions}
  </select>
  <label><input type="checkbox" id="shuffle-toggle" onchange="onShuffleChange()"> Shuffle</label>`;

  const body = `<h1>RAG Interview Quiz</h1>
<p>${items.length} questions (including ${scenarioTotal} scenario-based) across ${quizSectionsFor('02_interview_bank').length} architectures and ${quizSectionsFor('03_failure_modes').length} failure modes. Filter by difficulty, scenario, or section, then start the quiz.</p>
${filterBarHtml(extraControls)}<div id="main-content">
${tocGroups}
</div>
<div class="quiz-panel-overlay" id="quiz-overlay"></div>
<div id="quiz-panel"></div>
<script type="application/json" id="quiz-data">
${dataJson}
</script>
<script>
${QUIZ_JS}
</script>`;

  return wrapPage({ title: 'RAG Interview Quiz', bodyHtml: body, depth: 0, quiz: false });
}

function main() {
  rmSync(OUT, { recursive: true, force: true });

  const files = walk(ROOT);
  let pageCount = 0;
  let staticCount = 0;
  const quizItems = [];
  const quizSections = new Map(); // dir -> [{name, title, href, count}]

  // Track, per directory, whether it has a README.md and what other .md files it has.
  const dirInfo = new Map(); // rel dir -> { hasReadme: bool, pages: [{name, title}] }

  function touchDir(relDir) {
    if (!dirInfo.has(relDir)) dirInfo.set(relDir, { hasReadme: false, pages: [] });
    return dirInfo.get(relDir);
  }

  for (const { abs, rel } of files) {
    if (!rel.toLowerCase().endsWith('.md')) continue;
    const dir = path.posix.dirname(rel) === '.' ? '' : path.posix.dirname(rel);
    const base = path.posix.basename(rel);
    const info = touchDir(dir);
    if (base.toLowerCase() === 'readme.md') {
      info.hasReadme = true;
    } else {
      const md = readFileSync(abs, 'utf8');
      info.pages.push({ name: base.slice(0, -3), title: extractTitle(md) });
    }
  }

  for (const { abs, rel } of files) {
    const outRel = outputPathFor(rel);
    const outAbs = path.join(OUT, outRel);
    mkdirSync(path.dirname(outAbs), { recursive: true });

    if (!rel.toLowerCase().endsWith('.md')) {
      copyFileSync(abs, outAbs);
      staticCount++;
      continue;
    }

    const md = readFileSync(abs, 'utf8');
    const rawHtml = marked.parse(md);
    const srcDirRel = path.posix.dirname(rel) === '.' ? '' : path.posix.dirname(rel);
    const html = rewriteLinks(rawHtml, srcDirRel, rel);

    const title = extractTitle(md) || 'RAG Interview Questions';
    const depth = srcDirRel === '' ? 0 : srcDirRel.split('/').length;
    const quiz = QUIZ_DIRS.has(srcDirRel);

    if (quiz) {
      const pageHref = path.posix.join(srcDirRel, path.posix.basename(outRel));
      const items = extractQuizItems(html, { srcDirRel, pageHref, srcRel: rel });
      const expectedFromSource = countQuestions(md);
      if (items.length !== expectedFromSource) {
        extractionErrors++;
        console.error(
          `[error] ${rel}: markdown has ${expectedFromSource} well-formed questions but ${items.length} ` +
            `were extracted from the rendered HTML (a question likely has a paragraph/blank issue between its heading and <details>)`
        );
      }
      const sectionTitle = title;
      for (const item of items) {
        quizItems.push({ ...item, section: sectionTitle, sectionGroup: QUIZ_GROUP_LABELS[srcDirRel] });
      }
      if (!quizSections.has(srcDirRel)) quizSections.set(srcDirRel, []);
      quizSections.get(srcDirRel).push({
        name: path.posix.basename(rel),
        title: sectionTitle,
        href: pageHref,
        count: items.length,
        scenarioCount: items.filter((it) => it.tags.includes('Scenario')).length,
      });
    }

    const page = wrapPage({ title, bodyHtml: html, depth, quiz });
    writeFileSync(outAbs, page, 'utf8');
    pageCount++;
  }

  let indexCount = 0;
  for (const [relDir, info] of dirInfo) {
    if (relDir === '') continue; // root README already produces the top-level index.html
    if (info.hasReadme) continue;
    if (info.pages.length === 0) continue;
    const html = buildDirectoryIndex(relDir, info.pages);
    const outAbs = path.join(OUT, relDir, 'index.html');
    mkdirSync(path.dirname(outAbs), { recursive: true });
    writeFileSync(outAbs, html, 'utf8');
    indexCount++;
  }

  const quizPage = buildQuizPage(quizItems, quizSections);
  writeFileSync(path.join(OUT, 'quiz.html'), quizPage, 'utf8');

  console.log(
    `Built ${pageCount} pages, ${indexCount} generated indexes, ${staticCount} static files, ` +
      `quiz page with ${quizItems.length} questions ` +
      `(${warnings} warnings, ${brokenLinks} broken .md links, ${extractionErrors} extraction errors)`
  );

  if (brokenLinks > 0 || extractionErrors > 0) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exitCode = 1;
}

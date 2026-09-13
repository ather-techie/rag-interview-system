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

let warnings = 0;
let brokenLinks = 0;

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

/** Wraps rendered body HTML in the shared page shell. */
function wrapPage({ title, bodyHtml, depth, quiz }) {
  const backHref = '../'.repeat(depth) + 'index.html';
  const backLink = depth >= 1 ? `<a class="back" href="${backHref}">← Back to Index</a>\n` : '';
  const quizBar = quiz
    ? `<div class="quiz-bar">
  <button id="quiz-start-btn" onclick="startQuiz()">▶ Start Quiz</button>
  <span style="font-weight: 500;">Filter:</span>
  <button class="filter-btn active" onclick="setFilter('All')">All</button>
  <button class="filter-btn" onclick="setFilter('Basic')">Basic</button>
  <button class="filter-btn" onclick="setFilter('Intermediate')">Intermediate</button>
  <button class="filter-btn" onclick="setFilter('Advanced')">Advanced</button>
</div>
<div id="main-content">
`
    : '';
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

function main() {
  rmSync(OUT, { recursive: true, force: true });

  const files = walk(ROOT);
  let pageCount = 0;
  let staticCount = 0;

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

  console.log(
    `Built ${pageCount} pages, ${indexCount} generated indexes, ${staticCount} static files ` +
      `(${warnings} warnings, ${brokenLinks} broken .md links)`
  );

  if (brokenLinks > 0) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exitCode = 1;
}

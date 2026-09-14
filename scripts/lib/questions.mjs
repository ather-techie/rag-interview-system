// Pure markdown-source parser for the interview-question format used under
// 02_interview_bank/ and 03_failure_modes/ (see CONTRIBUTING.md "Question
// Format"). No file I/O here — callers read files and pass in the raw text.
//
// This module is intentionally separate from scripts/build_site.mjs, which
// extracts quiz items from *rendered HTML* via its own regex. The two paths
// look at the same content from different angles on purpose: this module
// validates and reports on the markdown source (fast, no `marked` needed,
// gives source line numbers), while build_site.mjs is the one true rendering
// path for the live site. build_site.mjs imports only `countQuestions` from
// here, as a cheap cross-check that nothing silently dropped out of the HTML
// extraction (see build_site.mjs's post-extraction count check).

// Exactly one of these must appear per heading (see parseHeading's
// difficultyCount check). Describes how much expertise the *answer* needs.
export const DIFFICULTIES = ['Basic', 'Intermediate', 'Advanced'];

// Zero or more of these may appear per heading, independent of difficulty.
// `Scenario` marks a question that drops the candidate into a concrete
// situation (a named domain, corpus, SLO, incident, or constraint) and asks
// them to design, diagnose, decide, or trade off — see CONTRIBUTING.md.
export const TAGS = ['Scenario'];

// A single backticked-bracket tag token, e.g. `[Advanced]` or `[Scenario]`.
const TAG_TOKEN_RE = /`\[([A-Za-z]+)\]`/g;

// A valid question heading: "## Q7. Question text? `[Basic]` `[Scenario]`"
// Tags may appear in any order (difficulty first is the documented
// convention) and the whole trailing run is optional in the *match* (so
// callers can detect and report a missing/malformed tag) but a single
// difficulty tag is required for a heading to be considered fully valid —
// see parseQuestionFile's problem collection. The lazy `(.*?)` means only a
// *trailing* run of backticked bracket tokens is ever treated as tags, so a
// mid-sentence backticked token (e.g. Self-RAG's `` `[IsSup]` `` reflection
// token, followed by more prose before the real difficulty tag) is correctly
// left as part of the question title.
export const HEADING_RE =
  /^## Q(\d+)\.\s+(.*?)\s*((?:`\[[A-Za-z]+\]`\s*)*)$/;

// Any other H2/bold pattern that looks like someone's attempt at a question
// but doesn't match HEADING_RE — the legacy `**Q: ...?**` style used in the
// pre-conversion 34-41 files, or a malformed heading (extra space, wrong
// case, "## Interview Q&A" wrapper section, etc).
export const LOOSE_Q_RE = /^(##\s*Q\s*\d+\.|\*\*Q[:.]|## Interview Q&A\s*$)/i;

const SUMMARY_LINE = '<summary>💡 Show Answer</summary>';
const STOPWORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'do', 'does', 'did', 'how', 'what', 'why', 'when', 'where', 'which',
  'who', 'you', 'your', 'and', 'or', 'of', 'in', 'on', 'for', 'to',
  'with', 'it', 'its', 'this', 'that', 'these', 'those', 'can', 'would',
  'should', 'could', 'does', 'as', 'at', 'by', 'from',
]);

/**
 * Parses one "## Qn. ... `[Tag]` `[Tag]`" heading line into its structural
 * pieces, or returns null if the line isn't a heading at all. Splits the
 * trailing tag run into the recognized difficulty (at most one is expected;
 * `difficultyCount` lets callers flag zero or more-than-one), recognized
 * non-difficulty tags (currently just `Scenario`), and anything else
 * (`unknownTags`, e.g. a typo'd or made-up tag).
 */
export function parseHeading(line) {
  const m = line.match(HEADING_RE);
  if (!m) return null;

  const n = Number(m[1]);
  const title = m[2].trim();
  const tagRun = m[3] || '';
  const tokens = [...tagRun.matchAll(TAG_TOKEN_RE)].map((t) => t[1]);

  const difficulties = tokens.filter((t) => DIFFICULTIES.includes(t));
  const tags = tokens.filter((t) => TAGS.includes(t));
  const unknownTags = tokens.filter((t) => !DIFFICULTIES.includes(t) && !TAGS.includes(t));

  return {
    n,
    title,
    difficulty: difficulties[0] || null,
    difficultyCount: difficulties.length,
    tags,
    unknownTags,
  };
}

/** Normalizes a question title for exact-duplicate comparison. */
export function normalizeTitle(title) {
  return title
    .toLowerCase()
    .replace(/`[^`]*`/g, ' ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Tokenizes a normalized title into a content-word set, for near-dup Jaccard comparison. */
function tokenSet(title) {
  return new Set(
    normalizeTitle(title)
      .split(' ')
      .filter((w) => w.length > 2 && !STOPWORDS.has(w))
  );
}

/** Jaccard similarity between two token sets. */
export function jaccard(a, b) {
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

/** Strips fenced code blocks (``` or ~~~) so headings/markers inside them are ignored. */
function maskFences(lines) {
  const masked = lines.slice();
  let inFence = false;
  let fenceMarker = null;
  for (let i = 0; i < masked.length; i++) {
    const line = masked[i];
    const fenceMatch = line.match(/^\s*(```+|~~~+)/);
    if (fenceMatch) {
      if (!inFence) {
        inFence = true;
        fenceMarker = fenceMatch[1][0];
      } else if (fenceMatch[1][0] === fenceMarker) {
        inFence = false;
        fenceMarker = null;
      }
      masked[i] = ''; // fence delimiter line itself is never a heading
      continue;
    }
    if (inFence) masked[i] = '';
  }
  return masked;
}

/**
 * Parses one interview-bank markdown file's raw text into its structural
 * pieces plus a list of format problems. Does not throw; problems are
 * collected and returned so callers (check_questions.mjs) decide how to
 * report/exit.
 */
export function parseQuestionFile(mdRaw, fileRel) {
  const md = mdRaw.replace(/\r\n/g, '\n');
  const rawLines = md.split('\n');
  const lines = maskFences(rawLines);

  const problems = [];
  const questions = [];
  const legacyHits = [];

  // H1 title: first `# ` line outside fences.
  let title = null;
  let h1Line = null;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^#\s+(.+?)\s*$/);
    if (m) {
      title = m[1];
      h1Line = i + 1;
      break;
    }
  }

  const expectedPrefix = fileRel && fileRel.match(/(\d{2})-/)?.[1];
  if (title && expectedPrefix && !new RegExp(`^${expectedPrefix}\\s*—`).test(title)) {
    problems.push({
      level: 'warn',
      code: 'W-H1',
      line: h1Line,
      msg: `H1 "${title}" does not start with "${expectedPrefix} — "`,
    });
  }

  let expectedN = 1;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const headingMatch = parseHeading(line);

    if (!headingMatch) {
      if (LOOSE_Q_RE.test(line) && !/^## Q\d+\./.test(line)) {
        legacyHits.push({ line: i + 1, text: rawLines[i].trim() });
        problems.push({
          level: 'error',
          code: 'E-LEGACY',
          line: i + 1,
          msg: `legacy/malformed question marker: "${rawLines[i].trim()}"`,
        });
      }
      continue;
    }

    const { n, title: qTitle, difficulty, difficultyCount, tags, unknownTags } = headingMatch;

    if (n !== expectedN) {
      problems.push({
        level: 'error',
        code: 'E-NUM',
        line: i + 1,
        msg: `expected Q${expectedN}, found Q${n} (numbering must be contiguous from Q1)`,
      });
    }
    expectedN = n + 1;

    if (difficultyCount === 0) {
      problems.push({
        level: 'error',
        code: 'E-TAG',
        line: i + 1,
        msg: `Q${n}: missing or malformed difficulty tag (expected trailing backticked [Basic|Intermediate|Advanced])`,
      });
    } else if (difficultyCount > 1) {
      problems.push({
        level: 'error',
        code: 'E-TAG',
        line: i + 1,
        msg: `Q${n}: more than one difficulty tag found (exactly one of [Basic|Intermediate|Advanced] is expected)`,
      });
    }

    if (unknownTags.length > 0) {
      problems.push({
        level: 'error',
        code: 'E-TAG',
        line: i + 1,
        msg: `Q${n}: unrecognized tag(s) ${unknownTags.map((t) => `[${t}]`).join(', ')} (accepted: ${[...DIFFICULTIES, ...TAGS].join(', ')})`,
      });
    }

    // Walk forward: <details> must be the first non-blank line after the heading.
    let j = i + 1;
    while (j < lines.length && lines[j].trim() === '') j++;
    const detailsLine = j;
    if (lines[j]?.trim() !== '<details>') {
      problems.push({
        level: 'error',
        code: 'E-DETAILS',
        line: i + 1,
        msg: `Q${n}: "<details>" must be the first non-blank line after the heading`,
      });
    } else {
      const summaryLine = j + 1;
      if (lines[summaryLine]?.trim() !== SUMMARY_LINE) {
        problems.push({
          level: 'error',
          code: 'E-SUMMARY',
          line: summaryLine + 1,
          msg: `Q${n}: line after <details> must be exactly "${SUMMARY_LINE}"`,
        });
      }
      let k = summaryLine + 1;
      while (k < lines.length && lines[k].trim() === '') k++;
      const answerMarkerLine = k;
      if (lines[k]?.trim() !== '**Answer:**') {
        problems.push({
          level: 'error',
          code: 'E-ANSWER',
          line: k + 1,
          msg: `Q${n}: expected "**Answer:**" to start the answer body`,
        });
      }

      // Find the matching </details> before the next H2 or EOF.
      let closeLine = -1;
      let m = k + 1;
      for (; m < lines.length; m++) {
        if (/^## /.test(lines[m])) break;
        if (lines[m].trim() === '</details>') {
          closeLine = m;
          break;
        }
      }
      const answerText = rawLines.slice(answerMarkerLine + 1, closeLine === -1 ? m : closeLine).join('\n').trim();
      const answerWords = answerText.split(/\s+/).filter(Boolean).length;

      if (closeLine === -1) {
        problems.push({
          level: 'error',
          code: 'E-EMPTY',
          line: i + 1,
          msg: `Q${n}: missing "</details>" before next heading or EOF`,
        });
      } else if (answerText.length === 0) {
        problems.push({
          level: 'error',
          code: 'E-EMPTY',
          line: answerMarkerLine + 1,
          msg: `Q${n}: answer body is empty`,
        });
      } else {
        if (answerWords < 40) {
          problems.push({
            level: 'warn',
            code: 'W-SHORT',
            line: answerMarkerLine + 1,
            msg: `Q${n}: answer is only ${answerWords} words (< 40)`,
          });
        }
        let hr = closeLine + 1;
        while (hr < lines.length && lines[hr].trim() === '') hr++;
        // Tolerate an optional single blockquote annotation line (e.g. a
        // "> Related: ..." cross-reference) between </details> and the ---
        // separator -- a legitimate enrichment some answers use.
        if (hr < lines.length && lines[hr].trim().startsWith('>')) {
          hr++;
          while (hr < lines.length && lines[hr].trim() === '') hr++;
        }
        if (hr < lines.length && !/^## /.test(lines[hr]) && lines[hr].trim() !== '---') {
          problems.push({
            level: 'error',
            code: 'E-HR',
            line: hr + 1,
            msg: `Q${n}: expected "---" separator after </details>`,
          });
        }
      }

      questions.push({
        n,
        line: i + 1,
        title: qTitle,
        difficulty,
        tags,
        detailsLine: detailsLine + 1,
        answerText,
        answerWords,
      });
    }
  }

  // Exact in-file duplicate titles.
  const seen = new Map();
  for (const q of questions) {
    const norm = normalizeTitle(q.title);
    if (seen.has(norm)) {
      problems.push({
        level: 'error',
        code: 'E-DUP',
        line: q.line,
        msg: `Q${q.n} title duplicates Q${seen.get(norm)} (normalized: "${norm}")`,
      });
    } else {
      seen.set(norm, q.n);
    }
  }

  return { fileRel, title, h1Line, questions, legacyHits, problems };
}

/** Counts well-formed (single-difficulty-tag) HEADING_RE matches outside fences — used as a cheap cross-check in build_site.mjs. */
export function countQuestions(md) {
  const lines = maskFences(md.replace(/\r\n/g, '\n').split('\n'));
  let n = 0;
  for (const line of lines) {
    const h = parseHeading(line);
    if (h && h.difficultyCount === 1) n++;
  }
  return n;
}

/** Tally of {Basic, Intermediate, Advanced} counts for a parsed file's questions. */
export function difficultyMix(questions) {
  const mix = { Basic: 0, Intermediate: 0, Advanced: 0 };
  for (const q of questions) {
    if (q.difficulty && mix[q.difficulty] !== undefined) mix[q.difficulty]++;
  }
  return mix;
}

/** Tally of non-difficulty tag counts (e.g. { Scenario: 12 }) for a parsed file's questions. */
export function tagCounts(questions) {
  const counts = Object.fromEntries(TAGS.map((t) => [t, 0]));
  for (const q of questions) {
    for (const t of q.tags || []) {
      if (counts[t] !== undefined) counts[t]++;
    }
  }
  return counts;
}

/** Finds near-duplicate title pairs among a list of {title} across one or more files. */
export function findNearDuplicates(entries, { inFileThreshold = 0.8, crossFileThreshold = 0.9 } = {}) {
  const withTokens = entries.map((e) => ({ ...e, tokens: tokenSet(e.title) }));
  const hits = [];
  for (let i = 0; i < withTokens.length; i++) {
    for (let j = i + 1; j < withTokens.length; j++) {
      const a = withTokens[i];
      const b = withTokens[j];
      const sim = jaccard(a.tokens, b.tokens);
      const threshold = a.fileRel === b.fileRel ? inFileThreshold : crossFileThreshold;
      if (sim >= threshold) {
        hits.push({ a, b, sim });
      }
    }
  }
  return hits;
}

export { SUMMARY_LINE };

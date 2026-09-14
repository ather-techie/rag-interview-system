#!/usr/bin/env node
// CLI for validating and reporting on the interview-question bank, and for
// regenerating the hand-maintained counts in README.md. See CONTRIBUTING.md
// for the question format this enforces.
//
// Subcommands:
//   check  [--strict] [--dir <dir>]     validate format; --strict also fails
//                                        on 02_interview_bank files that are
//                                        not yet at the 22-question target or
//                                        below the 2-question [Scenario] target
//   gaps   [--file NN] [--dir <dir>] [--all] [--json]
//                                        report per-file progress toward the
//                                        22-question / 6-8-8 / 2-scenario
//                                        target, with a heuristic rubric-slot
//                                        breakdown
//   readme --write | --check            regenerate README.md counts/badge
//   renumber <file>                      rewrite Qn sequentially outside fences
//
// Usage: node scripts/check_questions.mjs <command> [options]
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  parseQuestionFile,
  countQuestions,
  difficultyMix,
  tagCounts,
  findNearDuplicates,
  normalizeTitle,
  HEADING_RE,
} from './lib/questions.mjs';

const ROOT = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const BANK_DIR = '02_interview_bank';
const FAILURE_DIR = '03_failure_modes';
const QUIZ_DIRS = [BANK_DIR, FAILURE_DIR];
const TARGET_COUNT = 22; // per-file target for 02_interview_bank only
const TARGET_MIX = { Basic: 6, Intermediate: 8, Advanced: 8 };
const TARGET_SCENARIOS = 2; // per-file target count of [Scenario]-tagged questions, 02_interview_bank only
const MIX_TOLERANCE = 1;

// Heuristic rubric used by `gaps` to suggest what to write next. Keyword
// matching against question titles is approximate by design — it's a
// starting point for an author, not a gate. See plan doc for the full
// per-slot description. `tagMatch`, where present, also counts a question
// toward the rule if it carries that tag (independent of title wording).
const RUBRIC = [
  { id: 'R1', label: 'Definition & motivation', slots: ['Basic', 'Basic'], re: /what is|what problem|distinctive mechanism|differ.* from naive/i },
  { id: 'R2', label: 'Mechanism walkthrough', slots: ['Basic', 'Intermediate', 'Intermediate'], re: /how does .* work|pipeline|stage|algorithm|index[- ]time|query[- ]time|step by step/i },
  { id: 'R3', label: 'Taxonomy-neighbour comparison', slots: ['Basic', 'Intermediate'], re: /compar|\bvs\.?\b|versus|difference between|when (would|should) you (choose|prefer|use)/i },
  { id: 'R4', label: 'Implementation / code', slots: ['Intermediate', 'Intermediate'], re: /implement|\bbuild\b|integrat|\bcode\b|write a|reference implementation/i },
  { id: 'R5', label: 'Hyperparameters & knobs', slots: ['Intermediate'], re: /hyperparameter|tun(e|ing)|\bknob|threshold|default value/i },
  { id: 'R6', label: 'Evaluation & benchmarks', slots: ['Intermediate', 'Advanced'], re: /evaluat|benchmark|\bmetric|measure|dataset|recall@|ndcg/i },
  { id: 'R7', label: 'Failure modes & debugging', slots: ['Intermediate', 'Advanced'], re: /\bfail|debug|symptom|mitigat|troubleshoot/i },
  { id: 'R8', label: 'Production ops (cost/latency/scale)', slots: ['Advanced', 'Advanced'], re: /\bcost|latency|\bscale|scaling|throughput|freshness|production/i },
  { id: 'R9', label: 'Security / trust / privacy', slots: ['Advanced'], re: /secur|attack|privacy|poison|adversarial|\btrust/i },
  { id: 'R10', label: 'System-design scenario', slots: ['Basic', 'Advanced'], re: /design a|design an|system design|architecture for/i, tagMatch: 'Scenario' },
  { id: 'R11', label: 'Research origin & limitations', slots: ['Basic', 'Advanced'], re: /paper|arxiv|origin|introduced|limitation|superseded|successor/i },
];

function listMdFiles(dir) {
  const abs = path.join(ROOT, dir);
  return readdirSync(abs)
    .filter((f) => f.toLowerCase().endsWith('.md'))
    .sort()
    .map((f) => path.posix.join(dir, f));
}

function loadFile(relPath) {
  const raw = readFileSync(path.join(ROOT, relPath), 'utf8');
  const parsed = parseQuestionFile(raw, relPath);
  return { relPath, raw, ...parsed };
}

function loadDir(dir) {
  return listMdFiles(dir).map(loadFile);
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------
function runCheck({ strict, dir }) {
  const dirs = dir ? [dir] : QUIZ_DIRS;
  const files = dirs.flatMap(loadDir);

  let errorCount = 0;
  let warnCount = 0;
  const report = [];

  for (const f of files) {
    for (const p of f.problems) {
      report.push({ file: f.relPath, ...p });
      if (p.level === 'error') errorCount++;
      else warnCount++;
    }
  }

  // Bank-wide exact-duplicate check (across files, not just within one).
  const allBankQs = files
    .filter((f) => f.relPath.startsWith(BANK_DIR) || f.relPath.startsWith(FAILURE_DIR))
    .flatMap((f) => f.questions.map((q) => ({ fileRel: f.relPath, title: q.title, n: q.n })));
  const seenAcrossFiles = new Map();
  for (const q of allBankQs) {
    const norm = normalizeTitle(q.title);
    if (seenAcrossFiles.has(norm) && seenAcrossFiles.get(norm).fileRel !== q.fileRel) {
      const other = seenAcrossFiles.get(norm);
      report.push({
        file: q.fileRel,
        level: 'error',
        code: 'E-DUP',
        line: null,
        msg: `Q${q.n} title duplicates ${other.fileRel} Q${other.n} (normalized: "${norm}")`,
      });
      errorCount++;
    } else {
      seenAcrossFiles.set(norm, q);
    }
  }

  // Near-duplicate warning, bank-wide.
  const nearDupEntries = allBankQs.map((q) => ({ fileRel: q.fileRel, title: q.title }));
  for (const { a, b, sim } of findNearDuplicates(nearDupEntries)) {
    report.push({
      file: a.fileRel,
      level: 'warn',
      code: 'W-NEARDUP',
      line: null,
      msg: `near-duplicate (${(sim * 100).toFixed(0)}%) of ${b.fileRel} title "${b.title}"`,
    });
    warnCount++;
  }

  // Per-file bank count target (warning, or error under --strict).
  for (const f of files) {
    if (!f.relPath.startsWith(BANK_DIR)) continue;
    const count = f.questions.length;
    if (count !== TARGET_COUNT) {
      const level = strict ? 'error' : 'warn';
      report.push({
        file: f.relPath,
        level,
        code: 'W-COUNT',
        line: null,
        msg: `${count}/${TARGET_COUNT} questions`,
      });
      if (level === 'error') errorCount++;
      else warnCount++;
    }
  }

  // Per-file bank Scenario-tag target (warning, or error under --strict).
  for (const f of files) {
    if (!f.relPath.startsWith(BANK_DIR)) continue;
    const scenarioCount = tagCounts(f.questions).Scenario;
    if (scenarioCount < TARGET_SCENARIOS) {
      const level = strict ? 'error' : 'warn';
      report.push({
        file: f.relPath,
        level,
        code: 'W-SCENARIO',
        line: null,
        msg: `${scenarioCount}/${TARGET_SCENARIOS} questions tagged [Scenario]`,
      });
      if (level === 'error') errorCount++;
      else warnCount++;
    }
  }

  report.sort((a, b) => a.file.localeCompare(b.file) || (a.line || 0) - (b.line || 0));
  for (const r of report) {
    const tag = r.level === 'error' ? 'error' : 'warn';
    const loc = r.line ? `:${r.line}` : '';
    console.log(`[${tag}] ${r.file}${loc} ${r.code} ${r.msg}`);
  }

  console.log(`\n${files.length} files checked, ${errorCount} errors, ${warnCount} warnings.`);
  process.exitCode = errorCount > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// gaps
// ---------------------------------------------------------------------------
function rubricSlotsFor(questions) {
  const covered = new Map(); // ruleId -> [{n, title, difficulty}]
  for (const rule of RUBRIC) covered.set(rule.id, []);
  for (const q of questions) {
    for (const rule of RUBRIC) {
      const tagHit = rule.tagMatch && q.tags?.includes(rule.tagMatch);
      if (rule.re.test(q.title) || tagHit) covered.get(rule.id).push(q);
    }
  }
  return covered;
}

function gapsForFile(f) {
  const mix = difficultyMix(f.questions);
  const scenarios = tagCounts(f.questions).Scenario;
  const count = f.questions.length;
  const deltas = {
    Basic: TARGET_MIX.Basic - mix.Basic,
    Intermediate: TARGET_MIX.Intermediate - mix.Intermediate,
    Advanced: TARGET_MIX.Advanced - mix.Advanced,
    Scenario: TARGET_SCENARIOS - scenarios,
  };
  const rubricCoverage = rubricSlotsFor(f.questions);
  const missingSlots = [];
  for (const rule of RUBRIC) {
    const have = rubricCoverage.get(rule.id).length;
    const need = rule.slots.length;
    if (have < need) {
      missingSlots.push(`${rule.label} x${need - have} (${rule.slots.slice(have).join(',')})`);
    }
  }
  return { file: f.relPath, count, mix, scenarios, deltas, missingSlots, questions: f.questions, title: f.title };
}

function runGaps({ file, dir, all, json }) {
  const targetDir = dir || BANK_DIR;
  let files = loadDir(targetDir);
  if (file) {
    const prefix = String(file).padStart(2, '0');
    files = files.filter((f) => path.posix.basename(f.relPath).startsWith(prefix + '-'));
  }
  const results = files.map(gapsForFile);
  const filtered = all || file ? results : results.filter((r) => r.count < TARGET_COUNT || targetDir !== BANK_DIR);

  if (json) {
    console.log(JSON.stringify(filtered, null, 2));
    return;
  }

  for (const r of filtered) {
    console.log(`\n${r.file}  ${r.count}/${targetDir === BANK_DIR ? TARGET_COUNT : '-'}  B${r.mix.Basic} I${r.mix.Intermediate} A${r.mix.Advanced} S${r.scenarios}`);
    if (r.title) console.log(`  title: ${r.title}`);
    if (targetDir === BANK_DIR) {
      const needParts = [];
      if (r.deltas.Basic > 0) needParts.push(`+${r.deltas.Basic} B`);
      if (r.deltas.Intermediate > 0) needParts.push(`+${r.deltas.Intermediate} I`);
      if (r.deltas.Advanced > 0) needParts.push(`+${r.deltas.Advanced} A`);
      if (r.deltas.Scenario > 0) needParts.push(`+${r.deltas.Scenario} Scenario`);
      if (needParts.length) console.log(`  need: ${needParts.join('  ')}`);
    }
    if (file || all) {
      console.log('  have:');
      for (const q of r.questions) {
        const tagMarks = (q.tags || []).map((t) => `[${t[0]}]`).join('');
        console.log(`    Q${q.n} [${(q.difficulty || '?')[0]}]${tagMarks} ${q.title}`);
      }
      if (r.missingSlots.length) {
        console.log('  rubric slots likely missing (heuristic, verify manually):');
        for (const s of r.missingSlots) console.log(`    ${s}`);
      }
    }
  }

  const totalCount = results.reduce((s, r) => s + r.count, 0);
  const totalTarget = targetDir === BANK_DIR ? results.length * TARGET_COUNT : null;
  console.log(
    `\n${targetDir}: ${results.length} files, ${totalCount} questions` +
      (totalTarget ? ` / ${totalTarget} target (${totalTarget - totalCount} remaining)` : '')
  );
}

// ---------------------------------------------------------------------------
// readme
// ---------------------------------------------------------------------------
function computeCounts() {
  const bankFiles = loadDir(BANK_DIR);
  const failureFiles = loadDir(FAILURE_DIR);
  const byPath = new Map();
  for (const f of [...bankFiles, ...failureFiles]) {
    byPath.set(f.relPath.replace(/\\/g, '/'), f.questions.length);
  }
  const bankTotal = bankFiles.reduce((s, f) => s + f.questions.length, 0);
  const failureTotal = failureFiles.reduce((s, f) => s + f.questions.length, 0);
  const mix = { Basic: 0, Intermediate: 0, Advanced: 0 };
  let scenarioTotal = 0;
  for (const f of [...bankFiles, ...failureFiles]) {
    const m = difficultyMix(f.questions);
    mix.Basic += m.Basic;
    mix.Intermediate += m.Intermediate;
    mix.Advanced += m.Advanced;
    scenarioTotal += tagCounts(f.questions).Scenario;
  }
  return {
    byPath,
    bankTotal,
    failureTotal,
    mix,
    scenarioTotal,
    bankFileCount: bankFiles.length,
    failureFileCount: failureFiles.length,
  };
}

function regenerateReadme(text, counts) {
  const errors = [];
  let out = text;

  // --- Table row counts (marker-delimited regions) ---
  out = out.replace(
    /(<!-- questions:table -->)([\s\S]*?)(<!-- \/questions:table -->)/g,
    (whole, open, body, close) => {
      const newBody = body.replace(
        /^(\|[^|\n]*\|[^|\n]*\]\(\.\/([^)]+\.md)\)[^|\n]*\|\s*)(\d+)(\s*\|\s*)$/gm,
        (rowWhole, prefix, linkPath, _oldCount, suffix) => {
          const key = linkPath.replace(/\\/g, '/');
          if (!counts.byPath.has(key)) {
            errors.push(`README table row links to "${linkPath}" which has no parseable question count`);
            return rowWhole;
          }
          return `${prefix}${counts.byPath.get(key)}${suffix}`;
        }
      );
      return `${open}${newBody}${close}`;
    }
  );

  // --- Section totals ---
  out = out.replace(
    /(<!-- questions:total 02_interview_bank -->\*\*RAG Architectures Total: )\d+( questions\*\*<!-- \/questions:total -->)/,
    `$1${counts.bankTotal}$2`
  );
  out = out.replace(
    /(<!-- questions:total 03_failure_modes -->\*\*Failure Modes Total: )\d+( questions\*\*<!-- \/questions:total -->)/,
    `$1${counts.failureTotal}$2`
  );

  // --- Concepts total: read back, not recomputed ---
  const conceptsMatch = out.match(/\*\*Core Concepts Total: (\d+) questions across \d+ files\*\*/);
  if (!conceptsMatch) {
    errors.push('Could not find "Core Concepts Total" line to read back concepts count');
  }
  const conceptsTotal = conceptsMatch ? Number(conceptsMatch[1]) : 0;
  const grandTotal = counts.bankTotal + counts.failureTotal + conceptsTotal;

  // --- Grand total ---
  out = out.replace(
    /(<!-- questions:grand -->\*\*Grand Total: )\d+( questions\*\*<!-- \/questions:grand -->)/,
    `$1${grandTotal}$2`
  );

  // --- Difficulty mix (architectures + failure modes only; concepts aren't tagged) ---
  out = out.replace(
    /<!-- questions:mix -->\*\*Difficulty distribution[^*]*\*\*<!-- \/questions:mix -->/,
    `<!-- questions:mix -->**Difficulty distribution across the interview bank and failure modes: ${counts.mix.Basic} Basic, ${counts.mix.Intermediate} Intermediate, ${counts.mix.Advanced} Advanced**<!-- /questions:mix -->`
  );

  // --- Scenario-tagged total (architectures + failure modes only) ---
  if (/<!-- questions:scenarios -->[\s\S]*?<!-- \/questions:scenarios -->/.test(out)) {
    out = out.replace(
      /<!-- questions:scenarios -->\*\*Scenario-based questions[^*]*\*\*<!-- \/questions:scenarios -->/,
      `<!-- questions:scenarios -->**Scenario-based questions (tagged \`[Scenario]\`): ${counts.scenarioTotal}**<!-- /questions:scenarios -->`
    );
  } else {
    errors.push(
      'Could not find "<!-- questions:scenarios -->" marker in README.md — add it (see CONTRIBUTING.md) so scenario counts can be regenerated'
    );
  }

  // --- Badge ---
  out = out.replace(/questions-\d+-blue/, `questions-${grandTotal}-blue`);

  // --- Hero mentions (best-effort text substitutions, not marker-gated) ---
  const heroBadge = out.match(/\d+ Q&A covering \d+ architectures/);
  if (heroBadge) {
    out = out.replace(/\d+ Q&A covering \d+ architectures/, `${grandTotal} Q&A covering ${counts.bankFileCount} architectures`);
  } else {
    errors.push('Could not find banner alt-text "N Q&A covering N architectures" to update');
  }
  const heroPara = out.match(/\*\*\d+ RAG \(Retrieval-Augmented Generation\) interview questions and answers\*\*/);
  if (heroPara) {
    out = out.replace(
      /\*\*\d+ RAG \(Retrieval-Augmented Generation\) interview questions and answers\*\*/,
      `**${grandTotal} RAG (Retrieval-Augmented Generation) interview questions and answers**`
    );
  } else {
    errors.push('Could not find hero paragraph "N RAG ... interview questions and answers" to update');
  }

  // --- Missing/dangling rows: every file in the two dirs must have a row ---
  for (const [key] of counts.byPath) {
    if (!out.includes(`./${key})`)) {
      errors.push(`No README table row links to "${key}" — add one (npm run gaps -- --all can help produce the count)`);
    }
  }

  return { text: out, errors, grandTotal };
}

function runReadme({ write, check }) {
  const readmePath = path.join(ROOT, 'README.md');
  const original = readFileSync(readmePath, 'utf8');
  const counts = computeCounts();
  const { text, errors } = regenerateReadme(original, counts);

  if (errors.length) {
    for (const e of errors) console.error(`[error] README: ${e}`);
  }

  if (check) {
    if (text !== original) {
      console.error('[error] README.md is stale — run `npm run readme` to regenerate it.');
      process.exitCode = 1;
    } else if (errors.length === 0) {
      console.log('README.md is up to date.');
    }
    if (errors.length) process.exitCode = 1;
    return;
  }

  if (write) {
    if (text !== original) {
      writeFileSync(readmePath, text, 'utf8');
      console.log('README.md updated.');
    } else {
      console.log('README.md already up to date.');
    }
    if (errors.length) process.exitCode = 1;
  }
}

// ---------------------------------------------------------------------------
// renumber
// ---------------------------------------------------------------------------
function runRenumber(file) {
  if (!file) {
    console.error('Usage: node scripts/check_questions.mjs renumber <path-to-file.md>');
    process.exitCode = 1;
    return;
  }
  const abs = path.isAbsolute(file) ? file : path.join(ROOT, file);
  const raw = readFileSync(abs, 'utf8');
  const lines = raw.split(/\r?\n/);
  let inFence = false;
  let fenceMarker = null;
  let n = 1;
  let changed = 0;
  const out = lines.map((line) => {
    const fenceMatch = line.match(/^\s*(```+|~~~+)/);
    if (fenceMatch) {
      if (!inFence) {
        inFence = true;
        fenceMarker = fenceMatch[1][0];
      } else if (fenceMatch[1][0] === fenceMarker) {
        inFence = false;
      }
      return line;
    }
    if (inFence) return line;
    const m = line.match(HEADING_RE);
    if (!m) return line;
    const rest = line.slice(line.indexOf('.', line.indexOf('Q')) + 1);
    const newLine = `## Q${n}.${rest}`;
    if (newLine !== line) changed++;
    n++;
    return newLine;
  });
  writeFileSync(abs, out.join('\n'), 'utf8');
  console.log(`Renumbered ${n - 1} questions in ${file} (${changed} lines changed).`);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function main() {
  const [, , command, ...rest] = process.argv;
  const { values } = parseArgs({
    args: rest,
    options: {
      strict: { type: 'boolean', default: false },
      dir: { type: 'string' },
      file: { type: 'string' },
      all: { type: 'boolean', default: false },
      json: { type: 'boolean', default: false },
      write: { type: 'boolean', default: false },
      check: { type: 'boolean', default: false },
    },
    allowPositionals: true,
  });

  switch (command) {
    case 'check':
      runCheck({ strict: values.strict, dir: values.dir });
      break;
    case 'gaps':
      runGaps({ file: values.file, dir: values.dir, all: values.all, json: values.json });
      break;
    case 'readme':
      runReadme({ write: values.write, check: values.check });
      break;
    case 'renumber':
      runRenumber(rest.find((a) => !a.startsWith('--')));
      break;
    default:
      console.error(`Unknown command: ${command}\nUsage: node scripts/check_questions.mjs <check|gaps|readme|renumber> [options]`);
      process.exitCode = 1;
  }
}

main();

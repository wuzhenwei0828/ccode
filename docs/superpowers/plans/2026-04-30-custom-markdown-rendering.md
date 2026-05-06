# Custom Markdown Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frontend assistant-message markdown-it renderer with a custom, limited Markdown-to-HTML function while preserving DOMPurify sanitization and plain-text user messages.

**Architecture:** Keep the external render entrypoint as `renderMd(text)` inside `frontend/index.html`, but replace its internal markdown-it dependency with a small parser composed of HTML escaping, block parsing, inline parsing, and final DOMPurify sanitization. Keep assistant history rendering, new message rendering, and streaming updates on the same code path so the behavior stays consistent across all message flows.

**Tech Stack:** FastAPI static file serving, plain browser JavaScript, DOMPurify, npm lockfile

---

## File Structure

- **Modify** `frontend/index.html`
  - Remove markdown-it script usage and current `window.markdownit(...)` initialization
  - Add small helper functions for escaping HTML, inline parsing, block parsing, and final rendering
  - Keep `renderMessages()` and `updateLastMessage()` calling `renderMd(...)`
  - Preserve DOMPurify sanitization
- **Modify** `package.json`
  - Remove `markdown-it` dependency
- **Modify** `package-lock.json`
  - Remove `markdown-it` and its transitive dependency entries
- **Delete** `frontend/js/markdown-it.min.js`
  - Remove obsolete browser bundle from static assets

---

### Task 1: Remove markdown-it dependency metadata and static asset

**Files:**
- Modify: `package.json`
- Modify: `package-lock.json`
- Delete: `frontend/js/markdown-it.min.js`

- [ ] **Step 1: Write the failing dependency-removal check**

Run this check before changing files:

```bash
python - <<'PY'
from pathlib import Path
package = Path('/Users/zyb/PycharmProjects/CCode/package.json').read_text()
print('markdown-it' in package)
raise SystemExit(0 if 'markdown-it' not in package else 1)
PY
```

Expected: command exits with status 1 because `markdown-it` is still present in `package.json`

- [ ] **Step 2: Remove `markdown-it` from `package.json`**

Change the dependencies section from:

```json
{
  "dependencies": {
    "dompurify": "^3.4.1",
    "markdown-it": "^14.1.1"
  }
}
```

To:

```json
{
  "dependencies": {
    "dompurify": "^3.4.1"
  }
}
```

- [ ] **Step 3: Refresh the lockfile without markdown-it**

Run: `npm uninstall markdown-it`
Expected: `package-lock.json` removes `markdown-it` and its transitive packages while keeping `dompurify`

- [ ] **Step 4: Delete the obsolete browser bundle**

Delete this file from the repo:

```text
frontend/js/markdown-it.min.js
```

- [ ] **Step 5: Verify dependency and asset removal now passes**

Run:

```bash
python - <<'PY'
from pathlib import Path
package = Path('/Users/zyb/PycharmProjects/CCode/package.json').read_text()
lock = Path('/Users/zyb/PycharmProjects/CCode/package-lock.json').read_text()
print('package_has_markdown_it=', 'markdown-it' in package)
print('lock_has_markdown_it=', 'markdown-it' in lock)
print('asset_exists=', Path('/Users/zyb/PycharmProjects/CCode/frontend/js/markdown-it.min.js').exists())
raise SystemExit(0 if 'markdown-it' not in package and 'markdown-it' not in lock and not Path('/Users/zyb/PycharmProjects/CCode/frontend/js/markdown-it.min.js').exists() else 1)
PY
```

Expected:
- `package_has_markdown_it= False`
- `lock_has_markdown_it= False`
- `asset_exists= False`
- exit code 0

- [ ] **Step 6: Commit dependency and asset cleanup**

```bash
git add package.json package-lock.json frontend/js/markdown-it.min.js
git commit -m "build: remove markdown-it dependency"
```

---

### Task 2: Add a failing parser test harness in the browser page

**Files:**
- Modify: `frontend/index.html:946-966`

- [ ] **Step 1: Replace the current markdown-it render stub with a deliberate failing placeholder**

Temporarily replace the current block:

```js
const md = window.markdownit({
  html: true,
  breaks: true,
  linkify: true,
  typographer: false,
});

const processHtml = (html) => {
  html = html.replace(/<p>([^<]*(?:<(?!\/p>)[^<]*)*)<\/p>/g, "$1");
  return html;
};

function renderMd(text) {
  const cleaned = (text || '').replace(/\n{3,}/g, '\n');
  return processHtml(DOMPurify.sanitize(md.render(cleaned)));
}
```

With this failing placeholder:

```js
function renderMd(text) {
  throw new Error('custom markdown renderer not implemented');
}
```

- [ ] **Step 2: Run the page and verify the failure is real**

Run: `python /Users/zyb/PycharmProjects/CCode/main.py`
Expected: FastAPI starts successfully

Then load the page, open an existing conversation, and confirm the browser console shows:

```text
Error: custom markdown renderer not implemented
```

This proves history rendering still reaches `renderMd(...)` and the test is actually exercising the right path.

- [ ] **Step 3: Commit is NOT allowed here**

Do not commit the failing placeholder. Leave the workspace uncommitted and move directly to implementing the real parser.

---

### Task 3: Implement the minimal custom Markdown parser

**Files:**
- Modify: `frontend/index.html:946-966`

- [ ] **Step 1: Add HTML escaping helpers and inline parsing functions**

Replace the failing placeholder with these helper functions:

```js
function escapeHtml(text) {
  return (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function parseInline(text) {
  const codeSpans = [];
  let html = escapeHtml(text).replace(/`([^`]+)`/g, (_, code) => {
    const token = `__CODE_SPAN_${codeSpans.length}__`;
    codeSpans.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });

  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  codeSpans.forEach((snippet, index) => {
    html = html.replace(`__CODE_SPAN_${index}__`, snippet);
  });

  return html;
}
```

- [ ] **Step 2: Add the block-level parser**

Immediately below the inline helpers, add this parser:

```js
function markdownToHtml(text) {
  const lines = (text || '').split('\n');
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
      continue;
    }

    if (line.startsWith('```')) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length && lines[i].startsWith('```')) {
        i += 1;
      }
      html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      continue;
    }

    const headingMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html.push(`<h${level}>${parseInline(headingMatch[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (line.startsWith('> ')) {
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith('> ')) {
        quoteLines.push(lines[i].slice(2));
        i += 1;
      }
      html.push(`<blockquote>${quoteLines.map(parseInline).join('<br>')}</blockquote>`);
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(`<li>${parseInline(lines[i].replace(/^[-*]\s+/, ''))}</li>`);
        i += 1;
      }
      html.push(`<ul>${items.join('')}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(`<li>${parseInline(lines[i].replace(/^\d+\.\s+/, ''))}</li>`);
        i += 1;
      }
      html.push(`<ol>${items.join('')}</ol>`);
      continue;
    }

    const paragraphLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].startsWith('```') &&
      !lines[i].startsWith('> ') &&
      !/^[-*]\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !/^(#{1,4})\s+/.test(lines[i])
    ) {
      paragraphLines.push(lines[i]);
      i += 1;
    }
    html.push(`<p>${paragraphLines.map(parseInline).join('<br>')}</p>`);
  }

  return html.join('');
}
```

- [ ] **Step 3: Restore `renderMd(text)` as the sanitized entrypoint**

Add this final wrapper below `markdownToHtml`:

```js
function renderMd(text) {
  const cleaned = (text || '').replace(/\n{3,}/g, '\n\n').trim();
  return DOMPurify.sanitize(markdownToHtml(cleaned));
}
```

- [ ] **Step 4: Verify the failing symptom is now resolved for history rendering**

Run: `python /Users/zyb/PycharmProjects/CCode/main.py`
Expected: FastAPI starts successfully

Then load the page and open a conversation that previously failed. Expected:
- no `custom markdown renderer not implemented` error
- no `markdownit is not defined` error
- assistant history renders instead of disappearing

- [ ] **Step 5: Commit the parser implementation**

```bash
git add frontend/index.html
git commit -m "feat: add custom markdown renderer"
```

---

### Task 4: Verify supported Markdown subset and rendering paths

**Files:**
- Modify: `frontend/index.html` (only if verification reveals a minimal parser bug)

- [ ] **Step 1: Validate assistant subset rendering in the browser**

Run the app and in the browser console execute:

```js
state.messages = [
  {
    role: 'assistant',
    content: '# Heading\n\nParagraph with **bold**, *italic*, `inline code`, and [link](https://example.com).\n\n> quote line\n> second quote\n\n- one\n- two\n\n1. first\n2. second\n\n```js\nconsole.log("ok")\n```',
    sequence: 0,
  }
];
renderMessages();
```

Expected:
- heading renders as `<h1>`
- bold, italic, inline code, and link render correctly
- quote renders inside `<blockquote>`
- unordered and ordered lists render correctly
- fenced block renders inside `<pre><code>`

- [ ] **Step 2: Validate user messages remain plain text**

In the browser console run:

```js
state.messages = [
  { role: 'user', content: '# not heading\n**not bold**\n[not link](https://example.com)', sequence: 0 }
];
renderMessages();
```

Expected: user message displays literal Markdown markers and does not render as heading, bold text, or hyperlink

- [ ] **Step 3: Validate streaming update path**

In the browser console run:

```js
state.messages = [{ role: 'assistant', content: '', sequence: 0 }];
renderMessages();
updateLastMessage('## Live\n\n- chunked\n- output');
```

Expected: the last assistant message updates in place with rendered heading and list markup

- [ ] **Step 4: Validate sanitization remains active**

In the browser console run:

```js
state.messages = [
  {
    role: 'assistant',
    content: '<img src=x onerror=alert(1)>\n\n<script>alert(2)</script>\n\n`<b>safe as code</b>`',
    sequence: 0,
  }
];
renderMessages();
```

Expected:
- no alert executes
- unsafe HTML does not become executable DOM
- inline code still shows escaped angle brackets as text

- [ ] **Step 5: Apply the smallest fix if verification reveals a parser bug**

If one of the supported syntaxes fails, fix only that specific parser behavior in `frontend/index.html` and re-run the exact failing console reproduction until it passes.

- [ ] **Step 6: Commit any verification-driven parser fix if needed**

If no additional code changes were required, no commit is needed here.
If a parser fix was required, commit it with:

```bash
git add frontend/index.html
git commit -m "fix: polish custom markdown renderer"
```

---

### Task 5: Final static verification and plan coverage review

**Files:**
- Modify: none

- [ ] **Step 1: Run final static verification commands**

Run:

```bash
python -m py_compile /Users/zyb/PycharmProjects/CCode/main.py && python - <<'PY'
from pathlib import Path
html = Path('/Users/zyb/PycharmProjects/CCode/frontend/index.html').read_text()
checks = {
    'no_markdownit_usage': 'window.markdownit' not in html,
    'has_renderMd': 'function renderMd(text)' in html,
    'has_markdownToHtml': 'function markdownToHtml(text)' in html,
    'has_dompurify': 'DOMPurify.sanitize(' in html,
    'assistant_renders_markdown': "m.role === 'assistant' ? renderMd(m.content) : escHtml(m.content)" in html,
    'stream_renders_markdown': 'contentEl.innerHTML = renderMd(content);' in html,
}
for name, ok in checks.items():
    print(f'{name}={ok}')
raise SystemExit(0 if all(checks.values()) else 1)
PY
```

Expected:
- Python syntax check passes
- all printed checks are `True`
- command exits with status 0

- [ ] **Step 2: Run final diff review for intended files only**

Run: `git diff -- frontend/index.html package.json package-lock.json frontend/js/markdown-it.min.js frontend/js/purify.min.js`
Expected: diff shows only the custom renderer migration work and no unrelated files

- [ ] **Step 3: Check the work against the spec requirements**

Use this checklist and confirm each item against the code and browser verification:

```text
[ ] markdown-it removed
[ ] custom markdownToHtml added
[ ] renderMd remains the only assistant entrypoint
[ ] assistant supports headings, paragraphs, line breaks, bold, italic, inline code, code fences, blockquotes, unordered lists, ordered lists, links
[ ] user messages remain plain text
[ ] history rendering works
[ ] streaming rendering works
[ ] DOMPurify sanitization remains active
```

- [ ] **Step 4: Do not claim completion until manual browser verification is done**

If browser verification is still pending, report the exact remaining manual checks instead of claiming the work is complete.

---

## Self-Review

- **Spec coverage:** The plan removes markdown-it, adds a custom base-subset parser, preserves DOMPurify, preserves assistant-only rendering, and verifies history/new/streaming paths.
- **Placeholder scan:** No TBD/TODO placeholders remain; every task includes exact files, commands, and concrete code.
- **Type consistency:** Uses the same function names across tasks: `escapeHtml`, `parseInline`, `markdownToHtml`, `renderMd`, `renderMessages`, `updateLastMessage`.

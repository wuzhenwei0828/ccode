# Markdown-it Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frontend assistant-message Markdown renderer from marked to markdown-it while keeping user messages as plain text and preserving DOMPurify sanitization.

**Architecture:** Keep the existing single-file frontend structure in `frontend/index.html`, replace the global Markdown renderer setup with a minimal markdown-it instance, and remove the unused marked-specific extension file. Preserve the current `.markdown-body` styling and render flow so only the assistant message rendering path changes.

**Tech Stack:** FastAPI static file serving, plain browser JavaScript, markdown-it, DOMPurify, npm lockfile

---

## File Structure

- **Modify** `frontend/index.html`
  - Replace the marked script include with a markdown-it browser bundle include
  - Remove the autoDivBlock script include
  - Replace marked configuration with markdown-it initialization
  - Keep DOMPurify sanitization in the assistant render path
  - Keep user messages escaped via `escHtml`
- **Modify** `package.json`
  - Add `markdown-it` dependency
- **Modify** `package-lock.json`
  - Record the resolved `markdown-it` package version
- **Delete** `frontend/js/autoDivBlock.js`
  - Remove the no-longer-used marked extension implementation

---

### Task 1: Add markdown-it dependency metadata

**Files:**
- Modify: `package.json`
- Modify: `package-lock.json`

- [ ] **Step 1: Update `package.json` dependencies**

Change the dependencies object to include `markdown-it` alongside `dompurify`:

```json
{
  "dependencies": {
    "dompurify": "^3.4.1",
    "markdown-it": "^14.1.0"
  }
}
```

- [ ] **Step 2: Refresh the lockfile**

Run: `npm install`
Expected: `package-lock.json` updates to include `markdown-it` and its transitive packages without removing `dompurify`

- [ ] **Step 3: Verify dependency metadata changed as expected**

Run: `git diff -- package.json package-lock.json`
Expected: diff shows `markdown-it` added in both files and no unrelated dependency churn

- [ ] **Step 4: Commit dependency metadata change**

```bash
git add package.json package-lock.json
git commit -m "build: add markdown-it dependency"
```

---

### Task 2: Switch frontend rendering from marked to markdown-it

**Files:**
- Modify: `frontend/index.html:7-8`
- Modify: `frontend/index.html:947-964`
- Modify: `frontend/index.html:1126-1143`

- [ ] **Step 1: Write the failing test scenario manually in the browser**

Use this assistant message fixture in the running app after temporarily injecting it into `state.messages` from DevTools:

```js
state.messages = [
  {
    role: 'assistant',
    content: '# Title\n\n- item 1\n- item 2\n\n`code`',
    sequence: 0,
  }
];
renderMessages();
```

Expected before implementation: rendering still depends on `marked` and `autoDivBlock`, so the page has not yet switched to markdown-it.

- [ ] **Step 2: Replace the library script includes**

Update the top script includes from:

```html
<script src="/static/js/marked.min.js"></script>
<script src="/static/js/autoDivBlock.js"></script>
```

To:

```html
<script src="https://cdn.jsdelivr.net/npm/markdown-it@14/dist/markdown-it.min.js"></script>
```

- [ ] **Step 3: Replace the marked configuration with markdown-it initialization**

Replace the current marked block:

```js
marked.setOptions({
  gfm: true,
  extensions: [autoDivBlock],
  breaks: true,
  pedantic: false,
  sanitize: false,
  highlight: (code, lang) => {
    const hljs = require('highlight.js');
    const validLang = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language: validLang }).value;
  }
});

function renderMd(text) {
  const cleaned = text.replace(/\n{3,}/g, '\n\n');
  return marked.parse(cleaned);
}
```

With this markdown-it version:

```js
const md = window.markdownit({
  html: false,
  breaks: true,
  linkify: true,
  typographer: false,
});

function renderMd(text) {
  const cleaned = (text || '').replace(/\n{3,}/g, '\n\n');
  return DOMPurify.sanitize(md.render(cleaned));
}
```

- [ ] **Step 4: Keep assistant and user rendering behavior separated**

Ensure `renderMessages()` keeps this exact rendering split:

```js
function renderMessages() {
  const el = document.getElementById('messages');
  el.innerHTML = state.messages.map(m => `
    <div class="message ${m.role}">
      <div class="message-role"><span class="dot"></span>${m.role === 'user' ? 'You' : 'Assistant'}</div>
      <div class="message-content ${m.role === 'assistant' ? 'markdown-body' : ''}">${m.role === 'assistant' ? renderMd(m.content) : escHtml(m.content)}</div>
    </div>
  `).join('');
  scrollToBottom();
}
```

And `updateLastMessage()` keeps streaming assistant updates routed through `renderMd`:

```js
function updateLastMessage(content) {
  const messages = document.getElementById('messages');
  const lastMsgEl = messages.querySelector('.message.assistant:last-of-type');
  if (lastMsgEl) {
    const contentEl = lastMsgEl.querySelector('.message-content');
    if (contentEl) contentEl.innerHTML = renderMd(content);
  }
}
```

- [ ] **Step 5: Run a focused browser sanity check**

Run: `python -m http.server 9000 --directory /Users/zyb/PycharmProjects/CCode/frontend`
Expected: a local static preview server starts so you can inspect `index.html` rendering quickly in the browser

- [ ] **Step 6: Verify assistant Markdown and user plain text behavior**

In the browser console, run:

```js
state.messages = [
  { role: 'assistant', content: '# Title\n\n- a\n- b\n\n> quote', sequence: 0 },
  { role: 'user', content: '# not heading\n**not bold**', sequence: 1 },
];
renderMessages();
```

Expected:
- assistant message renders `<h1>`, `<ul>`, and `<blockquote>`
- user message displays literal `# not heading` and `**not bold**`

- [ ] **Step 7: Commit renderer switch**

```bash
git add frontend/index.html
git commit -m "feat: switch assistant markdown rendering to markdown-it"
```

---

### Task 3: Remove the obsolete marked extension file

**Files:**
- Delete: `frontend/js/autoDivBlock.js`

- [ ] **Step 1: Verify no code references remain**

Run: `rg "autoDivBlock" /Users/zyb/PycharmProjects/CCode/frontend /Users/zyb/PycharmProjects/CCode/package.json /Users/zyb/PycharmProjects/CCode/package-lock.json`
Expected: no matches after the `index.html` update

- [ ] **Step 2: Delete the unused file**

Remove this file from the repo:

```text
frontend/js/autoDivBlock.js
```

- [ ] **Step 3: Verify static assets are still coherent**

Run: `ls /Users/zyb/PycharmProjects/CCode/frontend/js`
Expected: `autoDivBlock.js` is gone and remaining static assets still exist

- [ ] **Step 4: Commit cleanup**

```bash
git add frontend/js/autoDivBlock.js
git commit -m "refactor: remove marked autoDiv extension"
```

---

### Task 4: Validate end-to-end behavior and regression boundaries

**Files:**
- Modify: `frontend/index.html` (only if validation reveals a minimal style fix is necessary)

- [ ] **Step 1: Start the application locally**

Run: `python /Users/zyb/PycharmProjects/CCode/main.py`
Expected: FastAPI app starts and serves `/` plus `/static/*`

- [ ] **Step 2: Verify the page loads with the new renderer available**

Open `http://127.0.0.1:8000/`
Expected: page loads without console errors such as `marked is not defined` or `autoDivBlock is not defined`

- [ ] **Step 3: Validate assistant rendering with a comprehensive fixture**

In the browser console, run:

```js
state.messages = [
  {
    role: 'assistant',
    content: '# Heading\n\nParagraph with `inline code`.\n\n- one\n- two\n\n```js\nconsole.log("ok")\n```\n\n> quoted\n\n[link](https://example.com)',
    sequence: 0,
  }
];
renderMessages();
```

Expected:
- heading, paragraph, list, fenced code block, blockquote, and link all render
- code block keeps existing dark styling from `.markdown-body pre`

- [ ] **Step 4: Validate sanitization is still active**

In the browser console, run:

```js
state.messages = [
  {
    role: 'assistant',
    content: '<img src=x onerror=alert(1)>\n\n<script>alert(2)</script>',
    sequence: 0,
  }
];
renderMessages();
```

Expected:
- no alert executes
- unsafe tags or attributes are stripped by DOMPurify before insertion

- [ ] **Step 5: Validate streaming update path**

In the browser console, run:

```js
state.messages = [{ role: 'assistant', content: '', sequence: 0 }];
renderMessages();
updateLastMessage('# Live\n\n- chunked');
```

Expected: the last assistant message updates in place and renders formatted Markdown

- [ ] **Step 6: Run final diff review**

Run: `git diff -- frontend/index.html package.json package-lock.json frontend/js/autoDivBlock.js`
Expected: only the intended renderer swap, dependency addition, and file removal are present

- [ ] **Step 7: Commit final validation fix if needed**

If no extra code changes were needed, note that no commit is required for this task.
If a minimal style or rendering fix was necessary, commit it with:

```bash
git add frontend/index.html
git commit -m "fix: polish markdown-it message rendering"
```

---

## Self-Review

- **Spec coverage:** Covers renderer replacement, assistant-only Markdown rendering, DOMPurify retention, autoDivBlock removal, and focused regression validation.
- **Placeholder scan:** No TBD/TODO placeholders remain; every step includes concrete files, commands, and expected outcomes.
- **Type consistency:** Uses the existing frontend functions and state names consistently: `renderMd`, `renderMessages`, `updateLastMessage`, `state.messages`, `.markdown-body`.

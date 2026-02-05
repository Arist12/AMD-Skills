---
name: git-conventions
description: >
  Enforces Git and GitHub CLI conventions for AI coding agents. Use this skill whenever
  performing any git operation (commit, push, branch, rebase, etc.) or GitHub CLI operation
  (gh pr create, gh issue create, etc.). Covers: commit message format (Conventional Commits),
  PR/issue writing style, co-authorship attribution removal, and safety guardrails preventing
  unsanctioned git-history-altering operations.
---

# Git Conventions for AI Agents

## 1. Commit Message Format (Conventional Commits)

Use this format for every commit:

```
<type>(<scope>): <short summary>
```

- **short summary**: imperative mood, lowercase start, no period, ≤72 chars
- **scope**: optional, indicates the affected module/area

### Allowed Types

| Type       | Use when…                                       |
|------------|--------------------------------------------------|
| `feat`     | Adding a new feature                             |
| `fix`      | Fixing a bug                                     |
| `chore`    | Maintenance (deps, build config, CI, etc.)       |
| `docs`     | Documentation-only changes                       |
| `refactor` | Code restructuring (no bug fix, no new feature)  |
| `perf`     | Performance improvements                         |
| `style`    | Formatting, whitespace, semicolons — no logic    |
| `test`     | Adding or updating tests                         |


## 2. PR & Issue Writing Style

When using `gh pr create`, `gh issue create`, or similar commands, write like a senior engineer — **concise, scannable, no filler**.

### Rules

- Lead with *what* and *why* in 1–2 sentences. Skip preambles like "This PR introduces…".
- Use short bullet lists only when enumerating distinct items (files changed, steps to repro). Prefer prose for context.
- Omit obvious information (e.g., "Tests have been added for the new feature" when the diff clearly shows tests).
- No marketing language, no superlatives, no "comprehensive", "robust", "streamlined".
- Keep the total body under ~150 words unless the change genuinely demands more.
- PR titles: use the same `<type>(<scope>): <summary>` format as commits.

### PR Body Template (guideline, not rigid)

```
<1–2 sentence summary of what changed and why>

Key changes:
- <change 1>
- <change 2>

Testing: <how it was verified, if non-obvious>
```

### Issue Body Template

```
<1–2 sentence description of the problem or request>

Steps to reproduce / context:
- <step or detail>

Expected behavior: <what should happen>
```

## 3. No Co-Authored-By Trailers

Never append `Co-authored-by` or `Signed-off-by` trailers referencing Claude, Anthropic, or any AI assistant to commits or PR descriptions. Do not add AI attribution metadata of any kind to git history.

If a git or gh command would automatically inject such trailers, strip them before executing.

## 4. Git Safety — Explicit Permission Required

**Never run the following commands unless the user explicitly requests them in the current message:**

- `git commit` / `git commit --amend`
- `git push` / `git push --force`
- `git rebase`
- `git merge`
- `git reset`
- `git tag`
- `gh pr create` / `gh pr merge`
- `gh issue create`
- Any command that creates, modifies, or deletes commits, branches, tags, PRs, or issues

Staging (`git add`), diffing (`git diff`), status checks (`git status`), and log inspection (`git log`) are always safe to run unprompted.

If a workflow logically leads to a commit or push, **stop and ask the user for confirmation** before executing.

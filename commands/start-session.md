---
description: Initialize a new session with continuity from previous work - loads followups, checks doc health, and suggests focus areas
allowed-tools: Read, Glob, Grep, Bash
---

# Session Start Workflow

You are initializing a new development session. This is a **read-only** command — it gathers context and presents a structured summary without modifying any files.

## Step 1: Load Previous Session State

Check for persisted session state from the last `/wrap-session`:

### 1a. Followup Items

```bash
cat .claude/session-state/followups.md 2>/dev/null || echo "No followups found"
```

If found, parse the priority levels and complexity estimates.

### 1b. Automation Log

```bash
cat .claude/session-state/automation-log.md 2>/dev/null || echo "No automation log found"
```

If found, look for `RECURRING` items (appeared 3+ sessions).

### 1c. Last Checkpoint

```bash
cat .claude/session-state/checkpoint.md 2>/dev/null || echo "No checkpoint found"
```

If found, check for in-progress work or blockers from a mid-session save.

---

## Step 2: Git Context

Gather current repository state:

```bash
git log --oneline -10
git branch --show-current
git status --short
```

Also check for open PRs and assigned issues if `gh` is available:

```bash
gh pr list --author @me --state open 2>/dev/null || echo "No gh CLI or no open PRs"
gh issue list --assignee @me --state open --limit 5 2>/dev/null || echo "No assigned issues"
```

---

## Step 3: Documentation Health Check

### 3a. Conflict Markers

Scan for unresolved git conflict markers:

```bash
grep -rn '<<<<<<\|======\|>>>>>>' .claude/ ~/.claude/skills/ --include='*.md' 2>/dev/null || echo "No conflicts found"
```

### 3b. File Sizes

Check documentation file sizes for bloat detection:

```bash
wc -l .claude/CLAUDE.md .claude/agents/*/SKILL.md ~/.claude/skills/*/SKILL.md 2>/dev/null
```

Flag any SKILL.md files over 500 lines as potentially needing pruning.

---

## Step 4: Present Summary

Print a structured summary to the console:

```markdown
# Session Start — [Date]

## Previous Session Followups
[Priority 1 items from followups.md, or "None found"]

## Recurring Automation Opportunities
[RECURRING items from automation-log.md, or "None yet"]

## In-Progress Work
[From checkpoint.md or git status, or "Clean slate"]

## Git Context
- **Branch:** [current branch]
- **Recent commits:** [last 3]
- **Open PRs:** [count and titles]
- **Assigned issues:** [count and titles]

## Doc Health
- **Conflicts:** [count or "None"]
- **Large files:** [any SKILL.md > 500 lines]

## Suggested Focus Areas
1. [Highest priority followup item]
2. [Any RECURRING automation opportunity worth implementing]
3. [Open PR needing attention]
```

---

## Step 5: Optional Task List

Ask the user if they want to create a task list from the prioritized items:

> Would you like me to create a task list from these items to track progress during this session?

If yes, use TaskCreate to build tasks from the Priority 1 and Priority 2 followup items.

---

## Notes

- This command is **read-only** — it never modifies files
- If no session state exists (first time), it still provides git context and doc health
- Runs quickly (no agents, just file reads and git commands)
- Pairs with `/wrap-session` for session continuity and `/checkpoint` for mid-session saves

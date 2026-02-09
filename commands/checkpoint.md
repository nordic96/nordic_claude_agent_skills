---
description: Lightweight mid-session save - capture progress, blockers, and next steps without running agents
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# Mid-Session Checkpoint

You are creating a lightweight checkpoint of the current session state. This should complete in under 30 seconds — no agents are involved.

## Step 1: Capture Progress

Analyze the current conversation to identify:

- **Accomplished:** What was completed this session so far
- **In-progress:** What is currently being worked on
- **Blockers:** Any issues preventing progress
- **Next steps:** What should be done next in this session

## Step 2: Git Snapshot

Gather current git state:

```bash
git diff --stat
git status --short | wc -l
```

Note the number of uncommitted changes and files modified.

## Step 3: Write Checkpoint

Create `.claude/session-state/` directory if it doesn't exist, then write to `.claude/session-state/checkpoint.md` (overwrite, not append):

```markdown
# Session Checkpoint

> Saved: [YYYY-MM-DD HH:MM]

## Accomplished
- [completed item 1]
- [completed item 2]

## In Progress
- [current work item] — [status/notes]

## Blockers
- [blocker description, or "None"]

## Next Steps
1. [next action]
2. [next action]

## Git State
- Modified files: [count]
- Uncommitted changes: [yes/no]
- Current branch: [branch name]
```

## Step 4: Confirm

Print a brief confirmation to the console:

```
Checkpoint saved to .claude/session-state/checkpoint.md
- [N] items accomplished, [N] in progress, [N] blockers
```

---

## Notes

- This command is fast — no agents, no analysis phases
- Overwrites the previous checkpoint (only latest state matters)
- Use `/wrap-session` for full session wrap-up with agents
- Use `/start-session` to load this checkpoint in the next session

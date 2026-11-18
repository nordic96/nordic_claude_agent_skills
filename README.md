# Nordic Claude Agent Skills

Global Claude Code commands, agents and skills shared across projects, symlinked into
`~/.claude/`.

```
agents/     # global agents (none yet)
skills/     # global skills (none yet; skills/synced/ is Anthropic's, gitignored)
commands/
  issue-dev.md       # fetch an issue, branch, implement
  run-pr-checks.md   # lint/test/build, push, open PR, link the issue
```

## Install

```bash
ln -s ~/projects/nordic_claude_agent_skills/agents   ~/.claude/agents
ln -s ~/projects/nordic_claude_agent_skills/skills   ~/.claude/skills
ln -s ~/projects/nordic_claude_agent_skills/commands ~/.claude/commands
```

## What goes here

Only what applies to **every** project. Project rules, agents and skills live in the
project's own `.claude/` (and decisions in its notes, if it has a vault). Don't add
general programming knowledge the model already has; add a skill only for a workflow
or a trap that actually bit.

## Removed (Nov 2026)

`wrap-session`, `start-session`, `checkpoint`, `fetch-pr-review`, the six session-doc
agents, the `frontend-dev` / `ui-ux-designer` agents and the `frontend-dev` /
`backend-dev` / `ui-ux-designer` skills. Unused for months or never; recover from git
history if needed.

---
description: Wrap up the current session - extract learnings, find automation opportunities, update docs, and suggest next steps
allowed-tools: Read, Glob, Grep, Edit, Task, Bash
---

# Session Wrap-Up Workflow

You are coordinating a session wrap-up workflow. Execute the following phases:

## Phase 1: Analysis (Run in PARALLEL)

Launch these two agents simultaneously in a single message with multiple Task tool calls:

### 1a. Learning Extractor + Automation Scout (Combined)

Use the Task tool with `subagent_type: learning-extractor` to analyze this session for BOTH learnings AND automation opportunities:

**Learnings:**
- Mistakes made and how they were resolved
- New patterns or techniques discovered
- Debugging approaches that worked well
- Performance insights gained

**Automation Opportunities:**
- Repetitive tasks that could become slash commands
- Manual processes that could be automated
- Patterns that warrant new agents
- Workflow optimizations

**IMPORTANT - Dual-Destination Routing:**

The agent must classify each learning and route to the correct destination:

| Pattern Type | Destination | Example |
|--------------|-------------|---------|
| CSS/JS gotcha | `~/.claude/skills/{agent}/SKILL.md` | "Transforms are atomic" |
| WCAG rule | `~/.claude/skills/{agent}/SKILL.md` | "4.5:1 contrast ratio" |
| Library API quirk | `~/.claude/skills/{agent}/SKILL.md` | "simple-icons SVG format" |
| Framework-agnostic pattern | `~/.claude/skills/{agent}/SKILL.md` | "useMemo for expensive calcs" |
| Component implementation | `.claude/agents/{agent}/SKILL.md` | "useStaggeredAnimation hook" |
| Design decision | `.claude/agents/{agent}/SKILL.md` | "Glass card 30% opacity" |
| Project integration | `.claude/agents/{agent}/SKILL.md` | "Vercel build script" |
| Session-dated learnings | `.claude/agents/{agent}/SKILL.md` | "## Session Learnings - Jan 24" |

**Classification Rules:**

1. **Global Skills** (`~/.claude/skills/`) - Patterns that would help in ANY project:
   - Programming language fundamentals
   - CSS/HTML gotchas
   - Accessibility standards
   - Performance optimization techniques
   - Design theory and principles
   - Library quirks that apply universally

2. **Project Skills** (`.claude/agents/`) - Patterns specific to THIS project:
   - Component implementations
   - Design system decisions
   - Session-dated learnings (always stay in project)
   - Build/deploy configurations
   - Project-specific hooks and utilities

### 1b. Doc Updator (Parallel with 1a)

Use the Task tool with `subagent_type: doc-updator` to suggest and apply updates to:
- `CLAUDE.md` for project-wide documentation
- Agent instructions if patterns changed
- New sections for undocumented features

The agent will edit `CLAUDE.md` directly.

**IMPORTANT:** Run 1a and 1b in parallel by including both Task tool calls in a single message.

## Phase 2: Review (After Phase 1 completes)

### 2. Followup Suggestor

Use the Task tool with `subagent_type: followup-suggestor` to identify:
- Incomplete work items from this session
- Technical debt introduced
- Testing gaps
- Prioritized list for next session

Print the followup items to console (don't persist).

## Phase 3: Consolidation (Run last)

### 3. Doc Consolidator (Includes duplicate detection)

Use the Task tool with `subagent_type: doc-consolidator` to:

**Step 1 - Detect duplicates:**
- Check CLAUDE.md for internal duplicates
- Check project SKILL.md files for redundant content
- Check global SKILL.md files for content that moved to project
- Identify contradictions between files

**Step 2 - Apply fixes:**
- Merge duplicate content into canonical locations
- Remove redundant copies and add cross-references
- Resolve contradictions by updating outdated information
- Ensure proper file ownership hierarchy

**Canonical File Hierarchy:**

| File | Owns | Should NOT contain |
|------|------|-------------------|
| `~/.claude/skills/*/SKILL.md` | Universal patterns, framework-agnostic | Project-specific details, session dates |
| `.claude/agents/*/SKILL.md` | Project patterns, session learnings | Universal patterns (link to global) |
| `.claude/CLAUDE.md` | Project architecture, design system | Agent-specific learnings |

The agent will edit files directly to consolidate documentation.

## Phase 4: Commit Global Skills (After Phase 3 completes)

After doc-consolidator finishes, check if any changes were made to the global skills repository and commit them.

### 4. Commit and Push Global Skills

Run the following steps using the Bash tool:

1. **Check for changes** in the global skills repo:
   ```bash
   cd ~/projects/nordic_claude_agent_skills && git status --porcelain
   ```

2. **If changes exist**, commit and push them:
   ```bash
   cd ~/projects/nordic_claude_agent_skills && git add -A && git commit -m "docs: update skills from session wrap-up

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" && git push
   ```

3. **Report the result** in the final summary:
   - If changes were committed: list the files changed
   - If no changes: note that global skills were unchanged

**Note:** The global `~/.claude/` directory symlinks to `~/projects/nordic_claude_agent_skills/`:
- `~/.claude/skills/` → `nordic_claude_agent_skills/skills/`
- `~/.claude/agents/` → `nordic_claude_agent_skills/agents/`
- `~/.claude/commands/` → `nordic_claude_agent_skills/commands/`

## Final Summary

After all phases complete, provide a brief summary:
- Key learnings captured (with destination: global vs project)
- Automation opportunities identified
- Documentation updates made
- Priority items for next session
- Duplicates found and consolidation actions taken
- Global skills commit status (committed/no changes)

---

## Optimization Notes

This workflow has been optimized from the original 6-agent sequential flow:

| Aspect | Before | After |
|--------|--------|-------|
| Agents | 6 sequential | 4 (2 parallel + 2 sequential) + git commit |
| Est. Time | ~3-4 min | ~2-2.5 min |
| File conflicts | Possible | Minimized |

**Changes:**
- Merged learning-extractor + automation-scout (both update SKILL.md)
- Merged duplicate-checker + doc-consolidator (related tasks)
- Phase 1 runs in parallel for faster execution
- Added dual-destination routing for global vs project learnings
- Phase 4 auto-commits global skills changes to nordic_claude_agent_skills repo

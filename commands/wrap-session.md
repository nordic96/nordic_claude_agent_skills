---
description: Wrap up the current session - extract learnings, find automation opportunities, update docs, and suggest next steps
allowed-tools: Read, Glob, Grep, Edit, Task, Bash
---

# Session Wrap-Up Workflow

You are coordinating a session wrap-up workflow. Execute the following phases in order:

## Phase 0: Pre-flight Validation

Before any agents run, perform these checks using Bash/Grep tools directly:

### 0a. Conflict Marker Scan

Scan for git conflict markers in all documentation files:

```bash
grep -rn '<<<<<<\|======\|>>>>>>' .claude/ ~/.claude/skills/ --include='*.md' || echo "No conflicts found"
```

**If conflicts are found:**
- Resolve by removing marker lines (`<<<<<<<`, `=======`, `>>>>>>>`), keeping both content sections
- If ambiguous (contradictory content on each side): ask the user which to keep
- **Block Phase 1** until all conflicts are resolved

### 0b. Baseline File Sizes

Record file sizes for delta reporting in the final summary:

```bash
wc -l .claude/CLAUDE.md .claude/agents/*/SKILL.md ~/.claude/skills/*/SKILL.md 2>/dev/null
```

Save these numbers for comparison in the final summary.

---

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

**Session Archive Rule:**

The learning-extractor will enforce a 3-session limit on project SKILL.md files:
- If 3+ `## Session Learnings -` sections exist, the oldest is moved to `SKILL_ARCHIVE.md`
- Only the 3 most recent sessions are kept in the active file

### 1b. Doc Updator (Parallel with 1a)

Use the Task tool with `subagent_type: doc-updator` to suggest and apply updates to:
- `CLAUDE.md` for project-wide documentation
- Agent instructions if patterns changed
- New sections for undocumented features

The agent will edit `CLAUDE.md` directly.

**IMPORTANT:** Run 1a and 1b in parallel by including both Task tool calls in a single message.

---

## Phase 2: Review & Persist (After Phase 1 completes)

### 2a. Followup Suggestor

Use the Task tool with `subagent_type: followup-suggestor` to identify:
- Incomplete work items from this session
- Technical debt introduced
- Testing gaps
- Prioritized list for next session

The agent will:
- Write followups to `.claude/session-state/followups.md` (creating the directory if needed)
- Print followups to console
- Include complexity estimates (`simple`/`medium`/`complex`) for each item

### 2b. Automation Opportunity Triage

The followup-suggestor also handles automation triage:
- Reads `.claude/session-state/automation-log.md` if it exists
- Compares current session's automation opportunities against history
- Flags items appearing in 3+ sessions as `RECURRING`
- Appends current session's opportunities to `automation-log.md`
- Prints triage summary to console

---

## Phase 3: Consolidation (Run last)

### 3a. Doc Consolidator (Includes duplicate detection)

Use the Task tool with `subagent_type: doc-consolidator` to:

**Step 0 - Conflict marker resolution (FIRST):**
- Scan all target files for `<<<<<<<`, `=======`, `>>>>>>>`
- Resolve by removing markers, keeping both content sections
- Must complete before other consolidation work

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
- Migrate any `(lines X-Y)` references to section heading references

**Canonical File Hierarchy:**

| File | Owns | Should NOT contain |
|------|------|-------------------|
| `~/.claude/skills/*/SKILL.md` | Universal patterns, framework-agnostic | Project-specific details, session dates |
| `.claude/agents/*/SKILL.md` | Project patterns, session learnings | Universal patterns (link to global) |
| `.claude/CLAUDE.md` | Project architecture, design system | Agent-specific learnings |

The agent will edit files directly to consolidate documentation.

### 3b. Post-edit Validation

After the doc-consolidator finishes, verify the edits are clean:

1. **Re-check for conflict markers** introduced by edits:
   ```bash
   grep -rn '<<<<<<\|======\|>>>>>>' .claude/ ~/.claude/skills/ --include='*.md' || echo "No conflicts found"
   ```

2. **Check for unclosed code blocks** (odd count of ``` lines):
   ```bash
   for f in .claude/CLAUDE.md .claude/agents/*/SKILL.md ~/.claude/skills/*/SKILL.md; do
     [ -f "$f" ] && count=$(grep -c '```' "$f") && [ $((count % 2)) -ne 0 ] && echo "UNCLOSED CODE BLOCK: $f (${count} backtick lines)"
   done
   echo "Code block check complete"
   ```

3. **Report and fix** any issues found before proceeding.

---

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

   Co-Authored-By: Claude Code <noreply@anthropic.com>" && git push
   ```

3. **Report the result** in the final summary:
   - If changes were committed: list the files changed
   - If no changes: note that global skills were unchanged

**Note:** The global `~/.claude/` directory symlinks to `~/projects/nordic_claude_agent_skills/`:
- `~/.claude/skills/` → `nordic_claude_agent_skills/skills/`
- `~/.claude/agents/` → `nordic_claude_agent_skills/agents/`
- `~/.claude/commands/` → `nordic_claude_agent_skills/commands/`

---

## Final Summary

After all phases complete, provide a brief summary covering:

- **Pre-flight results:** Conflicts found/resolved, baseline file sizes
- **Key learnings captured** (with destination: global vs project)
- **Session archive actions** (any sessions moved to SKILL_ARCHIVE.md)
- **Documentation updates made**
- **Followup items persisted** to `.claude/session-state/followups.md`
- **Automation triage:** New opportunities, recurring items (3+ sessions)
- **Post-edit validation results** (conflicts, code blocks, reference checks)
- **File size deltas** (compare against Phase 0 baselines)
- **Global skills commit status** (committed/no changes)

---

## Optimization Notes

This workflow has been optimized from the original 6-agent sequential flow:

| Aspect | Before | After |
|--------|--------|-------|
| Agents | 6 sequential | 4 (2 parallel + 2 sequential) + validation + git commit |
| Validation | None | Pre-flight + post-edit checks |
| Followups | Console only (lost) | Persisted to `.claude/session-state/` |
| SKILL.md growth | Unbounded | 3-session cap with archive |
| Conflict markers | Ignored | Detected and resolved |
| Line references | Fragile `(lines X-Y)` | Section heading references |
| Co-author tag | Model-specific | `Claude Code` (future-proof) |

**Session State Convention:**

```
.claude/session-state/
  followups.md        # Latest session followups (overwritten each /wrap-session)
  checkpoint.md       # Latest checkpoint (overwritten each /checkpoint)
  automation-log.md   # Cumulative automation suggestions (appended each session)
```

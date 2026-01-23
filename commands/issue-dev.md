---
description: Fetch a GitHub issue, create a branch, and implement the feature
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite
---

# /issue-dev {issueNumber}

Feature implementation workflow that fetches a GitHub issue, creates a branch, and implements the changes.

## Workflow

### Step 1: Fetch Issue Details

```bash
gh issue view {issueNumber} --json title,body,labels,assignees
```

Parse the issue body for:
- Implementation requirements
- Acceptance criteria
- Files to modify (if mentioned)

### Step 2: Setup Branch

```bash
git fetch origin
git checkout -b issue_{issueNumber} origin/develop || git checkout -b issue_{issueNumber} origin/main
```

Note: Try `develop` first, fall back to `main` if not available.

### Step 3: Analyze Complexity

Determine complexity based on:
- Number of files mentioned in issue
- Scope of changes described
- Labels (e.g., "enhancement" vs "bug")

| Complexity | Criteria | Approach |
|------------|----------|----------|
| Simple | <3 files, clear fix | Work directly |
| Medium | 3-6 files, defined scope | Use domain agent if available |
| Complex | >6 files, architectural | Use domain agent with task breakdown |

### Step 4: Detect Available Agents

Check for project-specific agents in `.claude/agents/`:

```bash
ls -d .claude/agents/*/ 2>/dev/null | xargs -I {} basename {}
```

**Agent Selection Priority:**
1. If issue labels contain "frontend", "ui", "component" → use `frontend-dev` agent
2. If issue labels contain "backend", "api", "database" → use `backend-dev` agent
3. If issue labels contain "design", "ux" → use `ui-ux-designer` agent
4. If no matching agent or no agents exist → work directly

### Step 5: Implementation

**If using an agent:**
```
Use Task tool with subagent_type: {detected_agent}

Prompt:
"Implement GitHub issue #{issueNumber}: {title}

Issue Details:
{issue_body}

Branch: issue_{issueNumber}

Instructions:
1. Read and understand the requirements
2. Implement the changes following project patterns
3. Add/update tests as needed
4. Ensure all tests pass
5. Commit changes with message: feat(scope): {description} (#{issueNumber})
"
```

**If working directly:**
1. Create todo list from issue requirements
2. Implement changes file by file
3. Run tests: `npm run test` (or project equivalent)
4. Commit with conventional commit message

### Step 6: Completion

Output summary:

```
## Issue #{issueNumber} Implementation Complete

**Title:** {issue_title}
**Branch:** issue_{issueNumber}
**Complexity:** {simple|medium|complex}
**Agent Used:** {agent_name|none}

### Changes Made
- {list of files modified/created}

### Tests
- {test_status}

### Next Steps
Run `/run-pr-checks issue_{issueNumber}` to validate and create PR
```

## Notes

- Adapts to any project's agent configuration
- Falls back gracefully when no agents are defined
- Uses conventional commit format: `feat|fix|docs(scope): message (#issue)`
- Does NOT push or create PR (use `/run-pr-checks` for that)

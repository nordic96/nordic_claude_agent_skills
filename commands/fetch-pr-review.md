---
description: Fetch PR review comments and apply critical/high priority fixes
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite
---

# /fetch-pr-review {prNumber}

Fetches PR review comments and applies fixes for critical/high priority issues.

## Workflow

### Step 1: Fetch PR Data

```bash
# Get PR details
gh pr view {prNumber} --json title,body,headRefName,state,url

# Get review comments
gh api repos/{owner}/{repo}/pulls/{prNumber}/comments

# Get review summaries
gh api repos/{owner}/{repo}/pulls/{prNumber}/reviews
```

### Step 2: Checkout PR Branch

```bash
gh pr checkout {prNumber}
# or
git fetch origin
git checkout {headRefName}
git pull origin {headRefName}
```

### Step 3: Categorize Comments

Parse comments and categorize by priority:

**Priority Keywords:**

| Priority | Keywords |
|----------|----------|
| Critical | "must", "required", "blocking", "critical", "bug", "broken", "fails", "security vulnerability" |
| High | "should", "important", "security", "memory leak", "performance issue" |
| Medium | "consider", "suggest", "could", "might", "would be better", "recommend" |
| Low | "nit", "nitpick", "optional", "minor", "style", "preference", "typo" |

**Output categorized list:**
```
## Review Comments for PR #{prNumber}

### Critical (Must Fix) - {n} items
1. **{file}:{line}** - {comment_summary}
   > {full_comment}

### High Priority - {n} items
1. **{file}:{line}** - {comment_summary}

### Medium Priority (Suggestions) - {n} items
1. **{file}:{line}** - {comment_summary}

### Low Priority (Nits) - {n} items
1. **{file}:{line}** - {comment_summary}
```

### Step 4: Implementation Decision

| Total Fixes | Complexity | Approach |
|-------------|------------|----------|
| 1-3 simple | Low | Work directly |
| 4+ items | Medium | Use domain agent if available |
| Complex changes | High | Use domain agent with detailed breakdown |

**Agent detection:**
```bash
ls -d .claude/agents/*/ 2>/dev/null | xargs -I {} basename {}
```

### Step 5: Apply Fixes

**If using agent:**
```
Use Task tool with subagent_type: {detected_agent}

Prompt:
"Apply PR review fixes for PR #{prNumber}

Branch: {headRefName}

Critical Issues (MUST FIX):
{list_of_critical_issues}

High Priority Issues:
{list_of_high_issues}

Instructions:
1. Fix all Critical issues
2. Fix all High priority issues
3. Fix Medium priority if solution is clear (<5 lines)
4. Skip Low priority (nits) unless trivial
5. Run tests after fixes
6. Commit: fix(pr): address review feedback for #{prNumber}
7. Push changes
"
```

**If working directly:**
1. Create todo list from Critical + High items
2. Fix each issue in priority order
3. Run tests after each significant change
4. Commit and push

### Step 6: Validation

After fixes:
```bash
npm run test  # or project equivalent
npm run lint
npm run build
```

### Step 7: Commit and Push

```bash
git add -A
git commit -m "fix(pr): address review feedback for #{prNumber}

Co-Authored-By: Claude Code <noreply@anthropic.com>"

git push origin {headRefName}
```

### Step 8: Output

```
## PR #{prNumber} Review Fixes Applied

### Summary
| Priority | Found | Fixed | Skipped |
|----------|-------|-------|---------|
| Critical | {n} | {n} | 0 |
| High | {n} | {n} | {n} |
| Medium | {n} | {n} | {n} |
| Low | {n} | 0 | {n} |

### Fixes Applied
1. **{file}** - {issue_summary} ✓

### Skipped Items
- {file}:{line} - {reason: "needs clarification" | "design decision" | "nit"}

### Validation
- Tests: PASS ({n} tests)
- Lint: PASS
- Build: PASS

### Commit
{commit_hash} - fix(pr): address review feedback for #{prNumber}

### Next Steps
- PR updated: {pr_url}
- Request re-review from reviewers
```

## Notes

- Always fixes Critical and High priority items
- Medium priority fixed only if solution is obvious
- Low priority (nits) skipped unless trivial (1-line)
- Comments requiring clarification are flagged, not auto-fixed
- Adapts to any project's agent configuration

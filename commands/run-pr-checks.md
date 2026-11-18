---
description: Run lint, test, and build checks, then create a PR if all pass
allowed-tools: Bash, Read, Glob, Grep
---

# /run-pr-checks {branchName}

Pre-push validation workflow that runs all checks and creates a PR.

## Workflow

### Step 0: Check for Git Hooks (Husky)

**FIRST, check if Husky or similar git hooks are configured:**

```bash
# Check for Husky setup
ls -d .husky 2>/dev/null && echo "HUSKY_FOUND"

# Or check package.json for husky dependency
grep -q '"husky"' package.json 2>/dev/null && echo "HUSKY_IN_DEPS"
```

**If Husky is found:**
- Skip Steps 1-3 (validation checks are handled by git hooks)
- Proceed directly to Step 4 (Push Branch)
- The pre-commit and pre-push hooks will automatically run lint, test, and build

**If NO Husky:**
- Continue with full validation workflow (Steps 1-3)

---

## Path A: Husky Detected (Fast Path)

When Husky is configured, git hooks handle validation:
- **Pre-commit hook**: Runs lint and tests
- **Pre-push hook**: Runs build

Simply push the branch and create the PR:

```bash
# Push branch (hooks will run automatically)
git push -u origin {branchName}

# Create PR
gh pr create --title "{pr_title}" --body "{pr_body}" --base {base_branch}
```

**Output for Husky path:**
```
## PR Checks Complete (Husky Mode)

Git hooks detected - validation handled by Husky.

### PR Created
- **URL:** {pr_url}
- **Title:** {pr_title}
- **Base:** {base_branch}
- **Closes:** #{issueNumber} (if applicable)
```

---

## Path B: No Husky (Full Validation)

### Step 1: Detect Project Type

Check for package manager and scripts:

```bash
# Check for package.json scripts
cat package.json 2>/dev/null | grep -E '"(test|lint|build)"'

# Or check for other build tools
ls Makefile Cargo.toml pyproject.toml go.mod 2>/dev/null
```

### Step 2: Run Validation Checks

Execute checks based on project type:

**Node.js/npm projects:**
```bash
npm run lint 2>&1
npm run test 2>&1
npm run build 2>&1
```

**Python projects:**
```bash
ruff check . 2>&1 || pylint **/*.py 2>&1
pytest 2>&1
```

**Go projects:**
```bash
go fmt ./... 2>&1
go test ./... 2>&1
go build ./... 2>&1
```

**Rust projects:**
```bash
cargo fmt --check 2>&1
cargo test 2>&1
cargo build --release 2>&1
```

### Step 3: Handle Failures

| Check | Auto-Fix? | Action |
|-------|-----------|--------|
| Lint errors | Yes (if available) | Run `npm run lint:fix` or equivalent, commit fixes |
| Type errors | No | Report errors, stop workflow |
| Test failures | No | Report failures, stop workflow |
| Build errors | No | Report errors, stop workflow |

**Auto-fix flow:**
```bash
# If lint failed and auto-fix available
npm run lint:fix
git add -A
git commit -m "fix: auto-fix lint errors"
# Re-run checks
```

### Step 4: Push Branch

Only if all checks pass (or Husky is handling validation):

```bash
git push -u origin {branchName}
```

### Step 5: Create PR

**Extract issue number from branch:**
```bash
# Handles "issue_123", "issue_#123", "issue-123", "feature_123" — extract trailing number
echo "{branchName}" | grep -oE '[0-9]+$'
```

**Get issue details (if applicable):**
```bash
gh issue view {issueNumber} --json title,body 2>/dev/null
```

**Detect base branch:**
```bash
# Use --base argument if provided, otherwise try develop, fall back to main
# Example: /run-pr-checks --base develop_v5
git ls-remote --heads origin develop && echo "develop" || echo "main"
```

**Create PR with issue link:**
```bash
gh pr create \
  --title "{pr_title}" \
  --body "{pr_body}" \
  --base {base_branch}
```

**IMPORTANT: Issue Linking**
- If branch name contains an issue number (e.g., `issue_#439`, `feature_123`), ALWAYS include `Closes #{issueNumber}` in the PR body
- This automatically links and closes the issue when the PR is merged
- The issue number should also be included in the PR title if it follows a pattern like "Phase 6: ... (#439)"

**PR Body Template:**
```markdown
## Summary
{Brief description from issue or commits}

## Changes
- {List key changes from git diff --stat}

## Checklist
- [x] Lint passing
- [x] Tests passing ({n} tests)
- [x] Build successful
- [ ] Manual testing

Closes #{issueNumber}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

**Note:** The `Closes #X` keyword is REQUIRED when the branch is linked to an issue. GitHub recognizes these keywords to auto-close issues:
- `Closes #123`
- `Fixes #123`
- `Resolves #123`

### Step 5b: Verify & Ensure the Issue Is Linked

**Critical caveat:** GitHub only auto-closes an issue from a `Closes #X` keyword when the PR
merges into the repository's **default branch** (usually `main`). When the PR targets a
non-default base (e.g. `develop`), the keyword often does **not** populate the PR's
**Development** panel, so the issue ends up unlinked and won't auto-close on merge.

After creating the PR, always verify the link and add a fallback cross-reference if it's
missing, so the issue and PR are connected in both timelines:

```bash
PR_NUMBER=$(gh pr view {branchName} --json number -q .number)
OWNER_REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
OWNER=${OWNER_REPO%/*}; REPO=${OWNER_REPO#*/}

LINKED=$(gh api graphql -f query='
  query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){ closingIssuesReferences(first:10){ nodes{ number } } }
    }
  }' -F owner="$OWNER" -F repo="$REPO" -F pr="$PR_NUMBER" \
  -q '.data.repository.pullRequest.closingIssuesReferences.nodes[].number' 2>/dev/null)

if echo "$LINKED" | grep -qx "{issueNumber}"; then
  echo "Issue #{issueNumber} is formally linked to PR #$PR_NUMBER."
else
  # Fallback when base branch is non-default: cross-reference both ways and label the PR
  gh issue comment {issueNumber} --body "Addressed by PR #$PR_NUMBER."
  echo "WARNING: closing-keyword link not registered (non-default base branch)."
  echo "Added a cross-reference comment; link manually via the PR's Development panel if you need auto-close."
fi
```

Report in the final output whether the issue is **formally linked** or only
**cross-referenced** (non-default base branch), so the user knows if auto-close on merge will work.

### Step 6: Output

**Success (Full Validation):**
```
## PR Checks Complete

### Validation Results
| Check | Status |
|-------|--------|
| Lint | PASS |
| Tests | PASS ({n} tests) |
| Build | PASS |

### PR Created
- **URL:** {pr_url}
- **Title:** {pr_title}
- **Base:** {base_branch}
- **Closes:** #{issueNumber} (if applicable)
```

**Failure:**
```
## PR Checks Failed

### Failed Checks
- {check}: {error_summary}

### Action Required
{Guidance on fixing the issue}

### To Retry
Fix the issues and run `/run-pr-checks {branchName}` again
```

## Notes

- **Husky detection**: Checks for `.husky/` directory or `husky` in package.json
- **Fast path**: When Husky is found, skips manual validation (hooks handle it)
- Auto-detects project type and appropriate commands
- Only auto-fixes lint errors (safe operation)
- Never auto-fixes tests or type errors
- Creates PR only when ALL checks pass
- Detects base branch automatically (develop or main), or use `--base` argument
- **ALWAYS links issue** when branch name contains issue number (e.g., `issue_#439`)
- Uses `Closes #X` keyword in PR body to auto-close linked issues on merge
- **Verifies the link after creating the PR** (Step 5b) via `closingIssuesReferences`; the
  `Closes #X` keyword only links/auto-closes against the **default branch**, so for PRs
  targeting `develop` (or any non-default base) it adds a cross-reference comment as a
  fallback and reports whether the issue is formally linked vs only cross-referenced

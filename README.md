# Nordic Claude Agent Skills

A centralized repository for Claude Code agents, global skills, and workflow commands used across multiple projects.

## Repository Structure

```
nordic_claude_agent_skills/
├── agents/                              # Agent definitions
│   ├── frontend-dev/
│   │   └── frontend-dev.md              # Global frontend developer agent
│   ├── ui-ux-designer/
│   │   └── ui-ux-designer.md            # Global UI/UX designer agent
│   ├── learning-extractor/
│   │   └── learning-extractor.md        # Session learning extraction
│   ├── doc-consolidator/
│   │   └── doc-consolidator.md          # Documentation consolidation
│   ├── doc-updator/
│   │   └── doc-updator.md               # Documentation updates
│   ├── duplicate-checker/
│   │   └── duplicate-checker.md         # Duplicate content detection
│   ├── followup-suggestor/
│   │   └── followup-suggestor.md        # Follow-up task suggestions
│   └── automation-scout/
│       └── automation-scout.md          # Automation opportunity detection
├── skills/                              # Global skills (tech-agnostic)
│   ├── frontend-dev/
│   │   └── SKILL.md                     # Universal frontend patterns
│   └── ui-ux-designer/
│       └── SKILL.md                     # Universal design patterns
└── commands/                            # Workflow commands
    ├── wrap-session.md                  # Session wrap-up workflow
    ├── run-pr-checks.md                 # PR validation and creation
    ├── issue-dev.md                     # Issue-based development
    └── fetch-pr-review.md               # PR review feedback handling
```

## Usage

### Installation

Create symlinks from `~/.claude/` to this repository:

```bash
# Backup existing (if any)
mv ~/.claude/agents ~/.claude/agents.bak 2>/dev/null
mv ~/.claude/commands ~/.claude/commands.bak 2>/dev/null

# Create symlinks
ln -s /path/to/nordic_claude_agent_skills/agents ~/.claude/agents
ln -s /path/to/nordic_claude_agent_skills/skills ~/.claude/skills
ln -s /path/to/nordic_claude_agent_skills/commands ~/.claude/commands
```

### Context Loading Order

Agents use a layered context system:

1. **Global Skills** (`~/.claude/skills/{agent}/SKILL.md`) - Universal patterns applicable across all projects
2. **Project Skills** (`.claude/agents/{agent}/SKILL.md`) - Project-specific patterns and session learnings
3. **Project CLAUDE.md** (`.claude/CLAUDE.md`) - Project architecture and design system

This hierarchy ensures:
- Common wisdom is shared across projects
- Project-specific context is preserved
- No duplication of universal patterns in each project

### Wrap-Session Dual-Destination Routing

The `wrap-session` command automatically classifies learnings:

| Pattern Type | Destination | Example |
|--------------|-------------|---------|
| CSS/JS gotcha | Global Skills | "Transforms are atomic" |
| WCAG rule | Global Skills | "4.5:1 contrast ratio" |
| Component hook | Project Skills | "useStaggeredAnimation for v5" |
| Design decision | Project Skills | "Glass card 30% opacity" |
| Library API quirk | Global Skills | "simple-icons SVG format" |
| Project integration | Project Skills | "Vercel build script" |

## Agents

### Development Agents

- **@frontend-dev** - Expert frontend developer for Next.js, React, TypeScript, and Tailwind CSS
- **@ui-ux-designer** - Expert UI/UX designer for design analysis and specifications

### Session Management Agents

- **@learning-extractor** - Extracts learnings, mistakes, and debugging patterns
- **@doc-updator** - Updates project documentation (CLAUDE.md)
- **@doc-consolidator** - Consolidates duplicate content across documentation
- **@duplicate-checker** - Detects redundant content in documentation
- **@followup-suggestor** - Identifies incomplete work and next steps
- **@automation-scout** - Finds automation opportunities

## Commands

- **`/wrap-session`** - End-of-session workflow that extracts learnings, updates docs, and suggests follow-ups
- **`/run-pr-checks`** - Validates code (lint, test, build) and creates PR
- **`/issue-dev`** - Fetches GitHub issue and implements the feature
- **`/fetch-pr-review`** - Fetches PR review comments and applies fixes

## Contributing

When adding new patterns:

1. Determine if the pattern is **global** (applicable to any project) or **project-specific**
2. Global patterns go in `skills/{agent}/SKILL.md`
3. Project-specific patterns stay in the project's `.claude/agents/{agent}/SKILL.md`

### Classification Guidelines

**Global (add to this repo):**
- Programming language gotchas
- Framework-agnostic best practices
- Accessibility standards (WCAG)
- Performance optimization techniques
- Design principles and theory

**Project-Specific (keep in project):**
- Component implementations
- Project architecture decisions
- Session-dated learnings
- Tool configurations
- Design system specifics (colors, typography)

## License

MIT

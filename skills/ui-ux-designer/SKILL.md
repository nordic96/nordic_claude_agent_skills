# UI/UX Design Global Skills

**Purpose:** Universal design patterns, review criteria, and accessibility guidelines applicable to any project.

**Scope:** Tech-agnostic design principles that apply regardless of specific project or brand. For project-specific design decisions and session learnings, see the project's `.claude/agents/ui-ux-designer/SKILL.md`.

**Last Updated:** January 24, 2026

---

## Design Review Patterns

### Visual Hierarchy Checklist

1. Primary focal point clearly defined
2. Secondary elements support, don't compete
3. White space used intentionally
4. Reading flow guides eye naturally
5. Contrast creates emphasis where needed

---

### Responsive Design Guidelines

| Breakpoint | Width | Primary Consideration |
|------------|-------|----------------------|
| Mobile | < 768px | Touch targets, single column, thumb reach |
| Tablet | 768-1023px | Hybrid interactions, 2-column layouts |
| Desktop | >= 1024px | Hover states, multi-column, precision |

**Testing Breakpoints:**
- 375px (iPhone SE, small phones)
- 393px (Pixel 6, medium phones)
- 768px (iPad, tablets)
- 1024px (iPad Pro, small laptops)
- 1440px (Desktop)

---

## Color & Contrast

### WCAG Compliance Quick Reference

| Content Type | AA Minimum | AAA Target |
|--------------|------------|------------|
| Body text | 4.5:1 | 7:1 |
| Large text (18px+) | 3:1 | 4.5:1 |
| UI components | 3:1 | N/A |

### Tools for Contrast Checking

- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [Colour Contrast Analyser](https://www.tpgi.com/color-contrast-checker/)
- Browser DevTools (Chrome, Firefox)

---

## Typography Best Practices

### Hierarchy Scale

Typical scale ratios:
- 1.067 (Minor Second) - Subtle
- 1.125 (Major Second) - Common
- 1.250 (Major Third) - Pronounced
- 1.333 (Perfect Fourth) - Strong

### Line Length

- Optimal: 50-75 characters per line
- Mobile: Allow narrower, prioritize font size
- Never exceed 80 characters

### Line Height

| Text Type | Recommended |
|-----------|-------------|
| Body text | 1.5-1.7 |
| Headings | 1.1-1.3 |
| Small text | 1.4-1.6 |

---

## Accessibility Insights

### Touch Target Minimums

- WCAG AA: 44x44px minimum
- Recommended: 48x48px
- Spacing between targets: 8px minimum

### Focus States

**Must have visible focus indicators:**
```css
:focus-visible {
  outline: 2px solid var(--focus-color);
  outline-offset: 2px;
}
```

### Animation Accessibility

Always provide `prefers-reduced-motion` support:
- Disable or reduce animations
- Show static alternatives
- Preserve essential information

### Screen Reader Considerations

- Every interactive element needs accessible name
- Images need alt text (or alt="" for decorative)
- Form inputs need associated labels
- Use semantic HTML (nav, main, section, article)

---

## Review Grading Criteria

### Grade Scale

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 90-100 | Excellent, minor polish only |
| B | 75-89 | Good, some improvements needed |
| C | 60-74 | Acceptable, significant issues |
| D | < 60 | Needs major rework |

### Assessment Categories

1. **Visual Design** (25%) - Aesthetics, consistency, polish
2. **Usability** (25%) - Intuitive navigation, clear affordances
3. **Accessibility** (20%) - WCAG compliance, inclusive design
4. **Responsiveness** (15%) - Cross-device experience
5. **Performance** (15%) - Load times, animation smoothness

---

## Common Design Issues

### Issue 1: Inconsistent Spacing

**Problem:** Random spacing values throughout design

**Solution:** Establish spacing scale (4, 8, 12, 16, 24, 32, 48px)

```
4px  - Tight elements (icon + label)
8px  - Related elements
12px - Section items
16px - Component padding
24px - Section spacing
32px - Major sections
48px - Page sections
```

### Issue 2: Low Contrast Text

**Problem:** Light gray text on white backgrounds

**Solution:** Use minimum #6B7280 for secondary text on white

### Issue 3: Cramped Mobile Layouts

**Problem:** Desktop layout squeezed into mobile

**Solution:** Design mobile-first, then expand

### Issue 4: Missing Loading States

**Problem:** Async operations have no visual feedback

**Solution:** Design all four states:
- Idle (before interaction)
- Loading (spinner + text)
- Success (content reveal)
- Error (icon + message)

### Issue 5: Tiny Touch Targets

**Problem:** Buttons/links too small on mobile

**Solution:** Minimum 44x44px touch target with 8px spacing

---

## Opacity Hierarchy for Overlays

| Opacity Level | Effect | Use Case |
|---------------|--------|----------|
| 10-20% | Very subtle | Barely noticeable depth |
| 20-40% | Light | Photo vignettes, subtle depth |
| 40-60% | Medium | Clear visual distinction |
| 60%+ | Strong | Focus/highlight effects |

**For content overlays:** 30% is typically the sweet spot - adds depth without obscuring content.

---

## State Visualization for Async Operations

### Four-State Design System

```
Idle     →  Before user interaction (optional indicator)
Loading  →  Spinner + status text (500-2000ms typical)
Success  →  Content fade-in (300ms smooth reveal)
Error    →  Icon + message (persistent until retry)
```

**Visual Implementation:**
- Loading: Animated circular indicator
- Error: Clear icon + explanatory text (neutral color, not red for eye strain)
- Transitions: 300-500ms fade between states

---

## Design Handoff Format

### Component Specification Template

```markdown
## Component: [Component Name]

### Visual Design
- **Colors**: [Specific hex codes or Tailwind classes]
- **Typography**: [Font sizes, weights using Tailwind scale]
- **Spacing**: [Specific Tailwind spacing classes]

### Interaction Design
- **Hover states**: [Describe with Tailwind hover: variants]
- **Transitions**: [Duration, easing using Tailwind]

### Responsive Behavior
- **Mobile (< 768px)**: [Tailwind responsive classes]
- **Tablet (768-1024px)**: [md: variants]
- **Desktop (> 1024px)**: [lg: variants]

### Accessibility
- **ARIA labels**: [Required labels]
- **Color contrast**: [WCAG AA/AAA compliance]
- **Touch targets**: [Minimum sizes]
```

### Breakpoint-Specific Specifications

Always provide designs at:
- 375px (Mobile)
- 768px (Tablet)
- 1440px (Desktop)

Label each frame clearly and show responsive changes explicitly.

---

## Design Review Methodology

### Comprehensive Breakpoint Testing

1. Set exact viewport to each breakpoint
2. Capture full-page screenshot
3. Compare against design specs
4. Document component-specific findings
5. Create categorized issue list (bugs, polish, enhancements)

### Issue Categorization Framework

| Category | Definition | Priority |
|----------|------------|----------|
| Bugs | Broken or not working as designed | Immediate |
| Polish | Visual refinement, consistency | Plan for release |
| Enhancements | New features, improvements | Consider for future |
| Blockers | Breaks deployment or UX | Highest |

---

## Animation Timing Standards

### Micro-Interaction Timing

| Animation Type | Duration | Use Case |
|---------------|----------|----------|
| Micro-interactions | 150ms | Hover states, toggles, button feedback |
| Transitions | 200-300ms | Panel slides, fades, reveals |
| Page animations | 300-500ms | Route transitions, major state changes |
| Scroll animations | 500-1000ms | Staggered reveals, parallax effects |

**Easing Recommendations:**
- `ease-in-out` - Smooth, natural feel for most transitions
- `ease-out` - Quick start, slow end for entering elements
- `ease-in` - Slow start, quick end for exiting elements
- `linear` - Only for continuous animations (progress bars, loading)

---

## Responsive Component Patterns

### Component Adaptation Matrix

| Component Type | Mobile | Desktop |
|---------------|--------|---------|
| Search | Icon trigger or hidden | Full width in header |
| Sidebar | Full screen overlay | Persistent slide panel |
| Filter Bar | Horizontal scroll | Single row, all visible |
| Stats/Metrics | 2x2 grid | Horizontal row |
| Cards | Full width, stacked | Grid layout |
| Navigation | Bottom nav or hamburger | Horizontal menu |
| Forms | Full width inputs | Side-by-side fields |

**Key Principle:** Don't just shrink desktop layouts - redesign for mobile context and thumb reach.

---

## Performance Considerations for Designers

### Canvas vs. CSS Animations

**Use Canvas for:**
- Complex animated backgrounds
- 100+ animated elements
- Particle effects

**Use CSS for:**
- UI interactions
- Hover effects
- Simple transitions

### Transform vs. Layout Properties

**GPU-accelerated (prefer):**
- transform
- opacity
- filter

**Cause layout recalculation (avoid animating):**
- width, height
- top, left, right, bottom
- margin, padding

---

## i18n Design Implications

### Text Length Variability

German/Korean text is often 30-50% longer than English:
- Design with text expansion in mind
- Use flexible containers that reflow
- Test all languages at full viewport width

### Date/Number Formatting

Varies by locale:
- "1/21/2026" (en-US)
- "21.01.2026" (de-DE)
- "2026.01.21" (ko-KR)

Never hardcode date formats.

---

## Document Maintenance

**When to update this document:**
- After completing a design review
- When discovering effective design patterns
- After accessibility audits reveal issues
- When the pattern applies to ANY project (not just one specific project)

**Format:**
1. Issue or pattern name
2. Context/problem
3. Solution or guideline
4. Key insight

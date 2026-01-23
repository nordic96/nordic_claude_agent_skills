# Frontend Development Global Skills

**Purpose:** Universal frontend patterns, CSS gotchas, and best practices applicable to any project.

**Scope:** Tech-agnostic patterns that apply regardless of specific framework or project. For project-specific patterns and session learnings, see the project's `.claude/agents/frontend-dev/SKILL.md`.

**Last Updated:** January 24, 2026

---

## CSS & Styling

### CSS Transforms Are Atomic (Not Additive)

**Problem:** CSS transforms override each other instead of combining.

**Wrong:**
```css
.element {
  transform: translate(-50%, -50%);
}
.element:hover {
  transform: scale(1.15); /* This removes translate! */
}
```

**Correct:**
```css
.element {
  transform: translate(-50%, -50%) rotate(var(--rotation, 0deg));
}
.element:hover {
  transform: translate(-50%, -50%) rotate(var(--rotation, 0deg)) scale(1.15);
}
```

**Key Insight:** Always include ALL transforms in every transform declaration. Use CSS custom properties for dynamic values.

---

### Transform Order Matters

The order of transform operations affects the final result:

```css
/* Order: translate → rotate → scale */
transform: translate(-50%, -50%) rotate(15deg) scale(1.15);
```

- **translate** - Move the element first
- **rotate** - Rotate around the (new) center
- **scale** - Scale last to avoid position drift

---

### Aspect Ratio Correction for Circles

When positioning elements in a circle on non-square containers, X-axis needs compensation:

```typescript
// Without correction: ellipse
const x = centerX + (radius * Math.cos(angle));

// With correction: perfect circle
const radiusX = radius * 0.7; // Aspect ratio correction
const x = centerX + (radiusX * Math.cos(angle));
```

**When to use:** Any circular positioning on rectangular containers (hero sections, cards, etc.)

---

## Animation Patterns

### Staggered Animation with Cleanup

```typescript
useEffect(() => {
  if (!animation) {
    setVisibleLogos(new Set(logos.map((_, i) => i)));
    return;
  }

  const interval = setInterval(() => {
    setVisibleLogos((prev) => {
      const next = new Set(prev);
      if (next.size < logos.length) {
        next.add(next.size);
      }
      return next;
    });
  }, DELAY_INTERVAL);

  return () => clearInterval(interval); // Critical: cleanup!
}, [animation, logos.length]);
```

**Key Points:**
- Always return cleanup function from useEffect
- Use Set for O(1) visibility checks
- Handle non-animated case (instant display)

---

### Reduced Motion Accessibility

**Always respect user preferences:**

```css
@media (prefers-reduced-motion: reduce) {
  .animated-element {
    animation: fadeInStatic 0.01ms forwards !important;
  }
}

@keyframes fadeInStatic {
  to {
    opacity: 1;
    transform: none;
  }
}
```

**In JavaScript:**
```typescript
const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
isReducedMotionRef.current = mediaQuery.matches;

const handleMotionChange = (e: MediaQueryListEvent) => {
  isReducedMotionRef.current = e.matches;
  if (e.matches) {
    cancelAnimationFrame(animationFrameRef.current);
    // Draw static version
  } else {
    animationFrameRef.current = requestAnimationFrame(animate);
  }
};
mediaQuery.addEventListener('change', handleMotionChange);
```

**Why:** Users with vestibular disorders need instant display instead of animations.

---

## Circular/Orbital Positioning

### Mathematical Circle Placement

```typescript
const calculatePositions = (totalItems: number, radiusPercent: number) => {
  const angleStep = (2 * Math.PI) / totalItems;
  const startAngle = -Math.PI / 2; // Start at 12 o'clock

  return Array.from({ length: totalItems }, (_, index) => {
    const angle = startAngle + index * angleStep;
    return {
      x: 50 + (radiusPercent * 0.7 * Math.cos(angle)), // 0.7 = aspect correction
      y: 50 + (radiusPercent * Math.sin(angle)),
      rotation: (angle * 180 / Math.PI) + 90, // Convert to degrees
    };
  });
};
```

**Parameters:**
- `startAngle = -Math.PI / 2` → Starts at top (12 o'clock)
- `angleStep = 2π / n` → Even distribution
- Multiply X by 0.7 for aspect ratio correction

---

## Component Architecture

### Props with Sensible Defaults

```typescript
interface ComponentProps {
  animation?: boolean;    // default: true
  radius?: number;        // default: 35 (%)
  centerX?: number;       // default: 50 (%)
  centerY?: number;       // default: 50 (%)
}
```

Make components configurable but usable out-of-the-box.

---

### useMemo for Expensive Calculations

```typescript
const logoPositions = useMemo(() => {
  return calculateCircularPositions(techLogos.length, radius, centerX, centerY);
}, [radius, centerX, centerY]); // Only recalculate when these change
```

**When to use:** Trigonometric calculations, array transformations, complex mappings.

---

## Third-Party Libraries

### Simple Icons for Tech Logos

**Best source for official brand logos:**

```bash
npm install simple-icons
```

```typescript
import { siReact, siTypescript } from 'simple-icons';

// Usage with SVG string
<div dangerouslySetInnerHTML={{ __html: siReact.svg }} />
```

**Benefits:**
- MIT Licensed (copyright safe)
- 2800+ brand logos
- Consistent style
- Small file sizes

**SVG Sizing Gotcha:**
- SVG elements from SimpleIcons may have fixed dimensions
- Apply `[&>svg]:w-full [&>svg]:h-full` to force SVG to fill container

---

### Flag Icons for Locale Switcher

```bash
npm install flag-icons
```

```css
@import 'flag-icons/css/flag-icons.min.css';
```

```tsx
<span className={`fi fi-${countryCode}`} />
```

**Note:** Country codes are ISO 3166-1-alpha-2 (e.g., `us`, `kr`, `jp`).

---

## Internationalization (i18n)

### next-intl Setup Pattern

```typescript
// i18n/routing.ts
export const routing = defineRouting({
  locales: ['en', 'ko'],
  defaultLocale: 'en',
});

// i18n/request.ts
export default getRequestConfig(async ({ requestLocale }) => {
  let locale = await requestLocale;
  if (!locale || !routing.locales.includes(locale as Locale)) {
    locale = routing.defaultLocale;
  }
  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
  };
});
```

---

### Locale Switcher Accessibility

**Always include:**
- `lang` attribute on locale buttons
- `aria-label` describing the action
- Visual indicator of current locale (not just color)

```tsx
<button
  lang={locale}
  aria-label={`Switch to ${langName}`}
  aria-pressed={isActive}
>
  <span className={`fi fi-${countryCode}`} aria-hidden="true" />
  <span className="sr-only">{langName}</span>
</button>
```

---

## Responsive Design

### Mobile-First Breakpoints

```css
/* Base: Mobile (< 768px) */
.element { width: 100%; }

/* Tablet (≥ 768px) */
@media (min-width: 768px) {
  .element { width: 50%; }
}

/* Desktop (≥ 1024px) */
@media (min-width: 1024px) {
  .element { width: 33%; }
}
```

**Tailwind:**
```tsx
<div className="w-full md:w-1/2 lg:w-1/3" />
```

---

### Dynamic Viewport Units

**Problem:** `100vh` doesn't account for mobile browser chrome.

**Solution:** Use dynamic viewport units:

```css
.hero {
  height: 100dvh; /* Dynamic viewport height */
}
```

**Tailwind:**
```tsx
<section className="h-dvh" />
```

---

### Tailwind Best Practices

- Prefer standard Tailwind classes over arbitrary values
- Example: Use `p-2.5` instead of `p-[10px]`
- Example: Use `w-5` instead of `w-[20px]`
- Standard classes are optimized by Tailwind's JIT compiler
- Arbitrary values should only be used when no standard class exists

---

## Performance

### GPU-Accelerated Animations

```css
.animated {
  will-change: transform, opacity;
  backface-visibility: hidden;
}
```

**Remove after animation:**
```css
.animation-complete {
  will-change: auto;
}
```

**GPU-accelerated properties:**
- `transform`
- `opacity`
- `filter`

**Avoid animating:**
- `width`, `height`
- `top`, `left`, `right`, `bottom`
- `margin`, `padding`

---

### High-DPI Canvas Rendering

```typescript
const dpr = window.devicePixelRatio || 1;
canvas.width = width * dpr;
canvas.height = height * dpr;
canvas.style.width = `${width}px`;
canvas.style.height = `${height}px`;
const ctx = canvas.getContext('2d');
ctx.scale(dpr, dpr);
```

**Why:** Makes canvas sharp on retina/high-DPI displays.

---

### SVG Over PNG for Icons

**Prefer SVGs because:**
- Infinitely scalable
- Smaller file size
- CSS-styleable
- No retina considerations

---

## Accessibility Checklist

### Decorative Elements

```tsx
<div aria-hidden="true">
  {/* Decorative content hidden from screen readers */}
</div>
```

### Focus Management

```css
/* Visible focus indicator */
:focus-visible {
  outline: 2px solid var(--accent-color);
  outline-offset: 2px;
}

/* Remove outline for mouse users */
:focus:not(:focus-visible) {
  outline: none;
}
```

### Color Contrast Minimums

| Context | WCAG AA | WCAG AAA |
|---------|---------|----------|
| Normal text | 4.5:1 | 7:1 |
| Large text (18px+) | 3:1 | 4.5:1 |
| UI components | 3:1 | N/A |

### Touch Target Sizing

Minimum touch target size for WCAG AA: **44x44px**

```tsx
<button className="min-w-[44px] min-h-[44px] p-3">
  Click me
</button>
```

---

## Common Gotchas

### 1. Interval Memory Leaks

**Always clean up intervals:**
```typescript
useEffect(() => {
  const id = setInterval(() => {}, 1000);
  return () => clearInterval(id); // Required!
}, []);
```

### 2. Hydration Mismatches

**Avoid:**
- `Math.random()` in render
- `new Date()` without useEffect
- Browser-only APIs during SSR

**Solution:** Use `useEffect` for client-only code.

### 3. Z-Index Wars

**Use consistent z-index scale:**
```css
:root {
  --z-base: 0;
  --z-dropdown: 100;
  --z-modal: 200;
  --z-tooltip: 300;
  --z-toast: 400;
}
```

### 4. SVG viewBox Case Sensitivity

```tsx
// Wrong (JSX)
<svg viewbox="0 0 24 24">  // lowercase 'b' won't work

// Correct
<svg viewBox="0 0 24 24">  // capital 'B' required
```

### 5. React.FC Typing (Deprecated Pattern)

**Avoid:**
```typescript
const Component: React.FC<Props> = (props) => { ... }
```

**Prefer:**
```typescript
function Component(props: Props) { ... }
// or
const Component = (props: Props): JSX.Element => { ... }
```

---

## Intersection Observer Pattern

### Lazy Loading with Intersection Observer

```typescript
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        setIsInView(true);
        setLoadingState('loading');
        observer.disconnect(); // Load only once
      }
    });
  },
  { rootMargin: '100px', threshold: 0.1 }
);
```

**Key Options:**
- `rootMargin: '100px'` - Start loading before visible
- `threshold: 0.1` - Trigger at 10% visibility
- `disconnect()` - Clean up after first intersection

---

## Document Maintenance

**When to update this document:**
- After solving a tricky bug
- When discovering a better pattern
- After learning a new library quirk
- When the pattern applies to ANY project (not just one specific project)

**Format:**
1. Problem description
2. Wrong approach (if applicable)
3. Correct solution
4. Key insight/takeaway

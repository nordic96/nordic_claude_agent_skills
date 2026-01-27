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

### Input and Button Labels

**Always add `aria-label` to interactive elements without visible text:**

```tsx
// Interactive input without visible label
<input
  type="text"
  aria-label="Search chat messages"
  placeholder="Type here..."
/>

// Button that might only have an icon
<button
  aria-label="Send message"
  type="submit"
>
  <SendIcon />
</button>
```

**Why:** Screen reader users need to know what each control does. Placeholder text is not sufficient as it's not always announced.

---

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

## React 19 Patterns

### Activity Component vs Conditional Rendering

**Problem:** React 19's `<Activity>` component hides content visually but still mounts children.

**Anti-Pattern:**
```typescript
// WRONG - Component still mounts and runs side effects
<Activity mode={isSearchVisible}>
  <SearchBar /> {/* Loads 50MB model even when hidden */}
</Activity>
```

**Correct Pattern:**
```typescript
// CORRECT - Component only mounts when condition is true
{isSearchVisible && <SearchBar />}
```

**Key Insight:**
- `<Activity>` controls visibility, NOT mounting
- Use for UX state (hiding/showing without unmounting)
- For performance (preventing resource loads), use true conditionals
- Heavy components should always use conditional rendering

---

### Lazy Initialization for Heavy Resources

**Problem:** Auto-initializing heavy resources in constructors blocks page load.

**Anti-Pattern:**
```typescript
// WRONG - Blocks creation, loads resources immediately
class HeavyService {
  constructor() {
    this.initWorker(); // 50MB model download on any page!
  }
}
```

**Correct Pattern:**
```typescript
// CORRECT - Constructor lightweight, init deferred
class HeavyService {
  private worker: Worker | null = null;
  private ready = false;

  constructor() {
    // Don't initialize here
  }

  private async ensureWorker() {
    if (!this.worker) {
      this.worker = await this.loadHeavyResource();
      this.ready = true;
    }
  }

  async doWork(data: Data) {
    await this.ensureWorker(); // Lazy trigger on first use
    // Process data...
  }

  getReadyState() {
    return this.ready;
  }
}
```

**Key Insight:** Never auto-initialize heavy resources in constructors. Defer to lazy init on first actual use.

---

### Singleton Factory with Lazy Initialization

```typescript
let instance: HeavyService | null = null;

export function getHeavyServiceInstance(): HeavyService {
  if (!instance) {
    instance = new HeavyService();
  }
  return instance;
}
```

**Key Insight:** Combined with lazy init in the class, the singleton is created on first request but doesn't load resources until actually needed.

---

## Hook Patterns

### useDebounce - Handling Stale Closure on Delay Changes

**Problem:** When `useDebounce` hook receives a new `delay` prop, the debounce timeout still references the old delay, causing stale behavior.

**Anti-Pattern (Stale Closure):**
```typescript
function useDebounce(value: string, delay: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);  // Captured delay is now stale!

    return () => clearTimeout(timer);
  }, [value]); // Missing delay dependency
}
```

When `delay` changes, the old timeout still uses the OLD delay value because the effect doesn't re-run.

**Correct Pattern - Update ref when delay changes:**
```typescript
function useDebounce(value: string, delay: number) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  const delayRef = useRef(delay);

  // Update ref whenever delay prop changes
  useEffect(() => {
    delayRef.current = delay;
  }, [delay]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delayRef.current);  // Always uses current delay

    return () => clearTimeout(timer);
  }, [value]); // Refs don't trigger re-renders, can exclude from deps
}
```

**Key Insight:** Refs update synchronously without triggering re-renders. Use them to store values that should be accessible in closures without forcing dependency re-runs.

---

### Timer Cleanup with useRef

**Problem:** Memory leaks from uncleaned timers when component unmounts.

**Anti-Pattern:**
```typescript
// WRONG - No cleanup, causes memory leak
useEffect(() => {
  setTimeout(() => {
    setState(value); // Crashes if component unmounted
  }, delay);
}, []);
```

**Correct Pattern:**
```typescript
// CORRECT - Cleanup with ref storage
const timeoutRef = useRef<NodeJS.Timeout | null>(null);

useEffect(() => {
  timeoutRef.current = setTimeout(() => {
    setState(value);
  }, delay);

  return () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };
}, [delay]);
```

**Key Insight:** Any async operations in hooks must be cleaned up - use useRef for timers, AbortController for fetch calls.

---

### Scroll Reveal Hook Pattern

**Complete implementation for scroll-triggered animations:**

```typescript
function useScrollReveal(
  ref: RefObject<HTMLElement>,
  options?: { threshold?: number; rootMargin?: string; delay?: number }
) {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (options?.delay) {
            timeoutRef.current = setTimeout(() => {
              setIsVisible(true);
            }, options.delay);
          } else {
            setIsVisible(true);
          }
          observer.disconnect(); // Only trigger once
        }
      },
      {
        threshold: options?.threshold ?? 0.1,
        rootMargin: options?.rootMargin ?? '0px',
      }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [ref, options?.threshold, options?.rootMargin, options?.delay]);

  return isVisible;
}
```

**Key Insight:** Intersection Observer is hardware-accelerated and doesn't block main thread (unlike scroll event listeners).

---

### Animated Counter with Easing

**Pattern for smooth number animations:**

```typescript
// Easing functions
const easeOutQuad = (t: number) => 1 - (1 - t) * (1 - t);
const easeInOutCubic = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

function useCountUp(
  start: number,
  end: number,
  duration: number,
  easing: (t: number) => number,
  trigger: boolean
) {
  const [value, setValue] = useState(start);
  const frameRef = useRef<number>();

  useEffect(() => {
    if (!trigger) return;

    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easing(progress);

      setValue(start + (end - start) * easedProgress);

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(animate);
      }
    };

    frameRef.current = requestAnimationFrame(animate);

    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [start, end, duration, easing, trigger]);

  return value;
}
```

**Usage:**
```typescript
const count = useCountUp(0, 2024, 2000, easeOutQuad, isVisible);
return <span>{Math.floor(count)}</span>;
```

---

## Testing Patterns

### Jest Browser API Mocks

**Common mocks needed for hooks using browser APIs:**

```typescript
// jest.setup.ts

// matchMedia mock (for media queries, Intersection Observer)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// IntersectionObserver mock
class MockIntersectionObserver {
  observe = jest.fn();
  disconnect = jest.fn();
  unobserve = jest.fn();
}
Object.defineProperty(window, 'IntersectionObserver', {
  writable: true,
  value: MockIntersectionObserver,
});

// ResizeObserver mock
class MockResizeObserver {
  observe = jest.fn();
  disconnect = jest.fn();
  unobserve = jest.fn();
}
Object.defineProperty(window, 'ResizeObserver', {
  writable: true,
  value: MockResizeObserver,
});
```

**Key Insight:** When adding hooks that depend on browser APIs, immediately add Jest mocks to prevent test failures.

---

## Tailwind Patterns

### Dynamic Class Names Anti-Pattern

**Problem:** Tailwind JIT compiler scans static strings at build time.

**Anti-Pattern:**
```typescript
// WRONG - JIT can't detect dynamic classes
<div className={`duration-${duration}`} /> // Never works!
```

**Solutions:**

```typescript
// Option 1: Fixed predefined classes (recommended)
const durationMap = { fast: 'duration-200', slow: 'duration-500' };
<div className={durationMap[speed]} />

// Option 2: Inline styles for dynamic values
<div style={{ transitionDuration: `${duration}ms` }} />

// Option 3: CSS variables with Tailwind base class
<div
  className="transition-all"
  style={{ '--duration': `${duration}ms` } as React.CSSProperties}
/>
```

**Key Insight:** Never use template literals to generate Tailwind class names. Use classMap, inline styles, or CSS variables.

---

### Client-Side Image Ping for Health Checks

**Problem:** Server-side health check APIs create SSRF (Server-Side Request Forgery) vulnerabilities and require careful domain allowlisting.

**Vulnerable Pattern (Avoid):**
```typescript
// app/api/health-check/route.ts - DO NOT USE
// This creates SSRF risk if not properly validated
export async function GET(req: Request) {
  const url = req.nextUrl.searchParams.get('url');
  const response = await fetch(url); // SSRF vulnerability!
  return Response.json({ status: response.ok ? 'live' : 'offline' });
}
```

**Secure Client-Side Pattern:**
```typescript
// hooks/useImagePing.ts - Safe alternative
export function useImagePing(
  url: string,
  enabled: boolean = true,
): ImagePingState {
  const [state, setState] = useState<ImagePingState>({
    status: 'unknown',
    isLoading: enabled,
  });

  useEffect(() => {
    if (!enabled || !url) return;

    const img = new Image();
    const timeoutId = setTimeout(() => {
      img.src = ''; // Cancel loading
      setState({ status: 'unknown', isLoading: false });
    }, 5000);

    img.onload = () => {
      clearTimeout(timeoutId);
      setState({ status: 'live', isLoading: false });
    };

    img.onerror = () => {
      clearTimeout(timeoutId);
      setState({ status: 'unknown', isLoading: false });
    };

    try {
      const urlObj = new URL(url);
      img.src = `${urlObj.origin}/favicon.ico`;
    } catch {
      setState({ status: 'unknown', isLoading: false });
    }

    return () => {
      clearTimeout(timeoutId);
      img.src = '';
    };
  }, [url, enabled]);

  return state;
}
```

**How It Works:**
1. Uses native browser `Image()` element to load favicon
2. Browser handles CORS checking automatically (fails gracefully)
3. No server-side code = zero SSRF risk
4. Timeout ensures hung requests don't block forever
5. Returns 'live' if favicon loads, 'unknown' if blocked/timeout

**Key Insights:**
- **Security:** Zero SSRF risk because no server-side fetching
- **CORS:** Browser blocks cross-origin image loads automatically (safe fallback)
- **Graceful Degradation:** Returns 'unknown' for blocked resources instead of error
- **Client-Only:** Eliminates need for domain allowlists or API validation
- **Performance:** Simple Image() load is faster than HTTP API call

**When to Use:**
- Website health/status indicators
- Checking if external sites are reachable
- Simple connectivity validation (not detailed health data)

**Limitations:**
- Can't detect detailed HTTP status codes (only success/failure)
- CORS prevents cross-origin favicon loading (by design)
- Limited to checking if favicon loads (not actual site responsiveness)
- No way to distinguish between "offline" and "blocked by CORS"

---

## Third-Party Libraries (Additional)

### react-icons as MUI Icons Alternative

**Lighter alternative to @mui/icons-material:**

```bash
npm install react-icons
```

```typescript
// Icon sets available: fa (FontAwesome), hi (Heroicons),
// md (Material), si (Simple Icons), tb (Tabler), fi (Feather)
import { FaGithub, FaLinkedin } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import { SiReact, SiTypescript } from 'react-icons/si';

<FaGithub size={24} />
<SiReact color="#61DAFB" />
```

**Benefits over MUI Icons:**
- Much smaller bundle size (tree-shaking per icon)
- Multiple icon sets in one package
- Consistent API across all sets
- No additional dependencies

---

## State & Side Effect Patterns

### useEffect State Reset Race Conditions with Cached Events

**Problem:** When resetting component state in useEffect, cached/synchronous events (like `onLoad` from image cache) can fire before or during the state reset, causing timing mismatches.

**Anti-Pattern (Race Condition):**
```typescript
useEffect(() => {
  // Reset state
  setCurrImg(0);
  setDisplayImages(newImages);
  // But onLoad events from cached images might fire BEFORE or DURING this update
}, [newImages]);
```

**Why it fails:**
1. Image cache is synchronous - onLoad can fire immediately after `src` is set
2. useEffect runs AFTER render completes - cached images load before effect runs
3. State reset in effect can race with onLoad handler execution
4. Result: Component shows loading spinner even though images are cached

**Correct Pattern - Key-Based Remounting:**
```typescript
// Force component to remount by changing key
<ImageCarousel key={selectedLocation.id} img={images} />

// Inside component: useState initialization runs BEFORE any events
function ImageCarousel({ img }: Props) {
  const [currImg, setCurrImg] = useState(0);  // Runs on mount, before onLoad
  const [displayImages, setDisplayImages] = useState(img);

  // No useEffect state reset needed - fresh state on remount
}
```

**Key Insight:** React's `key` prop forces unmount/remount, which synchronously initializes fresh state via `useState` before any DOM events fire. This is safer than trying to race with async event handlers.

**When to use:**
- Resetting component state when data/parent changes
- Components with cached event handlers (images, videos)
- Preventing race conditions between state updates and side effects

**Alternative (if can't change parent):**
```typescript
// Store pending state separately
const pendingStateRef = useRef({ img: 0 });

useEffect(() => {
  // Mark state as pending
  pendingStateRef.current.img = 0;
}, [img]);

// In onLoad handler
function handleImageLoad() {
  if (pendingStateRef.current.img === 0) {
    setCurrImg(0);
  }
}
```

---

### useTransition - When NOT to Use It

**Problem:** Using `startTransition` for cheap state updates causes timing issues with synchronous events.

**Anti-Pattern:**
```typescript
useEffect(() => {
  startTransition(() => {
    setCurrImg(0);           // Deferred update
    setDisplayImages(img);   // Deferred update
  });
}, [img]);

// Problem: onLoad events fire BEFORE these deferred updates complete
// Result: Cached images don't reset their loaded state
```

**Correct Usage:**
```typescript
useEffect(() => {
  // Only defer the expensive operation
  setCurrImg(0); // Immediate, cheap update

  startTransition(() => {
    setLoadedImages(new Set());  // Expensive state tracking
    setDisplayImages(img);        // Complex state update
  });
}, [img]);
```

**Key Insight:** `useTransition` is for deferring EXPENSIVE operations (large renders, complex state). Don't use it for cheap resets - the timing delay can cause issues with synchronous events like `onLoad`.

**When to use useTransition:**
- Large array re-renders
- Complex component tree updates
- Expensive computations

**When NOT to use:**
- Simple state resets
- Index/counter resets
- Operations that sync with events

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

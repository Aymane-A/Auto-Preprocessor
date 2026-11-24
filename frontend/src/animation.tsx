import { AnimatePresence, MotionConfig, animate, motion, useInView, useIsPresent } from "framer-motion";
import { forwardRef, useEffect, useRef, useSyncExternalStore } from "react";
import type { ReactNode } from "react";

const reducedMotionQuery = typeof window === "undefined" ? null : window.matchMedia("(prefers-reduced-motion: reduce)");
function subscribeMotionPreference(update: () => void) {
  reducedMotionQuery?.addEventListener("change", update);
  return () => reducedMotionQuery?.removeEventListener("change", update);
}

/** React to preference changes during a session as well as on initial load. */
export function useReducedMotionPreference() {
  return useSyncExternalStore(subscribeMotionPreference, () => reducedMotionQuery?.matches ?? true, () => true);
}

export const motionTiming = {
  ease: [.16, 1, .3, 1] as [number, number, number, number],
  entrance: .32,
  exit: .12,
  interaction: .14,
  stagger: .035,
};

export const revealVariants = {
  initial: { opacity: .55, y: 6 },
  visible: { opacity: 1, y: 0, transition: { duration: motionTiming.entrance, ease: motionTiming.ease } },
  exit: { opacity: 0, transition: { duration: motionTiming.exit } },
};
const reducedVariants = {
  initial: { opacity: 1, y: 0 },
  visible: { opacity: 1, y: 0, transition: { duration: 0 } },
  exit: { opacity: 0, transition: { duration: 0 } },
};

const entranceSelector = [
  ".stage-heading", ".import-introduction", ".panel", ".metric", ".pipeline-stage", ".processing-overview",
  ".completion-overview", ".comparison", ".comparison-statistics > div", ".chart", ".boxplot",
  "tbody > tr:nth-child(-n+12)", ".downloads > a", ".workflow-nav > button", ".scene-node-controls > button",
  ".panel li:nth-child(-n+12)", "dialog[open]", '[role="dialog"]', '[role="listbox"]',
  ".advanced-grid > label:nth-child(-n+12)", ".step-controls > details:nth-child(-n+12)",
].join(",");
const interactiveSelector = 'button, a.button, a.primary, select, summary';

/** Apply viewport entrances and interaction feedback to existing and future panels. */
export function AnimationSystem({ children }: { children: ReactNode }) {
  const scope = useRef<HTMLDivElement>(null);
  const reducedMotion = useReducedMotionPreference();
  useEffect(() => {
    const root = scope.current;
    if (!root || reducedMotion) return;
    const play = animate;
    const seen = new WeakSet<Element>();
    const active = new Map<Element, ReturnType<typeof play>>();
    const interactions = new Set<ReturnType<typeof play>>();
    const pending = new Set<Element>();
    function reveal(element: Element) {
      if (!element.isConnected || element.closest('[inert], [aria-hidden="true"]')) return;
      active.get(element)?.stop();
      const index = Array.from(element.parentElement?.children ?? []).indexOf(element);
      const controls = play(element, { opacity: [.55, 1], y: [element.tagName === "TR" ? 3 : 6, 0] }, {
        duration: motionTiming.entrance, delay: Math.min(Math.max(index, 0), 7) * motionTiming.stagger,
        ease: motionTiming.ease,
      });
      active.set(element, controls);
      void controls.then(() => { if (active.get(element) === controls) active.delete(element); });
    }
    const viewport = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        viewport.unobserve(entry.target);
        pending.delete(entry.target);
        reveal(entry.target);
      });
    }, { threshold: .04 });
    function register(node: Element) {
      const elements = node.matches(entranceSelector) ? [node, ...node.querySelectorAll(entranceSelector)] : [...node.querySelectorAll(entranceSelector)];
      elements.forEach(element => {
        if (seen.has(element) || element.closest('[inert], .gpu-canvas')) return;
        seen.add(element);
        pending.add(element);
        viewport.observe(element);
      });
    }
    register(root);
    const observer = new MutationObserver(records => {
      records.forEach(record => {
        if (record.type === "attributes") {
          const element = record.target as Element;
          if (record.attributeName === "open") {
            if (!element.hasAttribute("open")) return;
            if (element.tagName === "DETAILS") {
              Array.from(element.children).filter(child => child.tagName !== "SUMMARY").slice(0, 8).forEach(reveal);
              return;
            }
          }
          viewport.unobserve(element);
          pending.add(element);
          viewport.observe(element);
          return;
        }
        record.addedNodes.forEach(node => { if (node instanceof Element) register(node); });
      });
      active.forEach((controls, element) => {
        if (!element.isConnected) { controls.stop(); active.delete(element); }
      });
      pending.forEach(element => {
        if (!element.isConnected) { viewport.unobserve(element); pending.delete(element); }
      });
    });
    observer.observe(root, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-motion-key", "open"] });
    function interact(event: Event) {
      if (!(event.target instanceof Element)) return;
      const element = event.target.closest(interactiveSelector);
      if (!element || element.closest('[inert], .gpu-canvas')) return;
      const related = (event as PointerEvent).relatedTarget;
      if (related instanceof Node && element.contains(related)) return;
      const pressed = event.type === "pointerdown";
      const resting = element.matches(":disabled") || event.type === "pointerout" || event.type === "focusout" || event.type === "pointercancel";
      const controls = play(element, { scale: pressed ? .98 : resting ? 1 : 1.012 }, { duration: motionTiming.interaction, ease: motionTiming.ease });
      interactions.add(controls);
      void controls.then(() => interactions.delete(controls));
    }
    const events = ["pointerover", "pointerout", "pointerdown", "pointerup", "pointercancel", "focusin", "focusout"];
    events.forEach(name => root.addEventListener(name, interact));
    return () => {
      observer.disconnect();
      viewport.disconnect();
      pending.clear();
      events.forEach(name => root.removeEventListener(name, interact));
      active.forEach(controls => controls.stop());
      interactions.forEach(controls => controls.stop());
      root.querySelectorAll<HTMLElement>(entranceSelector + "," + interactiveSelector).forEach(element => {
        element.style.removeProperty("transform");
        element.style.removeProperty("opacity");
      });
    };
  }, [reducedMotion]);
  return <MotionConfig reducedMotion={reducedMotion ? "always" : "never"} transition={{ duration: reducedMotion ? 0 : motionTiming.entrance, ease: motionTiming.ease }}>
    <div ref={scope} className="animation-root">{children}</div>
  </MotionConfig>;
}

/** Remove exiting steps from interaction and accessibility immediately. */
export const WorkflowStage = forwardRef<HTMLDivElement, { children: ReactNode }>(function WorkflowStage({ children }, ref) {
  const present = useIsPresent();
  const reducedMotion = useReducedMotionPreference();
  return <motion.div ref={ref} className="stage-body" inert={!present} aria-hidden={!present || undefined}
    initial={reducedMotion ? false : "initial"} animate="visible" exit="exit"
    variants={reducedMotion ? reducedVariants : revealVariants}>
    {children}
  </motion.div>;
});

/** Keep accessible values exact while a finite visual counter settles. */
export function AnimatedNumber({ value, digits = 4 }: { value: number; digits?: number }) {
  const element = useRef<HTMLSpanElement>(null);
  const current = useRef(0);
  const visible = useInView(element, { once: true });
  const reducedMotion = useReducedMotionPreference();
  const formatted = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: digits }).format(value);
  useEffect(() => {
    const node = element.current;
    if (!node || !visible) return;
    const format = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: digits });
    const write = (next: number) => {
      current.current = next;
      if (node.firstChild) node.firstChild.nodeValue = format.format(next);
    };
    if (reducedMotion || !Number.isFinite(value)) { write(value); return; }
    const controls = animate(current.current, value, { duration: .65, ease: motionTiming.ease, onUpdate: write, onComplete: () => write(value) });
    return () => controls.stop();
  }, [value, digits, visible, reducedMotion]);
  return <span className="animated-number"><span className="sr-only">{formatted}</span><span className="number-reserve" aria-hidden="true">{formatted}</span>
    <span ref={element} aria-hidden="true" data-value={value}>{reducedMotion ? formatted : new Intl.NumberFormat("fr-FR", { maximumFractionDigits: digits }).format(current.current)}</span></span>;
}

export function Notice({ message, error = false, children }: { message: string; error?: boolean; children?: ReactNode }) {
  const present = useIsPresent();
  const reducedMotion = useReducedMotionPreference();
  return <motion.div className={error ? "notice error" : "notice"} role={present ? error ? "alert" : "status" : undefined}
    aria-hidden={!present || undefined} inert={!present} initial={reducedMotion ? false : "initial"}
    animate="visible" exit="exit" variants={reducedMotion ? reducedVariants : revealVariants}>
    {!error && <span className="spinner" />}<div><strong>{message}</strong>{children}</div>
  </motion.div>;
}

export function ChartTooltip({ active, label, children }: { active?: boolean; label: ReactNode; children: ReactNode }) {
  const reducedMotion = useReducedMotionPreference();
  return <AnimatePresence>{active && <motion.div className="chart-tooltip" key={String(label)}
    initial={reducedMotion ? false : { opacity: .5, y: 3 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
    transition={{ duration: reducedMotion ? 0 : motionTiming.interaction }}><strong>{label}</strong>{children}</motion.div>}</AnimatePresence>;
}

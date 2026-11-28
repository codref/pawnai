<script setup lang="ts">
// The red string. Drawn in world coordinates inside the board's transformed
// world (the <svg> is 1×1 with overflow visible), under the cards. Each
// string has a wide invisible hit path for hover, select and the scissors.
import { computed } from "vue";
import type { BrainEdge, BrainSuggestion } from "@/api";
import { midpoint, polygonPath, stringPath, zigzagPath, type Pt } from "@/lib/brainmap/string";

const props = defineProps<{
  uid: string;
  edges: BrainEdge[];
  suggestions: BrainSuggestion[];
  pins: Map<string, Pt>;
  wobble: Record<string, number>;
  selectedEdge: string | null;
  activeSuggestion: string | null;
  dim: Set<string> | null;
  proposed: Set<string>;
  scissors: boolean;
  live: { from: Pt; to: Pt; ok: boolean | null } | null;
  lasso: Pt[] | null;
  /** Bumped by the board every animation frame so paths follow the cards. */
  frame: number;
}>();

const emit = defineEmits<{
  edge: [edge: BrainEdge, at: Pt, ev: MouseEvent];
  suggestion: [id: string, verdict: "confirm" | "snip"];
  hoverSuggestion: [s: BrainSuggestion | null, at: Pt | null];
}>();

interface Drawn {
  edge: BrainEdge;
  d: string;
  mid: Pt;
  cls: string[];
}

const drawn = computed<Drawn[]>(() => {
  void props.frame;
  const out: Drawn[] = [];
  for (const e of props.edges) {
    const a = props.pins.get(e.from);
    const b = props.pins.get(e.to);
    if (!a || !b) continue;
    const wobble = props.wobble[e.id] ?? 0;
    const d = e.role === "contradicts" ? zigzagPath(a, b, { wobble }) : stringPath(a, b, { wobble });
    const dimmed = props.dim && !(props.dim.has(e.from) && props.dim.has(e.to));
    out.push({
      edge: e,
      d,
      mid: midpoint(a, b, { wobble }),
      cls: [
        `by-${e.by}`,
        `role-${e.role}`,
        e.id.startsWith("pending:") ? "pending" : "",
        props.proposed.has(e.from) || props.proposed.has(e.to) ? "proposed" : "",
        dimmed ? "dim" : "",
        props.selectedEdge === e.id ? "selected" : "",
        e.cuttable ? "cuttable" : "",
        e.new ? "new" : "",
      ].filter(Boolean),
    });
  }
  return out;
});

const ghosts = computed(() => {
  void props.frame;
  const out: { s: BrainSuggestion; d: string; mid: Pt; active: boolean; dim: boolean }[] = [];
  for (const s of props.suggestions) {
    const a = props.pins.get(s.from);
    const b = props.pins.get(s.to);
    if (!a || !b) continue;
    out.push({
      s,
      d: stringPath(a, b, { sag: 0.22 }),
      mid: midpoint(a, b, { sag: 0.22 }),
      active: props.activeSuggestion === s.id,
      dim: !!props.dim && !(props.dim.has(s.from) && props.dim.has(s.to)),
    });
  }
  return out;
});

const livePath = computed(() => (props.live ? stringPath(props.live.from, props.live.to, { sag: 0.24 }) : ""));
const lassoPath = computed(() => (props.lasso && props.lasso.length > 1 ? polygonPath(props.lasso) : ""));

function onEdge(d: Drawn, ev: MouseEvent): void {
  emit("edge", d.edge, d.mid, ev);
}
</script>

<template>
  <svg class="bm-strings" :class="{ scissors }" width="1" height="1" aria-hidden="true">
    <defs>
      <filter :id="`${uid}-glow`" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="b" />
        <feMerge>
          <feMergeNode in="b" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>

    <g v-for="d in drawn" :key="d.edge.id" class="str" :class="d.cls">
      <path class="shadow" :d="d.d" />
      <path class="yarn" :d="d.d" :filter="d.edge.by === 'user' || d.cls.includes('selected') ? `url(#${uid}-glow)` : undefined" />
      <g v-if="d.edge.role === 'evidence_against'" class="knot-x" :transform="`translate(${d.mid.x},${d.mid.y})`">
        <line x1="-6" y1="-6" x2="6" y2="6" />
        <line x1="6" y1="-6" x2="-6" y2="6" />
      </g>
      <circle v-else-if="d.edge.by === 'user'" class="knot" :cx="d.mid.x" :cy="d.mid.y" r="3.5" />
      <path class="hit" :d="d.d" @click.stop="onEdge(d, $event)" />
    </g>

    <g
      v-for="g in ghosts"
      :key="g.s.id"
      class="ghost"
      :class="{ active: g.active, dim: g.dim }"
      @mouseenter="emit('hoverSuggestion', g.s, g.mid)"
      @mouseleave="emit('hoverSuggestion', null, null)"
    >
      <path class="yarn" :d="g.d" />
      <path class="hit" :d="g.d" />
      <g :transform="`translate(${g.mid.x},${g.mid.y})`" class="ghost-btns">
        <g class="btn ok" transform="translate(-14,0)" @click.stop="emit('suggestion', g.s.id, 'confirm')">
          <circle r="10" />
          <text text-anchor="middle" dominant-baseline="central">✓</text>
        </g>
        <g class="btn no" transform="translate(14,0)" @click.stop="emit('suggestion', g.s.id, 'snip')">
          <circle r="10" />
          <text text-anchor="middle" dominant-baseline="central">✂</text>
        </g>
      </g>
    </g>

    <path v-if="live" class="live" :class="{ ok: live.ok === true, bad: live.ok === false }" :d="livePath" :filter="`url(#${uid}-glow)`" />
    <path v-if="lassoPath" class="lasso" :d="lassoPath" />
  </svg>
</template>

<style scoped>
.bm-strings {
  position: absolute;
  left: 0;
  top: 0;
  overflow: visible;
  pointer-events: none;
}
.str path,
.ghost path,
.live,
.lasso {
  fill: none;
  stroke-linecap: round;
}
.shadow {
  stroke: rgba(0, 0, 0, 0.55);
  stroke-width: 4;
  transform: translate(3px, 5px);
}
.yarn {
  stroke: #ff2a4d;
  stroke-width: 2;
  transition: stroke-width 0.15s, opacity 0.2s;
}
.by-user .yarn {
  stroke: #ff1f45;
  stroke-width: 3.2;
}
.by-pawn .yarn {
  stroke: #e0324f;
  stroke-width: 1.8;
  opacity: 0.85;
}
/* Tied since the user last looked (the night shift, a meeting): static gold. */
.new .yarn {
  stroke: #ffd23f;
  opacity: 1;
}
.by-goal .yarn {
  stroke: #5ce1ff;
  stroke-width: 2.2;
}
.by-derived .yarn {
  stroke: #c9d6df;
  stroke-width: 1;
  opacity: 0.32;
}
.by-derived .shadow {
  opacity: 0.3;
}
.role-evidence_against .yarn {
  stroke: #ffb020;
  stroke-dasharray: 9 5;
}
.role-contradicts .yarn {
  stroke: #ff4df0;
  stroke-width: 2.2;
}
.role-parent .yarn {
  stroke-width: 2.8;
}
.proposed .yarn {
  stroke-dasharray: 3 6;
  opacity: 0.7;
}
.pending .yarn {
  stroke-dasharray: 2 4;
  /* A save in flight: a few marches, not forever. */
  animation: bm-march 0.6s linear 5;
}
.dim {
  opacity: 0.08;
}
.selected .yarn {
  stroke: #fff200;
  stroke-width: 3.6;
}
.knot {
  fill: #ff1f45;
  stroke: #2a0008;
  stroke-width: 1;
}
.knot-x line {
  stroke: #ffb020;
  stroke-width: 2.5;
  stroke-linecap: round;
}
.hit {
  stroke: transparent;
  stroke-width: 16;
  pointer-events: stroke;
  cursor: pointer;
}
.scissors .cuttable .hit {
  cursor:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24'><text y='18' font-size='18'>✂</text></svg>")
      8 8,
    crosshair;
}
.scissors .cuttable:hover .yarn {
  stroke: #fff200;
  stroke-dasharray: 4 3;
}
.str:not(.by-derived):hover .yarn {
  stroke-width: 3.6;
}

/* Ghost strings march only while you look at one: an endless
   stroke-dashoffset animation repaints the whole SVG layer every frame (and
   every card backdrop above it), which kept the CPU busy with the board idle. */
.ghost .yarn {
  stroke: #5ce1ff;
  stroke-width: 1.6;
  stroke-dasharray: 6 7;
  opacity: 0.55;
}
.ghost .hit {
  cursor: help;
}
.ghost.active .yarn,
.ghost:hover .yarn {
  opacity: 1;
  stroke-width: 2.4;
  filter: drop-shadow(0 0 4px #5ce1ff);
  animation: bm-march 1.4s linear infinite;
}
.ghost-btns {
  opacity: 0;
  transition: opacity 0.15s;
}
.ghost:hover .ghost-btns,
.ghost.active .ghost-btns {
  opacity: 1;
}
.btn {
  pointer-events: all;
  cursor: pointer;
}
.btn circle {
  fill: #071017;
  stroke: #5ce1ff;
  stroke-width: 1.5;
}
.btn text {
  fill: #5ce1ff;
  font-size: 11px;
  font-weight: 700;
}
.btn.no circle {
  stroke: #ff4d9a;
}
.btn.no text {
  fill: #ff4d9a;
}
.btn:hover circle {
  fill: #12303d;
}
.live {
  stroke: #ff2a4d;
  stroke-width: 3;
  stroke-dasharray: 10 4;
  animation: bm-march 0.5s linear infinite;
}
.live.ok {
  stroke: #3dffb0;
  stroke-dasharray: none;
}
.live.bad {
  stroke: #6b7782;
}
.lasso {
  stroke: #5ce1ff;
  stroke-width: 1.6;
  stroke-dasharray: 5 5;
  fill: rgba(92, 225, 255, 0.06);
  animation: bm-march 0.8s linear infinite;
}
@keyframes bm-march {
  to {
    stroke-dashoffset: -26;
  }
}
@media (prefers-reduced-motion: reduce) {
  .ghost.active .yarn,
  .ghost:hover .yarn,
  .live,
  .lasso,
  .pending .yarn {
    animation: none;
  }
}
</style>

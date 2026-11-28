<script setup lang="ts">
// One card pinned to a brainmap board. The push-pin is the string handle
// (drag it onto another card to tie); the card body is the drag handle.
// Kind decides the costume: index card, dossier, IOU ticket, mugshot, tape…
import { computed } from "vue";
import type { BrainNode } from "@/api";
import { kindMeta } from "@/lib/brainmap/boards";
import { hashSeed } from "@/lib/brainmap/physics";
import { kindLabel } from "@/lib/inbox";
import { formatClock } from "@/lib/audio";

const props = withDefaults(
  defineProps<{
    node: BrainNode;
    selected?: boolean;
    dimmed?: boolean;
    glow?: boolean;
    burning?: boolean;
    flashing?: boolean;
    busy?: boolean;
    tieTarget?: "ok" | "bad" | null;
  }>(),
  { selected: false, dimmed: false, glow: false, burning: false, flashing: false, busy: false, tieTarget: null },
);

const emit = defineEmits<{ verb: [verb: string] }>();

const meta = computed(() => kindMeta(props.node.kind));
const tilt = computed(() => {
  if (props.selected) return 0;
  const h = hashSeed(props.node.id) % 1000;
  const span = props.node.proposed ? 7 : 2.6;
  return ((h / 1000) * 2 - 1) * span;
});

const stamp = computed(() => {
  if (props.node.kind === "item") return kindLabel(props.node.meta.item_kind || "").toUpperCase();
  return (meta.value.stamp || meta.value.label).toUpperCase();
});

const initials = computed(() =>
  props.node.title
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? "")
    .join(""),
);

const momentum = computed(() => {
  const m = props.node.meta.momentum;
  if (m === null || m === undefined) return null;
  return { value: Math.max(-1, Math.min(1, m)), label: m > 0.05 ? "RISING" : m < -0.05 ? "FADING" : "STEADY" };
});

const countdown = computed(() => {
  const when = props.node.meta.by_when || props.node.meta.due;
  if (!when) return "";
  const days = Math.round((Date.parse(String(when)) - Date.now()) / 86_400_000);
  if (Number.isNaN(days)) return String(when);
  if (days < 0) return `OVERDUE ${-days}D`;
  if (days === 0) return "TODAY";
  return `T-${days}D`;
});

const direction = computed(() => {
  const d = (props.node.meta.direction || "").toLowerCase();
  return d === "mine" ? "MINE" : d === "theirs" ? "THEIRS" : "";
});

const ITEM_VERBS: { verb: string; icon: string; label: string }[] = [
  { verb: "todo", icon: "check_box", label: "Add TODO" },
  { verb: "file", icon: "folder", label: "File" },
  { verb: "task", icon: "task_alt", label: "Task" },
];

const verbs = computed(() => {
  const a = new Set(props.node.actions);
  const out: { verb: string; icon: string; label: string; hot?: boolean }[] = [];
  if (a.has("accept")) out.push({ verb: "accept", icon: "verified", label: "Accept" });
  for (const v of ITEM_VERBS) if (a.has(v.verb)) out.push(v);
  if (a.has("spill")) out.push({ verb: "spill", icon: "unfold_more_double", label: "Pull the thread" });
  if (a.has("remove")) out.push({ verb: "remove", icon: "remove_circle_outline", label: "Take off the board" });
  if (a.has("dismiss")) out.push({ verb: "dismiss", icon: "local_fire_department", label: "Burn", hot: true });
  else if (a.has("ignore")) out.push({ verb: "ignore", icon: "local_fire_department", label: "Burn", hot: true });
  return out;
});

const time = computed(() => {
  const s = props.node.meta.start_sec;
  return typeof s === "number" ? formatClock(s) : "";
});
</script>

<template>
  <div
    class="bm-card"
    :class="[
      `kind-${node.kind}`,
      {
        selected,
        dimmed,
        glow,
        burning,
        flashing,
        proposed: node.proposed,
        done: ['done', 'dismissed', 'resolved', 'filed', 'task'].includes(node.status),
        'tie-ok': tieTarget === 'ok',
        'tie-bad': tieTarget === 'bad',
      },
    ]"
    :style="{ '--pin': meta.pin, '--w': `${meta.w}px`, '--h': `${meta.h}px`, '--tilt': `${tilt}deg` }"
    :data-node="node.id"
  >
    <div class="pin" :title="'Drag the pin onto another card to tie a string'" />
    <div class="tape tape-l" />
    <div class="tape tape-r" />

    <!-- Person: mugshot -->
    <template v-if="node.kind === 'person'">
      <div class="mug">
        <div class="mug-lines" />
        <div class="mug-face">{{ initials || "?" }}</div>
      </div>
      <div class="mug-name">{{ node.title }}</div>
    </template>

    <!-- Session: cassette -->
    <template v-else-if="node.kind === 'session'">
      <div class="tape-label">{{ node.title }}</div>
      <div class="reels"><span class="reel" /><span class="window" /><span class="reel" /></div>
    </template>

    <template v-else>
      <div class="hdr">
        <span class="stamp">{{ stamp }}</span>
        <span v-if="node.meta.new" class="chip new">NEW</span>
        <span v-if="node.meta.proposal_id" class="chip ghost">{{ node.origin === "night" ? "NIGHT SHIFT" : "PAWN PROPOSES" }}</span>
        <span v-if="direction" class="chip" :class="direction.toLowerCase()">{{ direction }}</span>
        <span class="sp" />
        <span v-if="countdown" class="chip" :class="{ hot: countdown.startsWith('OVERDUE') }">{{ countdown }}</span>
        <span v-else-if="time" class="chip">{{ time }}</span>
        <span v-if="node.meta.evidence_count" class="chip">{{ node.meta.evidence_count }} EV</span>
      </div>
      <div class="title" :class="{ big: node.kind === 'question' }">{{ node.title }}</div>
      <div v-if="node.kind === 'thread' && momentum" class="momentum">
        <span class="bar"><span :style="{ width: `${Math.abs(momentum.value) * 100}%` }" :class="{ neg: momentum.value < 0 }" /></span>
        <span class="label">{{ momentum.label }}</span>
      </div>
      <div v-if="node.meta.assumptions?.length" class="leds">
        <span
          v-for="(a, i) in node.meta.assumptions"
          :key="i"
          class="led"
          :class="a.health"
          :title="`${a.health}: ${a.text}`"
        />
        <span class="label">ASSUMPTIONS</span>
      </div>
      <div v-if="node.quote && (node.kind === 'item' || node.kind === 'scrap')" class="quote">“{{ node.quote }}”</div>
      <div v-else-if="node.body && (node.kind === 'hypothesis' || node.kind === 'note')" class="quote">{{ node.body }}</div>
      <div v-if="node.meta.owner || node.meta.speaker" class="who">
        ◉ {{ node.meta.owner || node.meta.speaker }}
      </div>
      <div v-if="verbs.length" class="verbs" @pointerdown.stop>
        <q-btn
          v-for="v in verbs"
          :key="v.verb"
          flat
          dense
          round
          size="xs"
          :icon="v.icon"
          :class="{ hot: v.hot }"
          :loading="busy && v.verb !== 'spill'"
          :aria-label="v.label"
          @click.stop="emit('verb', v.verb)"
        >
          <q-tooltip>{{ v.label }}</q-tooltip>
        </q-btn>
      </div>
    </template>

    <div v-if="node.proposed" class="redacted">PROPOSED</div>
    <div v-if="burning" class="embers" />
  </div>
</template>

<style scoped>
.bm-card {
  position: relative;
  width: var(--w);
  min-height: var(--h);
  padding: 18px 12px 10px;
  box-sizing: border-box;
  color: #d8e6ee;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.035), transparent 40%),
    linear-gradient(135deg, #141a22 0%, #10151c 100%);
  border: 1px solid color-mix(in srgb, var(--pin) 45%, transparent);
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
  transform: rotate(var(--tilt));
  transition:
    transform 0.25s cubic-bezier(0.2, 0.9, 0.3, 1.3),
    opacity 0.25s,
    filter 0.25s,
    box-shadow 0.25s;
  box-shadow:
    0 10px 22px rgba(0, 0, 0, 0.55),
    inset 3px 0 0 var(--pin);
  font-size: 12.5px;
  line-height: 1.35;
  user-select: none;
  cursor: grab;
}
.bm-card:active {
  cursor: grabbing;
}
.bm-card.selected {
  transform: rotate(0deg) scale(1.04);
  box-shadow:
    0 0 0 1px var(--pin),
    0 0 24px color-mix(in srgb, var(--pin) 55%, transparent),
    0 14px 30px rgba(0, 0, 0, 0.6),
    inset 3px 0 0 var(--pin);
}
.bm-card.glow {
  box-shadow:
    0 0 0 2px #fff200,
    0 0 28px rgba(255, 242, 0, 0.5),
    inset 3px 0 0 var(--pin);
}
.bm-card.dimmed {
  opacity: 0.18;
  filter: grayscale(0.8) blur(0.4px);
}
.bm-card.done {
  filter: saturate(0.35);
  opacity: 0.65;
}
.bm-card.tie-ok {
  box-shadow:
    0 0 0 2px #ff2a4d,
    0 0 30px rgba(255, 42, 77, 0.65);
}
.bm-card.tie-bad {
  animation: bm-shake 0.35s;
}
.bm-card.flashing {
  animation: bm-flash 0.9s ease-out;
}
.bm-card.burning {
  animation: bm-burn 0.75s ease-in forwards;
  pointer-events: none;
}

/* push-pin: glowing bead, the string handle */
.pin {
  position: absolute;
  top: 3px;
  left: 50%;
  width: 14px;
  height: 14px;
  margin-left: -7px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff 0 12%, var(--pin) 40%, color-mix(in srgb, var(--pin) 40%, #000) 100%);
  box-shadow:
    0 0 10px var(--pin),
    0 2px 3px rgba(0, 0, 0, 0.8);
  cursor: crosshair;
  z-index: 2;
  transition: transform 0.15s;
}
.pin:hover {
  transform: scale(1.35);
}
.tape {
  position: absolute;
  top: -4px;
  width: 38px;
  height: 13px;
  /* No backdrop-filter: two per card, re-blurred whenever the strings
     behind them repaint, was the board's biggest idle CPU cost. */
  background: rgba(92, 225, 255, 0.16);
  border: 1px dashed rgba(92, 225, 255, 0.25);
}
.tape-l {
  left: -8px;
  transform: rotate(-28deg);
}
.tape-r {
  right: 4px;
  transform: rotate(24deg);
  display: none;
}
.kind-thread .tape-r,
.kind-hypothesis .tape-r {
  display: block;
}

.hdr {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 5px;
  font-family: var(--pawn-mono);
  font-size: 9.5px;
  letter-spacing: 0.12em;
}
.sp {
  flex: 1;
}
.stamp {
  color: var(--pin);
  border: 1px solid var(--pin);
  padding: 0 4px;
  text-shadow: 0 0 6px color-mix(in srgb, var(--pin) 60%, transparent);
}
.chip {
  padding: 0 4px;
  background: rgba(255, 255, 255, 0.07);
  color: #9fb4c2;
}
.chip.hot,
.chip.theirs {
  background: rgba(255, 42, 77, 0.18);
  color: #ff8095;
}
.chip.mine {
  background: rgba(61, 255, 176, 0.14);
  color: #7dffc8;
}
.chip.new {
  background: rgba(255, 210, 63, 0.2);
  color: #ffe27a;
}
.chip.ghost {
  background: rgba(92, 225, 255, 0.14);
  color: #9feeff;
}
.title {
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
  word-break: break-word;
}
.title.big {
  font-size: 14px;
  font-weight: 600;
}
.kind-thread .title {
  font-family: var(--pawn-mono);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #e9fbff;
}
.kind-thread {
  background:
    repeating-linear-gradient(0deg, transparent 0 21px, rgba(92, 225, 255, 0.05) 21px 22px),
    linear-gradient(135deg, #142028, #0e161d);
}
.kind-question::after {
  content: "?";
  position: absolute;
  right: 10px;
  bottom: -6px;
  font: 900 64px/1 var(--pawn-mono);
  color: rgba(255, 210, 63, 0.12);
  pointer-events: none;
}
.kind-commitment {
  border-style: dashed;
  background:
    radial-gradient(circle at 0 50%, transparent 7px, transparent 7px),
    linear-gradient(135deg, #18142a, #110f1d);
}
.kind-hypothesis {
  background: linear-gradient(135deg, #1e1218, #140d12);
  clip-path: polygon(0 4%, 8% 0, 22% 3%, 40% 0, 58% 4%, 76% 0, 92% 3%, 100% 0, 100% 96%, 88% 100%, 70% 97%, 50% 100%, 30% 96%, 12% 100%, 0 97%);
}
.kind-note {
  background: linear-gradient(135deg, #22201a, #17150f);
  border-color: rgba(244, 241, 222, 0.28);
}
.kind-scrap {
  font-size: 11.5px;
  background: linear-gradient(135deg, #171717, #101010);
  border-color: rgba(255, 59, 92, 0.3);
}

.quote {
  margin-top: 5px;
  font-style: italic;
  color: #9fb4c2;
  border-left: 2px solid color-mix(in srgb, var(--pin) 60%, transparent);
  padding-left: 6px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.who {
  margin-top: 5px;
  font-family: var(--pawn-mono);
  font-size: 10px;
  color: #8fa3b8;
  letter-spacing: 0.06em;
}
.momentum,
.leds {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  font-family: var(--pawn-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  color: #8fa3b8;
}
.bar {
  flex: 1;
  height: 4px;
  background: rgba(255, 255, 255, 0.08);
}
.bar > span {
  display: block;
  height: 100%;
  background: #5ce1ff;
  box-shadow: 0 0 8px #5ce1ff;
}
.bar > span.neg {
  background: #ff9e3d;
  box-shadow: 0 0 8px #ff9e3d;
}
.led {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #3dffb0;
  box-shadow: 0 0 6px #3dffb0;
}
.led.weak {
  background: #ffd23f;
  box-shadow: 0 0 6px #ffd23f;
}
.led.unchecked {
  background: #4a5662;
  box-shadow: none;
}
.led.broken {
  background: #ff2a4d;
  box-shadow: 0 0 8px #ff2a4d;
  /* Blink to catch the eye, then stay lit. */
  animation: bm-blink 1s steps(2) 6;
}
.verbs {
  display: flex;
  justify-content: flex-end;
  gap: 2px;
  margin: 6px -6px -4px 0;
  opacity: 0.55;
  transition: opacity 0.15s;
}
.bm-card:hover .verbs,
.bm-card.selected .verbs {
  opacity: 1;
}
.verbs :deep(.q-btn) {
  color: #9fd8ea;
}
.verbs :deep(.q-btn.hot) {
  color: #ff6b3d;
}

/* mugshot */
.kind-person {
  padding: 18px 10px 8px;
  text-align: center;
}
.mug {
  position: relative;
  height: 96px;
  background: #0b0f14;
  overflow: hidden;
}
.mug-lines {
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(180deg, transparent 0 11px, rgba(255, 255, 255, 0.09) 11px 12px);
}
.mug-face {
  position: absolute;
  left: 50%;
  bottom: -6px;
  width: 64px;
  height: 78px;
  margin-left: -32px;
  border-radius: 32px 32px 6px 6px;
  background: linear-gradient(180deg, #2a3541, #1b232c);
  display: flex;
  align-items: center;
  justify-content: center;
  font: 700 20px var(--pawn-mono);
  color: #e8e8e8;
}
.mug-name {
  margin-top: 6px;
  font: 700 11px var(--pawn-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* cassette */
.kind-session {
  background: linear-gradient(180deg, #1d242c, #151a20);
  padding: 16px 10px 8px;
}
.tape-label {
  background: #e8dcc0;
  color: #1a1a1a;
  font: 600 10.5px var(--pawn-mono);
  padding: 3px 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transform: rotate(-0.6deg);
}
.reels {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 8px;
}
.reel {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 3px dotted #8fa3b8;
}
/* Reels turn while you hover the tape, not on every session card at once. */
.bm-card:hover .reel {
  animation: bm-spin 6s linear infinite;
}
.window {
  width: 56px;
  height: 14px;
  background: rgba(92, 225, 255, 0.1);
  border: 1px solid rgba(92, 225, 255, 0.3);
}

.redacted {
  position: absolute;
  right: 8px;
  bottom: 30%;
  padding: 1px 8px;
  font: 900 12px var(--pawn-mono);
  letter-spacing: 0.3em;
  color: #ff2a4d;
  border: 2px solid #ff2a4d;
  transform: rotate(-14deg);
  opacity: 0.85;
  text-shadow: 0 0 6px rgba(255, 42, 77, 0.6);
  pointer-events: none;
  mix-blend-mode: screen;
}
.embers {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 30% 100%, rgba(255, 120, 40, 0.9), transparent 45%),
    radial-gradient(circle at 70% 100%, rgba(255, 42, 77, 0.8), transparent 40%);
  mix-blend-mode: screen;
  pointer-events: none;
}

@keyframes bm-shake {
  0%,
  100% {
    transform: rotate(var(--tilt)) translateX(0);
  }
  25% {
    transform: rotate(var(--tilt)) translateX(-7px);
  }
  75% {
    transform: rotate(var(--tilt)) translateX(7px);
  }
}
@keyframes bm-flash {
  0% {
    box-shadow:
      0 0 0 3px #3dffb0,
      0 0 60px #3dffb0;
    filter: brightness(1.8);
  }
  100% {
    filter: none;
  }
}
@keyframes bm-burn {
  0% {
    filter: none;
    opacity: 1;
  }
  40% {
    filter: brightness(1.6) sepia(1) hue-rotate(-20deg) saturate(4);
  }
  100% {
    filter: brightness(0.2) sepia(1) saturate(6);
    opacity: 0;
    transform: rotate(calc(var(--tilt) + 8deg)) translateY(30px) scale(0.85);
    clip-path: polygon(0 100%, 100% 100%, 100% 100%, 0 100%);
  }
}
@keyframes bm-blink {
  50% {
    opacity: 0.2;
  }
}
@keyframes bm-spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .bm-card,
  .bm-card:hover .reel,
  .led.broken {
    animation: none !important;
    transition: none;
  }
  .bm-card.burning {
    opacity: 0;
  }
}
</style>

<script setup lang="ts">
// The bottom strip: everything Pawn wants a human verdict on, one at a time.
// Y ties / accepts, N snips / burns, → skips. The board flies the camera to
// the entry and lights its ghost string or card.
import { computed } from "vue";
import type { BrainNode } from "@/api";
import { kindMeta } from "@/lib/brainmap/boards";
import type { QueueEntry } from "@/lib/brainmap/graph";

const props = defineProps<{
  entry: QueueEntry | null;
  index: number;
  total: number;
  nodes: Map<string, BrainNode>;
  open: boolean;
}>();
const emit = defineEmits<{ yes: []; no: []; skip: []; toggle: [] }>();

function short(text: string | undefined, n = 70): string {
  const one = (text || "").replace(/\s+/g, " ").trim();
  return one.length > n ? `${one.slice(0, n - 1)}…` : one;
}

const prompt = computed(() => {
  const e = props.entry;
  if (!e) return null;
  if (e.type === "suggestion") {
    const from = props.nodes.get(e.suggestion.from);
    const to = props.nodes.get(e.suggestion.to);
    return {
      tag: "GHOST STRING",
      text: `Does “${short(from?.title)}” belong to ${kindMeta(to?.kind || "thread").label.toLowerCase()} “${short(to?.title, 50)}”?`,
      why: e.suggestion.reason,
      yes: "Tie it",
      no: "Snip",
    };
  }
  return {
    tag: `PROPOSED ${kindMeta(e.node.kind).label.toUpperCase()}`,
    text: e.node.title,
    why: e.node.body && e.node.body !== e.node.title ? short(e.node.body, 140) : "Pawn noticed this on its own. Keep it on the board?",
    yes: "Accept",
    no: "Burn",
  };
});
</script>

<template>
  <div class="bm-queue" :class="{ open }" @pointerdown.stop>
    <button type="button" class="toggle" @click="emit('toggle')">
      <span class="dot" :class="{ live: total > 0 }" />
      PAWN SUGGESTS {{ total }}
      <q-icon :name="open ? 'expand_more' : 'expand_less'" size="14px" />
    </button>
    <div v-if="open && prompt" class="entry">
      <div class="meta">
        <span class="tag">{{ prompt.tag }}</span>
        <span class="count">{{ index + 1 }} / {{ total }}</span>
      </div>
      <div class="text">{{ prompt.text }}</div>
      <div v-if="prompt.why" class="why">{{ prompt.why }}</div>
      <div class="btns">
        <q-btn dense no-caps unelevated class="yes" :label="prompt.yes" @click="emit('yes')"><kbd>Y</kbd></q-btn>
        <q-btn dense no-caps unelevated class="no" :label="prompt.no" @click="emit('no')"><kbd>N</kbd></q-btn>
        <q-btn dense no-caps flat class="skip" label="Skip" @click="emit('skip')"><kbd>→</kbd></q-btn>
      </div>
    </div>
    <div v-else-if="open" class="entry empty">Nothing waiting. Every string on this board has a verdict.</div>
  </div>
</template>

<style scoped>
.bm-queue {
  position: absolute;
  left: 50%;
  bottom: 12px;
  transform: translateX(-50%);
  width: min(620px, calc(100% - 24px));
  z-index: 7;
  font-size: 13px;
  color: #d8e6ee;
}
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  font: 600 10.5px var(--pawn-mono);
  letter-spacing: 0.16em;
  color: #5ce1ff;
  background: rgba(7, 11, 16, 0.92);
  border: 1px solid rgba(92, 225, 255, 0.4);
  cursor: pointer;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #3b4652;
}
.dot.live {
  background: #ff4d9a;
  box-shadow: 0 0 8px #ff4d9a;
  animation: q-pulse 1.4s infinite;
}
.entry {
  margin-top: -1px;
  padding: 10px 14px;
  background: rgba(7, 11, 16, 0.95);
  border: 1px solid rgba(92, 225, 255, 0.4);
  border-left: 3px solid #ff4d9a;
  box-shadow: 0 0 40px rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(6px);
}
.entry.empty {
  color: #8fa3b8;
  font-size: 12px;
}
.meta {
  display: flex;
  justify-content: space-between;
  font: 600 10px var(--pawn-mono);
  letter-spacing: 0.16em;
}
.tag {
  color: #ff4d9a;
}
.count {
  color: #8fa3b8;
}
.text {
  margin: 4px 0 2px;
  font-size: 14px;
  font-weight: 500;
}
.why {
  color: #8fa3b8;
  font-size: 12px;
  font-style: italic;
}
.btns {
  display: flex;
  gap: 6px;
  margin-top: 8px;
}
.btns :deep(.q-btn) {
  padding: 0 10px;
}
.btns kbd {
  margin-left: 6px;
  font: 10px var(--pawn-mono);
  padding: 0 4px;
  border: 1px solid currentColor;
  opacity: 0.7;
}
.yes {
  background: #0c6a80;
  color: #fff;
}
.no {
  background: #7a1530;
  color: #fff;
}
.skip {
  color: #8fa3b8;
}
@keyframes q-pulse {
  50% {
    opacity: 0.35;
  }
}
@media (prefers-reduced-motion: reduce) {
  .dot.live {
    animation: none;
  }
}
</style>

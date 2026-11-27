<script setup lang="ts">
// The one global right-click menu. App.vue mounts it once; a document-level
// `contextmenu` listener opens it whenever the click lands on (or the
// selection sits inside) a `v-target` element. Shift+right-click, or a click
// that a component's own menu already handled, keeps the native behaviour.
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { actionsFor, type Action } from "@/lib/actions";
import { resolveTarget, type LocalAction, type ResolvedTarget } from "@/lib/targets";

const open = ref(false);
const x = ref(0);
const y = ref(0);
const target = ref<ResolvedTarget | null>(null);
const anchor = ref<HTMLElement | null>(null);

const groups = computed(() => {
  const t = target.value;
  if (!t) return [];
  const out: { key: string; items: (Action | LocalAction)[] }[] = [];
  const shared = actionsFor(t);
  for (const g of ["ask", "open"] as const) {
    const items = shared.filter((a) => a.group === g);
    if (items.length) out.push({ key: g, items });
  }
  if (t.local?.length && !t.quote) out.push({ key: "local", items: t.local });
  for (const g of ["save", "copy"] as const) {
    const items = shared.filter((a) => a.group === g);
    if (items.length) out.push({ key: g, items });
  }
  return out;
});

function labelOf(item: Action | LocalAction): string {
  return typeof item.label === "function" ? item.label(target.value!) : item.label;
}

function isDanger(item: Action | LocalAction): boolean {
  return "danger" in item && Boolean(item.danger);
}

async function run(item: Action | LocalAction): Promise<void> {
  const t = target.value;
  open.value = false;
  if (!t) return;
  if ("when" in item) await item.run(t, { x: x.value, y: y.value });
  else await item.run();
}

async function onContextMenu(ev: MouseEvent): Promise<void> {
  if (ev.shiftKey || ev.defaultPrevented) return;
  const t = resolveTarget(ev);
  if (!t) return;
  ev.preventDefault();
  open.value = false;
  await nextTick();
  target.value = t;
  x.value = ev.clientX;
  y.value = ev.clientY;
  open.value = true;
}

function onKey(ev: KeyboardEvent): void {
  if (ev.key === "Escape") open.value = false;
}

onMounted(() => {
  document.addEventListener("contextmenu", onContextMenu);
  document.addEventListener("keydown", onKey);
});
onBeforeUnmount(() => {
  document.removeEventListener("contextmenu", onContextMenu);
  document.removeEventListener("keydown", onKey);
});
</script>

<template>
  <div ref="anchor" class="ctx-anchor" :style="{ left: `${x}px`, top: `${y}px` }">
    <q-menu
      v-model="open"
      no-parent-event
      :target="anchor ?? undefined"
      anchor="bottom left"
      self="top left"
      transition-show="jump-down"
      transition-duration="120"
    >
      <q-list v-if="target" dense style="min-width: 220px; max-width: 320px">
        <q-item-label header class="ctx-header ellipsis">{{ target.quote ? `“${target.quote}”` : target.label }}</q-item-label>
        <template v-for="(g, gi) in groups" :key="g.key">
          <q-separator v-if="gi > 0" />
          <q-item
            v-for="(item, i) in g.items"
            :key="`${g.key}-${i}`"
            clickable
            :class="{ 'text-negative': isDanger(item) }"
            @click="run(item)"
          >
            <q-item-section avatar class="ctx-icon">
              <q-icon :name="item.icon || 'chevron_right'" size="18px" />
            </q-item-section>
            <q-item-section>{{ labelOf(item) }}</q-item-section>
          </q-item>
        </template>
      </q-list>
    </q-menu>
  </div>
</template>

<style scoped>
.ctx-anchor {
  position: fixed;
  width: 0;
  height: 0;
  pointer-events: none;
  z-index: 9000;
}
.ctx-header {
  padding: 6px 16px 4px;
  font-size: 0.72rem;
  max-width: 320px;
}
.ctx-icon {
  min-width: 30px;
  padding-right: 6px;
}
</style>

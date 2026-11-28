<script setup lang="ts">
// Right-side dossier for the selected card: full text, quote, who and when,
// the subject's state and evidence timeline, and every verb. Jumping to a
// moment opens the session and seeks its audio to that second.
import { computed, ref, watch } from "vue";
import { api, type BrainNode, type EmergenceSubjectDetail } from "@/api";
import { kindMeta } from "@/lib/brainmap/boards";
import { formatClock } from "@/lib/audio";
import { notifyError } from "@/lib/errors";
import { openFile, openSession } from "@/router";
import MarkdownView from "@/components/MarkdownView.vue";

const props = defineProps<{ node: BrainNode; busy: boolean; links: number }>();
const emit = defineEmits<{ close: []; verb: [verb: string]; focus: [] }>();

const detail = ref<EmergenceSubjectDetail | null>(null);
const loading = ref(false);
const meta = computed(() => kindMeta(props.node.kind));

watch(
  () => props.node.meta.subject_id,
  async (id) => {
    detail.value = null;
    if (!id) return;
    loading.value = true;
    try {
      detail.value = await api().emergenceSubject(id);
    } catch (e) {
      notifyError(e, "Could not open the dossier");
    } finally {
      loading.value = false;
    }
  },
  { immediate: true },
);

const sourceRef = computed(() => props.node.meta.source_ref || "");
const startSec = computed(() => (typeof props.node.meta.start_sec === "number" ? props.node.meta.start_sec : null));

function jump(ref: string, sec: number | null): void {
  if (!ref) return;
  if (sec !== null) openSession(ref, { where: "split" }, undefined, sec);
  else openSession(ref, { where: "split" }, "items");
}

const VERBS: Record<string, { label: string; icon: string; hot?: boolean }> = {
  accept: { label: "Accept", icon: "verified" },
  approve: { label: "Approve", icon: "check" },
  reject: { label: "Reject", icon: "close", hot: true },
  todo: { label: "Add TODO", icon: "check_box" },
  file: { label: "File", icon: "folder" },
  task: { label: "Task", icon: "task_alt" },
  later: { label: "Later", icon: "snooze" },
  spill: { label: "Pull the thread", icon: "unfold_more_double" },
  council: { label: "Convene council", icon: "groups" },
  done: { label: "Close", icon: "done_all" },
  resolved: { label: "Resolved", icon: "gavel" },
  ignore: { label: "Burn", icon: "local_fire_department", hot: true },
  dismiss: { label: "Burn", icon: "local_fire_department", hot: true },
  delete: { label: "Delete", icon: "delete", hot: true },
  remove: { label: "Take off the board", icon: "remove_circle_outline" },
};

const verbs = computed(() => props.node.actions.filter((a) => VERBS[a]).map((a) => ({ verb: a, ...VERBS[a] })));
</script>

<template>
  <aside class="bm-dossier pawn-scroll" :style="{ '--pin': meta.pin }" @pointerdown.stop @wheel.stop>
    <header>
      <span class="stamp">{{ meta.label }}</span>
      <span class="status">{{ node.proposed ? "PROPOSED" : node.status.toUpperCase() }}</span>
      <q-space />
      <q-btn flat dense round size="sm" icon="center_focus_strong" aria-label="Focus" @click="emit('focus')">
        <q-tooltip>Focus on this card</q-tooltip>
      </q-btn>
      <q-btn flat dense round size="sm" icon="close" aria-label="Close" @click="emit('close')" />
    </header>
    <h3>{{ node.title }}</h3>
    <p v-if="node.body && node.body !== node.title && !node.meta.subject_id" class="body">{{ node.body }}</p>
    <blockquote v-if="node.quote">“{{ node.quote }}”</blockquote>

    <dl class="facts">
      <template v-if="node.meta.owner"><dt>OWNER</dt><dd>{{ node.meta.owner }}</dd></template>
      <template v-if="node.meta.speaker"><dt>SAID BY</dt><dd>{{ node.meta.speaker }}</dd></template>
      <template v-if="node.meta.due"><dt>DUE</dt><dd>{{ node.meta.due }}</dd></template>
      <template v-if="node.meta.by_when"><dt>BY</dt><dd>{{ node.meta.by_when }}</dd></template>
      <template v-if="node.meta.direction"><dt>WHO OWES</dt><dd>{{ node.meta.direction }}</dd></template>
      <template v-if="node.origin"><dt>ORIGIN</dt><dd>{{ node.origin }}</dd></template>
      <template v-if="sourceRef"><dt>TAPE</dt><dd>{{ sourceRef }}<template v-if="startSec !== null"> @ {{ formatClock(startSec) }}</template></dd></template>
      <dt>STRINGS</dt><dd>{{ links }}</dd>
    </dl>

    <div v-if="node.meta.assumptions?.length" class="section">
      <div class="label">ASSUMPTIONS</div>
      <div v-for="(a, i) in node.meta.assumptions" :key="i" class="assumption" :class="a.health">
        <span class="led" /> {{ a.text }}
      </div>
    </div>

    <div class="actions">
      <q-btn
        v-if="sourceRef"
        dense
        no-caps
        unelevated
        class="neon"
        icon="play_arrow"
        :label="startSec !== null ? `Jump to ${formatClock(startSec)}` : 'Open session'"
        @click="jump(sourceRef, startSec)"
      />
      <q-btn v-if="node.meta.note_key" dense no-caps flat icon="description" label="Note" @click="openFile(node.meta.note_key!, { pin: true })" />
      <q-btn
        v-for="v in verbs"
        :key="v.verb"
        dense
        no-caps
        flat
        :icon="v.icon"
        :label="v.label"
        :class="{ hot: v.hot }"
        :loading="busy"
        @click="emit('verb', v.verb)"
      />
    </div>
    <div v-if="!node.proposed && node.kind !== 'note'" class="actions boards">
      <q-btn dense no-caps flat icon="push_pin" label="Pin to a board…" @click="emit('verb', 'pin_to')" />
      <q-btn v-if="node.meta.subject_id" dense no-caps flat icon="dashboard" label="Open as board" @click="emit('verb', 'open_board')" />
      <q-btn
        v-if="node.meta.subject_id && ['question', 'thread', 'decision', 'hypothesis'].includes(node.kind)"
        dense
        no-caps
        flat
        icon="bedtime"
        label="Investigate overnight"
        @click="emit('verb', 'investigate')"
      >
        <q-tooltip>A question board: the night shift looks for evidence and hunches</q-tooltip>
      </q-btn>
    </div>

    <template v-if="node.meta.subject_id">
      <q-linear-progress v-if="loading" indeterminate color="accent" class="q-my-sm" />
      <template v-if="detail">
        <div v-if="detail.subject.state_md" class="section">
          <div class="label">STATE OF THE CASE</div>
          <MarkdownView :source="detail.subject.state_md" class="state" />
        </div>
        <div class="section">
          <div class="label">EVIDENCE · {{ detail.observations.length }}</div>
          <div v-if="!detail.observations.length" class="muted">No evidence linked yet.</div>
          <div v-for="o in detail.observations" :key="o.id" class="obs" :class="`role-${o.role}`">
            <div class="obs-hdr">
              <span class="role">{{ o.role.replace("_", " ") }}</span>
              <span v-if="o.speaker_name">{{ o.speaker_name }}</span>
              <q-space />
              <button
                v-if="o.source_ref"
                type="button"
                class="clock"
                :title="`Open ${o.source_ref}`"
                @click="jump(o.source_ref, o.start_sec)"
              >
                {{ o.start_sec !== null ? formatClock(o.start_sec) : "▶" }}
              </button>
            </div>
            <div>{{ o.text }}</div>
            <div v-if="o.quote" class="obs-quote">“{{ o.quote }}”</div>
          </div>
        </div>
        <div v-if="detail.events.length" class="section">
          <div class="label">HISTORY</div>
          <div v-for="(ev, i) in detail.events.slice(-8).reverse()" :key="i" class="event">
            <span class="clock">{{ ev.at.slice(0, 10) }}</span> {{ ev.kind.replace(/_/g, " ") }}
          </div>
        </div>
      </template>
    </template>
  </aside>
</template>

<style scoped>
.bm-dossier {
  position: absolute;
  top: 52px;
  right: 10px;
  bottom: 70px;
  width: min(360px, calc(100% - 20px));
  overflow-y: auto;
  z-index: 6;
  padding: 12px 14px;
  background: rgba(7, 11, 16, 0.94);
  border: 1px solid color-mix(in srgb, var(--pin) 50%, transparent);
  box-shadow:
    0 0 30px rgba(0, 0, 0, 0.7),
    inset 0 0 0 1px rgba(255, 255, 255, 0.03);
  clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 0 100%);
  color: #d8e6ee;
  font-size: 13px;
  backdrop-filter: blur(6px);
}
header {
  display: flex;
  align-items: center;
  gap: 6px;
  font: 600 10px var(--pawn-mono);
  letter-spacing: 0.14em;
}
.stamp {
  color: var(--pin);
  border: 1px solid var(--pin);
  padding: 0 5px;
  text-transform: uppercase;
}
.status {
  color: #8fa3b8;
}
h3 {
  margin: 10px 0 6px;
  font-size: 16px;
  line-height: 1.3;
  font-weight: 600;
}
.body {
  color: #b9cad6;
  white-space: pre-wrap;
}
blockquote {
  margin: 8px 0;
  padding-left: 10px;
  border-left: 2px solid var(--pin);
  font-style: italic;
  color: #9fb4c2;
}
.facts {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2px 10px;
  margin: 10px 0;
  font-size: 12px;
}
dt {
  font: 600 10px var(--pawn-mono);
  letter-spacing: 0.12em;
  color: #5ce1ff;
  padding-top: 2px;
}
dd {
  margin: 0;
  word-break: break-word;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin: 10px 0;
}
.actions :deep(.q-btn) {
  color: #9fd8ea;
}
.actions :deep(.q-btn.hot) {
  color: #ff6b3d;
}
.actions :deep(.q-btn.neon) {
  background: #0c6a80;
  color: #fff;
}
.section {
  margin-top: 14px;
}
.label {
  font: 600 10px var(--pawn-mono);
  letter-spacing: 0.16em;
  color: #ff4d9a;
  margin-bottom: 6px;
}
.state {
  font-size: 12.5px;
}
.obs {
  padding: 6px 8px;
  margin-bottom: 6px;
  border-left: 2px solid #e0324f;
  background: rgba(255, 255, 255, 0.03);
  font-size: 12px;
}
.obs.role-evidence_against {
  border-left-color: #ffb020;
}
.obs.role-contradicts {
  border-left-color: #ff4df0;
}
.obs-hdr {
  display: flex;
  gap: 6px;
  font: 10px var(--pawn-mono);
  letter-spacing: 0.08em;
  color: #8fa3b8;
  margin-bottom: 2px;
  text-transform: uppercase;
}
.role {
  color: #ff8095;
}
.obs-quote {
  margin-top: 3px;
  font-style: italic;
  color: #8fa3b8;
}
.clock {
  font: 10px var(--pawn-mono);
  color: #5ce1ff;
  background: transparent;
  border: 1px solid rgba(92, 225, 255, 0.35);
  padding: 0 4px;
  cursor: pointer;
}
.event {
  font-size: 11.5px;
  color: #9fb4c2;
  margin-bottom: 2px;
}
.event .clock {
  cursor: default;
}
.assumption {
  display: flex;
  gap: 6px;
  align-items: baseline;
  font-size: 12px;
  margin-bottom: 3px;
}
.led {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #3dffb0;
  box-shadow: 0 0 6px #3dffb0;
}
.assumption.weak .led {
  background: #ffd23f;
  box-shadow: 0 0 6px #ffd23f;
}
.assumption.unchecked .led {
  background: #4a5662;
  box-shadow: none;
}
.assumption.broken .led {
  background: #ff2a4d;
  box-shadow: 0 0 6px #ff2a4d;
}
.muted {
  color: #8fa3b8;
  font-size: 12px;
}
</style>

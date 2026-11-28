<script setup lang="ts">
// Browse recorded sessions: search, needs-speakers, date range, speaker
// facets, and a multi-select for reidentify / mark trained.
import { computed, onMounted, ref, watch } from "vue";
import { Notify } from "quasar";
import { api, type SessionSummary } from "@/api";
import { notifyError } from "@/lib/errors";
import { formatTalk } from "@/lib/audio";
import { vTarget, type Target } from "@/lib/targets";
import { askAbout, type ActionContext } from "@/lib/actions";
import { openFile, openSession, openSpeakers } from "@/router";
import { useJobsStore } from "@/stores/jobs";
import { useSessionsStore, type SessionRange, type SessionSort } from "@/stores/sessions";
import { useTabsStore } from "@/stores/tabs";
import SpeakerChip from "@/components/SpeakerChip.vue";
import StatusBadge from "@/components/StatusBadge.vue";
import SessionActionDialogs from "@/components/SessionActionDialogs.vue";
import SegmentedControl from "@/components/SegmentedControl.vue";

const props = defineProps<{ tabId: string }>();

const sessions = useSessionsStore();
const tabs = useTabsStore();
const jobs = useJobsStore();
const root = ref<HTMLElement | null>(null);
const searchEl = ref<{ focus: () => void } | null>(null);
const cursor = ref(0);
const anchor = ref(0);
const busy = ref(false);
const job = ref<{ kind: "extract" | "reprocess"; row: SessionSummary } | null>(null);
let queryTimer = 0;

const ranges: { value: SessionRange; label: string }[] = [
  { value: "all", label: "All time" },
  { value: "today", label: "Today" },
  { value: "week", label: "This week" },
  { value: "month", label: "This month" },
];
const sorts: { value: SessionSort; label: string }[] = [
  { value: "recent", label: "Recent" },
  { value: "oldest", label: "Oldest" },
  { value: "duration", label: "Longest" },
];

const rangeLabel = computed(() => ranges.find((item) => item.value === sessions.range)?.label ?? "All time");
const visibleSpeakers = computed(() => {
  const facets = sessions.facets.speakers ?? [];
  const head = facets.slice(0, 8);
  if (sessions.speaker && !head.some((item) => item.name === sessions.speaker)) {
    const extra = facets.find((item) => item.name === sessions.speaker);
    if (extra) head.unshift(extra);
  }
  return head;
});
const moreSpeakers = computed(() => (sessions.facets.speakers ?? []).slice(8));

interface DayGroup {
  key: string;
  title: string;
  rows: { row: SessionSummary; index: number }[];
}

const groups = computed<DayGroup[]>(() => {
  const out: DayGroup[] = [];
  sessions.rows.forEach((row, index) => {
    const key = dayKey(row.created_at ?? row.updated_at);
    const last = out[out.length - 1];
    if (!last || last.key !== key) out.push({ key, title: dayTitle(key), rows: [] });
    out[out.length - 1].rows.push({ row, index });
  });
  return out;
});

function dayKey(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso.slice(0, 10);
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${m}-${d}`;
}

function dayTitle(key: string): string {
  if (!key) return "Undated";
  const today = dayKey(new Date().toISOString());
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  if (key === today) return "Today";
  if (key === dayKey(yesterday.toISOString())) return "Yesterday";
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year, (month || 1) - 1, day || 1).toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

function clock(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function typing(ev: KeyboardEvent): boolean {
  const el = ev.target instanceof HTMLElement ? ev.target : null;
  return !!el?.closest("input, textarea, select, [contenteditable]");
}

function scrollCursor(): void {
  root.value?.querySelector(`[data-row="${cursor.value}"]`)?.scrollIntoView({ block: "nearest" });
}

function train(id: string): void {
  sessions.focusRail = id;
  openSession(id);
}

function chat(row: SessionSummary, ev: Event): void {
  const point = ev instanceof MouseEvent ? ev : null;
  const ctx: ActionContext = { x: point?.clientX ?? 0, y: point?.clientY ?? 0 };
  void askAbout({ kind: "session", ref: row.session_id, label: row.title || row.session_id }, ctx);
}

function onRowClick(ev: MouseEvent, index: number, id: string): void {
  if ((ev.target as HTMLElement).closest("button, a, .q-checkbox")) return;
  if (ev.shiftKey) {
    const [from, to] = anchor.value < index ? [anchor.value, index] : [index, anchor.value];
    sessions.selectRange(sessions.rows.slice(from, to + 1).map((row) => row.session_id));
    return;
  }
  anchor.value = index;
  cursor.value = index;
  openSession(id);
}

function onKey(ev: KeyboardEvent): void {
  if (ev.metaKey || ev.ctrlKey || ev.altKey || typing(ev)) return;
  const row = sessions.rows[cursor.value];
  if (ev.key === "j" || ev.key === "ArrowDown") {
    ev.preventDefault();
    cursor.value = Math.min(sessions.rows.length - 1, cursor.value + 1);
    scrollCursor();
  } else if (ev.key === "k" || ev.key === "ArrowUp") {
    ev.preventDefault();
    cursor.value = Math.max(0, cursor.value - 1);
    scrollCursor();
  } else if (ev.key === "Enter" && row) {
    openSession(row.session_id);
  } else if (ev.key === "o" && row) {
    openSession(row.session_id, { where: "split" });
  } else if (ev.key === "t" && row) {
    train(row.session_id);
  } else if (ev.key === "/") {
    ev.preventDefault();
    searchEl.value?.focus();
  }
}

function onScroll(ev: Event): void {
  const el = ev.target as HTMLElement;
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 240) void sessions.more();
}

function rowTarget(row: SessionSummary): Target {
  return {
    kind: "session",
    ref: row.session_id,
    label: row.title || row.session_id,
    local: [
      {
        label: "Extract…",
        icon: "playlist_add_check",
        run: () => {
          job.value = { kind: "extract", row };
        },
      },
      {
        label: "Reprocess…",
        icon: "refresh",
        run: () => {
          job.value = { kind: "reprocess", row };
        },
      },
      { label: "Reidentify", icon: "record_voice_over", run: () => void reidentifyOne(row.session_id) },
      {
        label: "Open note",
        icon: "description",
        run: () => {
          if (row.transcript_path) openFile(row.transcript_path);
        },
      },
    ],
  };
}

async function reidentifyOne(id: string): Promise<void> {
  try {
    const result = await api().reidentifySession(id);
    const jobId = result.job_id;
    if (typeof jobId === "string") void jobs.watchJob(jobId);
    Notify.create({ message: "Reidentify queued.", position: "bottom-right" });
    await sessions.reload();
  } catch (e) {
    notifyError(e, "Reidentify failed");
  }
}

async function bulkReidentify(): Promise<void> {
  if (!sessions.selected.length || busy.value) return;
  busy.value = true;
  try {
    const result = await api().reidentifySessions({ session_ids: sessions.selected, limit: 50 });
    if (result.job_id) void jobs.watchJob(result.job_id);
    Notify.create({ message: `Reidentify queued for ${sessions.selected.length}.`, position: "bottom-right" });
    sessions.clearSelected();
  } catch (e) {
    notifyError(e, "Reidentify failed");
  } finally {
    busy.value = false;
  }
}

async function bulkTrained(): Promise<void> {
  if (!sessions.selected.length || busy.value) return;
  busy.value = true;
  try {
    await Promise.all(sessions.selected.map((id) => api().setSessionTrained(id, true)));
    Notify.create({ message: "Marked trained.", position: "bottom-right" });
    await sessions.reload();
  } catch (e) {
    notifyError(e, "Could not mark trained");
  } finally {
    busy.value = false;
  }
}

watch(
  () => sessions.q,
  () => {
    window.clearTimeout(queryTimer);
    queryTimer = window.setTimeout(() => void sessions.reload(), 250);
  },
);
watch(
  () => [sessions.scope, sessions.range, sessions.sort] as const,
  () => void sessions.reload(),
);
watch(
  () => sessions.needsSpeakers,
  (count) => tabs.setStatus(props.tabId, { title: count ? `Sessions (${count})` : "Sessions" }),
  { immediate: true },
);

async function openGallery(name: string): Promise<void> {
  try {
    const rows = await api().listSpeakers({ q: name, include_inactive: true });
    const needle = name.trim().toLowerCase();
    const hit = rows.find(
      (row) =>
        row.display_name.toLowerCase() === needle ||
        row.id.toLowerCase() === needle ||
        row.aliases.some((alias) => alias.toLowerCase() === needle),
    );
    if (!hit) {
      Notify.create({
        message: `${name} is not in the gallery`,
        position: "bottom-right",
        timeout: 1500,
      });
      openSpeakers("");
      return;
    }
    openSpeakers(hit.id);
  } catch (e) {
    notifyError(e, "Could not open the gallery");
  }
}

onMounted(() => {
  void sessions.reload();
});
</script>

<template>
  <div ref="root" class="column no-wrap full-height" tabindex="0" :data-tab="tabId" @keydown="onKey">
    <div class="row items-center q-pa-xs q-gutter-xs border-b">
      <q-input
        ref="searchEl"
        v-model="sessions.q"
        dense
        outlined
        clearable
        class="col"
        placeholder="Search sessions"
        aria-label="Search sessions"
        @clear="sessions.q = ''"
      >
        <template #prepend><q-icon name="search" /></template>
      </q-input>
      <SegmentedControl
        v-model="sessions.scope"
        aria-label="Session filter"
        :options="[
          { label: 'All', value: 'all' },
          { label: 'Needs speakers', value: 'needs', icon: 'record_voice_over', count: sessions.needsSpeakers },
        ]"
      />
      <q-btn-dropdown dense no-caps outline :label="rangeLabel">
        <q-list dense>
          <q-item
            v-for="item in ranges"
            :key="item.value"
            v-close-popup
            clickable
            :active="sessions.range === item.value"
            @click="sessions.range = item.value"
          >
            <q-item-section>{{ item.label }}</q-item-section>
          </q-item>
          <q-separator />
          <q-item
            v-for="item in sorts"
            :key="item.value"
            v-close-popup
            clickable
            :active="sessions.sort === item.value"
            @click="sessions.sort = item.value"
          >
            <q-item-section>{{ item.label }}</q-item-section>
          </q-item>
        </q-list>
      </q-btn-dropdown>
      <q-btn flat dense round icon="refresh" aria-label="Refresh" :loading="sessions.loading" @click="sessions.reload()" />
      <q-btn flat dense round icon="help_outline" aria-label="Keyboard shortcuts">
        <q-menu anchor="bottom right" self="top right">
          <q-list dense style="min-width: 180px">
            <q-item><q-item-section>j / k move</q-item-section></q-item>
            <q-item><q-item-section>Enter open</q-item-section></q-item>
            <q-item><q-item-section>o open to the side</q-item-section></q-item>
            <q-item><q-item-section>t train</q-item-section></q-item>
            <q-item><q-item-section>/ search</q-item-section></q-item>
          </q-list>
        </q-menu>
      </q-btn>
    </div>

    <div class="row items-center q-px-sm q-py-xs q-gutter-xs facets">
      <span class="sub-label">Speakers</span>
      <button
        v-for="facet in visibleSpeakers"
        :key="facet.name"
        type="button"
        class="pawn-pill"
        :class="{ active: sessions.speaker === facet.name }"
        :aria-pressed="sessions.speaker === facet.name"
        @click="sessions.setSpeaker(facet.name)"
      >
        {{ facet.name }} <span class="count">{{ facet.count }}</span>
        <q-menu context-menu>
          <q-list dense>
            <q-item v-close-popup clickable @click="openGallery(facet.name)">
              <q-item-section>Open in gallery</q-item-section>
            </q-item>
          </q-list>
        </q-menu>
      </button>
      <q-btn v-if="moreSpeakers.length" flat dense no-caps size="sm" label="more">
        <q-menu>
          <q-list dense>
            <q-item
              v-for="facet in moreSpeakers"
              :key="facet.name"
              v-close-popup
              clickable
              @click="sessions.setSpeaker(facet.name)"
            >
              <q-item-section>{{ facet.name }} {{ facet.count }}</q-item-section>
              <q-menu context-menu>
                <q-list dense>
                  <q-item v-close-popup clickable @click="openGallery(facet.name)">
                    <q-item-section>Open in gallery</q-item-section>
                  </q-item>
                </q-list>
              </q-menu>
            </q-item>
          </q-list>
        </q-menu>
      </q-btn>
    </div>

    <div v-if="sessions.selected.length" class="row items-center q-px-sm q-py-xs q-gutter-xs border-b">
      <span class="text-caption">{{ sessions.selected.length }} selected</span>
      <q-btn dense no-caps flat icon="record_voice_over" label="Reidentify" :loading="busy" @click="bulkReidentify" />
      <q-btn dense no-caps flat icon="check" label="Mark trained" :loading="busy" @click="bulkTrained" />
      <q-btn dense flat no-caps label="Clear" @click="sessions.clearSelected()" />
    </div>

    <q-banner v-if="sessions.error" dense class="pawn-banner danger">{{ sessions.error }}</q-banner>
    <div class="col pawn-scroll" @scroll="onScroll">
      <div v-if="!sessions.rows.length && !sessions.loading" class="q-pa-lg text-center text-muted">No sessions</div>
      <section v-for="group in groups" :key="group.key">
        <div class="day-head section-label">{{ group.title }}</div>
        <div
          v-for="{ row, index } in group.rows"
          :key="row.session_id"
          v-target="rowTarget(row)"
          :data-row="index"
          class="sess-row"
          :class="{ cursor: cursor === index, picked: sessions.selected.includes(row.session_id) }"
          @click="onRowClick($event, index, row.session_id)"
          @dblclick.stop="openSession(row.session_id, { pin: true })"
        >
          <q-checkbox
            dense
            class="tick"
            :model-value="sessions.selected.includes(row.session_id)"
            @update:model-value="(on: boolean) => sessions.toggleSelected(row.session_id, on)"
          />
          <div class="col" style="min-width: 0">
            <div class="row items-center no-wrap q-gutter-x-xs">
              <span class="title ellipsis">{{ row.title || row.session_id }}</span>
              <span class="text-caption text-muted clock">{{ clock(row.created_at) }} · {{ formatTalk(row.duration_seconds ?? 0) }}</span>
              <SpeakerChip v-for="name in (row.speakers ?? []).slice(0, 4)" :key="name" :name="name" :size="18" class="spk" />
              <span v-if="(row.speakers?.length ?? 0) > 4" class="text-caption text-muted">+{{ (row.speakers?.length ?? 0) - 4 }}</span>
              <StatusBadge v-if="row.items_open" kind="info" :label="`${row.items_open} items`" />
              <StatusBadge v-if="row.needs_speakers" kind="warn" label="needs training" />
              <StatusBadge v-else-if="row.speakers_trained" kind="ok" label="trained" />
              <q-icon v-if="row.event" name="event" size="16px" class="text-muted">
                <q-tooltip>{{ row.event.title }}</q-tooltip>
              </q-icon>
            </div>
            <div v-if="row.summary" class="summary ellipsis text-caption text-muted">{{ row.summary }}</div>
          </div>
          <div class="hover-actions row no-wrap">
            <q-btn flat dense no-caps size="sm" label="Open" @click.stop="openSession(row.session_id)" />
            <q-btn flat dense no-caps size="sm" label="Chat" @click.stop="chat(row, $event)" />
            <q-btn flat dense no-caps size="sm" label="Train" @click.stop="train(row.session_id)" />
          </div>
        </div>
      </section>
      <div v-if="sessions.loading" class="q-pa-sm text-center text-caption text-muted">Loading…</div>
    </div>

    <SessionActionDialogs
      v-if="job"
      :session-id="job.row.session_id"
      :transcript-path="job.row.transcript_path"
      :extract="job.kind === 'extract'"
      :reprocess="job.kind === 'reprocess'"
      @update:extract="(open) => !open && (job = null)"
      @update:reprocess="(open) => !open && (job = null)"
      @done="sessions.reload()"
    />
  </div>
</template>

<style scoped>
.border-b {
  border-bottom: 1px solid var(--pawn-border);
}
.facets {
  min-height: 32px;
}
.day-head {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 4px 12px;
  background: var(--pawn-bg);
  border-bottom: 1px solid var(--pawn-border);
  letter-spacing: 0.04em;
}
.sess-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-bottom: 1px solid var(--pawn-border);
  cursor: pointer;
}
.sess-row:hover,
.sess-row.cursor {
  background: var(--pawn-hover);
}
.sess-row.cursor {
  box-shadow: inset 2px 0 0 var(--q-primary);
}
.sess-row.picked {
  background: var(--pawn-active);
}
/* Avatars only in the list row: names live in the tooltip. */
.spk :deep(.q-ml-xs) {
  display: none;
}
.title {
  font-weight: 500;
  max-width: 16rem;
}
.clock {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.summary {
  max-width: 48rem;
}
.tick {
  opacity: 0;
}
.sess-row:hover .tick,
.sess-row.picked .tick,
.sess-row.cursor .tick {
  opacity: 1;
}
.hover-actions {
  opacity: 0;
}
.sess-row:hover .hover-actions,
.sess-row.cursor .hover-actions {
  opacity: 1;
}
</style>

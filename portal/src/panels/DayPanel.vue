<script setup lang="ts">
// Day tool tab: a time grid, an agenda rail, and an event editor drawer.
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Notify } from "quasar";
import { api, type PlannerDay, type PlannerEvent, type PlannerSuggestion, type SessionSummary } from "@/api";
import { notifyError } from "@/lib/errors";
import { confirmAction } from "@/lib/dialogs";
import { isOpen } from "@/lib/inbox";
import {
  addDays,
  columnLabel,
  dayHeading,
  minutesOf,
  minutesOn,
  rangeLabel,
  recordingSpan,
  stampAt,
  startOfWeek,
  todayISO,
  type DayColumn,
  type GridItem,
} from "@/lib/planner";
import { loadString, saveString } from "@/lib/storage";
import { vTarget, type Target } from "@/lib/targets";
import { dayRoute, openChat, openDay, openFile, openSession } from "@/router";
import { isSideChat, useChatsStore } from "@/stores/chats";
import { useFilesStore } from "@/stores/files";
import { useInboxStore } from "@/stores/inbox";
import { useJobsStore } from "@/stores/jobs";
import { useTabsStore } from "@/stores/tabs";
import ItemCard from "@/components/ItemCard.vue";
import MarkdownView from "@/components/MarkdownView.vue";
import TimeGrid from "@/components/TimeGrid.vue";
import EventDrawer from "@/components/EventDrawer.vue";
import SegmentedControl from "@/components/SegmentedControl.vue";

const props = defineProps<{ tabId: string }>();

const tabs = useTabsStore();
const files = useFilesStore();
const chats = useChatsStore();
const inbox = useInboxStore();
const jobs = useJobsStore();

const days = ref<PlannerDay[]>([]);
const loading = ref(false);
const agendaOpen = ref(loadString("day.agenda") !== "0");
const density = ref(loadString("day.density") === "comfortable" ? "comfortable" : "compact");
const nowMinutes = ref(0);
const quick = ref<{ date: string; start: number; end: number; x: number; y: number; title: string } | null>(null);
const root = ref<HTMLElement | null>(null);
let seq = 0;
let clockTimer = 0;
let stopEvents: (() => void) | null = null;
let stopDaily: (() => void) | null = null;

const date = computed(() => dayRoute.date || todayISO());
const week = computed(() => dayRoute.view === "week");
const pxPerHour = computed(() => (density.value === "comfortable" ? 72 : 48));
const focused = computed(() => days.value.find((day) => day.date === date.value) ?? days.value[0] ?? null);
const selected = computed(() => {
  const id = dayRoute.event;
  if (!id) return null;
  for (const day of days.value) {
    const ev = day.events.find((item) => item.id === id);
    if (ev) return ev;
  }
  return null;
});
const heading = computed(() => {
  if (!week.value) return dayHeading(date.value);
  const start = startOfWeek(date.value);
  return `${dayHeading(start)} – ${dayHeading(addDays(start, 6))}`;
});

watch(agendaOpen, (open) => saveString("day.agenda", open ? "1" : "0"));
watch(density, (value) => saveString("day.density", value));
watch([date, week, heading], () => {
  tabs.setStatus(props.tabId, { title: `${week.value ? "Week" : "Day"} · ${date.value}` });
}, { immediate: true });

function tick(): void {
  const now = new Date();
  nowMinutes.value = now.getHours() * 60 + now.getMinutes();
}

function eventById(id: string): PlannerEvent | null {
  for (const day of days.value) {
    const ev = day.events.find((item) => item.id === id);
    if (ev) return ev;
  }
  return null;
}

function sessionById(id: string): SessionSummary | null {
  for (const day of days.value) {
    const row = day.sessions.find((item) => item.session_id === id);
    if (row) return row;
  }
  return null;
}

function relocate(event: PlannerEvent): void {
  for (const day of days.value) day.events = day.events.filter((item) => item.id !== event.id);
  const key = (event.day || event.start || "").slice(0, 10);
  const day = days.value.find((item) => item.date === key);
  if (day) day.events.push(event);
}

function columnFor(day: PlannerDay): DayColumn {
  const linked = new Set(day.events.map((ev) => ev.session_id).filter(Boolean) as string[]);
  const items: GridItem[] = [];
  const allDay: { id: string; title: string }[] = [];
  for (const ev of day.events) {
    const start = minutesOf(ev.start);
    if (start == null) {
      allDay.push({ id: ev.id, title: ev.title });
      continue;
    }
    const endMin = minutesOf(ev.end);
    const end = endMin != null && endMin > start ? endMin : start + 30;
    items.push({
      id: ev.id,
      kind: "event",
      start,
      end,
      draggable: true,
      focusBefore: ev.focus ? ev.focus_before || 15 : 0,
    });
  }
  for (const session of day.sessions) {
    if (linked.has(session.session_id)) continue;
    const span = recordingSpan(session);
    if (!span) continue;
    const start = minutesOn(day.date, span.start);
    const end = minutesOn(day.date, span.end);
    if (end <= start) continue;
    items.push({
      id: session.session_id,
      kind: "session",
      start,
      end,
      draggable: false,
      focusBefore: 0,
    });
  }
  return { date: day.date, label: columnLabel(day.date), today: day.date === todayISO(), items, allDay };
}

const columns = computed(() => days.value.map(columnFor));

const unlinked = computed(() => {
  const day = focused.value;
  if (!day) return [];
  const linked = new Set(day.events.map((ev) => ev.session_id).filter(Boolean));
  return day.sessions.filter((session) => !linked.has(session.session_id));
});
const todos = computed(() => (focused.value?.other_items ?? []).filter((item) => isOpen(item)));

async function load(): Promise<void> {
  const token = ++seq;
  loading.value = true;
  try {
    if (week.value) {
      const start = startOfWeek(date.value);
      const range = await api().plannerRange(start, addDays(start, 6));
      if (token === seq) days.value = range.days;
    } else {
      const day = await api().plannerDay(date.value);
      if (token === seq) days.value = [day];
    }
  } catch (e) {
    if (token === seq) notifyError(e, "Could not load the day");
  } finally {
    if (token === seq) loading.value = false;
  }
}

function shift(step: number): void {
  const by = week.value ? step * 7 : step;
  openDay({ date: addDays(date.value, by) });
}

function goToday(): void {
  openDay({ date: todayISO() });
}

function setView(view: "day" | "week"): void {
  openDay({ view });
}

function onPick(day: string): void {
  openDay({ date: day, view: "day" });
}

function onSelect(id: string): void {
  if (eventById(id)) {
    openDay({ eventId: id });
    return;
  }
  if (sessionById(id)) openSession(id, { where: "split" });
}

function onOpen(id: string): void {
  const ev = eventById(id);
  if (ev?.path) openFile(ev.path, { pin: true });
  else if (sessionById(id)) openSession(id, { where: "split", pin: true });
}

function onCreate(payload: { date: string; start: number; end: number; x: number; y: number }): void {
  quick.value = { ...payload, title: "" };
}

async function saveQuick(): Promise<void> {
  const draft = quick.value;
  if (!draft) return;
  const title = draft.title.trim();
  if (!title) return;
  try {
    const ev = await api().upsertPlannerEvent({
      title,
      start: stampAt(draft.date, draft.start),
      end: stampAt(draft.date, draft.end),
      source: "manual",
    });
    quick.value = null;
    relocate(ev);
  } catch (e) {
    notifyError(e, "Could not add the event");
  }
}

async function moreQuick(): Promise<void> {
  const draft = quick.value;
  if (!draft) return;
  try {
    const ev = await api().upsertPlannerEvent({
      title: draft.title.trim() || "New event",
      start: stampAt(draft.date, draft.start),
      end: stampAt(draft.date, draft.end),
      source: "manual",
    });
    quick.value = null;
    relocate(ev);
    openDay({ date: draft.date, eventId: ev.id });
  } catch (e) {
    notifyError(e, "Could not add the event");
  }
}

function newEvent(): void {
  let start = nowMinutes.value;
  start = Math.round(start / 15) * 15 + 15;
  if (start < 7 * 60 || start > 20 * 60) start = 9 * 60;
  const rect = root.value?.getBoundingClientRect();
  quick.value = {
    date: date.value,
    start,
    end: start + 30,
    x: (rect?.left ?? 80) + 72,
    y: (rect?.top ?? 48) + 48,
    title: "",
  };
}

async function onCommit(payload: { id: string; date: string; start: number; end: number }): Promise<void> {
  const ev = eventById(payload.id);
  if (!ev) return;
  const prev = { start: ev.start, end: ev.end, day: ev.day };
  ev.start = stampAt(payload.date, payload.start);
  ev.end = stampAt(payload.date, payload.end);
  ev.day = payload.date;
  relocate(ev);
  try {
    const saved = await api().patchPlannerEvent(ev.id, { start: ev.start, end: ev.end });
    Object.assign(ev, saved);
    relocate(ev);
  } catch (e) {
    ev.start = prev.start;
    ev.end = prev.end;
    ev.day = prev.day;
    relocate(ev);
    notifyError(e, "Could not move the event");
  }
}

async function toggleFocus(ev: PlannerEvent): Promise<void> {
  const prev = ev.focus;
  ev.focus = !prev;
  if (ev.focus && !ev.focus_before) ev.focus_before = 15;
  try {
    const saved = await api().patchPlannerEvent(ev.id, {
      focus: Boolean(ev.focus),
      focus_before: ev.focus_before || 15,
    });
    Object.assign(ev, saved);
  } catch (e) {
    ev.focus = prev;
    notifyError(e, "Could not update focus");
  }
}

function adopt(ev: PlannerEvent): void {
  const current = eventById(ev.id);
  if (current) Object.assign(current, ev);
  relocate(ev.id === current?.id ? current : ev);
}

async function removeSelected(): Promise<void> {
  const ev = selected.value;
  if (!ev) return;
  const ok = await confirmAction("Delete event", `Delete “${ev.title}”?`);
  if (!ok) return;
  try {
    await api().deletePlannerEvent(ev.id);
    for (const day of days.value) day.events = day.events.filter((item) => item.id !== ev.id);
    openDay({ eventId: "" });
  } catch (e) {
    notifyError(e, "Could not delete the event");
  }
}

function suggestionFor(sessionId: string): PlannerSuggestion | undefined {
  return (focused.value?.suggestions ?? []).find((row) => row.session_id === sessionId);
}

async function linkSuggestion(row: PlannerSuggestion): Promise<void> {
  try {
    await api().linkPlannerEvent(row.event_id, row.session_id);
    await load();
    Notify.create({ message: "Session linked.", position: "bottom-right", timeout: 1500 });
  } catch (e) {
    notifyError(e, "Could not link the session");
  }
}

async function planDay(): Promise<void> {
  try {
    const rows = await api().listChats({ origin_ref: date.value });
    const existing = rows
      .filter((chat) => isSideChat(chat) && chat.origin?.kind === "day")
      .sort((a, b) => (b.updated_at ?? "").localeCompare(a.updated_at ?? ""));
    if (existing[0]) {
      openChat(existing[0].id, { where: "split" });
      return;
    }
  } catch {
    /* start a new side chat */
  }
  const chat = await chats.createChat({
    title: `Plan ${date.value}`,
    origin: { kind: "day", ref: date.value, quote: dayHeading(date.value) },
  });
  if (chat) openChat(chat.id, { where: "split" });
}

function eventTarget(ev: PlannerEvent): Target {
  return {
    kind: "event",
    ref: ev.id,
    label: ev.title,
    text: ev.title,
    local: [
      { label: ev.prep ? "Regenerate prep" : "Prep", icon: "auto_awesome", run: () => void prep(ev) },
      {
        label: ev.focus ? "Remove focus" : "Focus",
        icon: ev.focus ? "star" : "star_border",
        run: () => void toggleFocus(ev),
      },
      { label: "Delete", icon: "delete", danger: true, run: () => void removeEvent(ev) },
      {
        label: "Open note",
        icon: "description",
        run: () => {
          if (ev.path) openFile(ev.path);
        },
      },
    ],
  };
}

async function prep(ev: PlannerEvent): Promise<void> {
  try {
    const job = await api().prepPlannerEvent(ev.id);
    ev.prep_job_status = "running";
    await jobs.watchJob(job.id);
    await load();
  } catch (e) {
    notifyError(e, "Could not generate prep");
  }
}

function onDrawerDeleted(): void {
  openDay({ eventId: "" });
  void load();
}

function linkTitle(sessionId: string): string {
  const row = suggestionFor(sessionId);
  const title = row ? eventById(row.event_id)?.title : "";
  return title ? `Link to ${title}` : "Link";
}

async function removeEvent(ev: PlannerEvent): Promise<void> {
  const ok = await confirmAction("Delete event", `Delete “${ev.title}”?`);
  if (!ok) return;
  try {
    await api().deletePlannerEvent(ev.id);
    for (const day of days.value) day.events = day.events.filter((item) => item.id !== ev.id);
    if (dayRoute.event === ev.id) openDay({ eventId: "" });
  } catch (e) {
    notifyError(e, "Could not delete the event");
  }
}

function sessionOf(ev: PlannerEvent): SessionSummary | null {
  if (!ev.session_id) return null;
  return sessionById(ev.session_id);
}

function typing(ev: KeyboardEvent): boolean {
  const el = ev.target instanceof HTMLElement ? ev.target : null;
  return !!el?.closest("input, textarea, select, [contenteditable], .cm-content");
}

function onKey(ev: KeyboardEvent): void {
  if (ev.metaKey || ev.ctrlKey || ev.altKey || typing(ev)) return;
  if (ev.key === "ArrowLeft") {
    ev.preventDefault();
    shift(-1);
  } else if (ev.key === "ArrowRight") {
    ev.preventDefault();
    shift(1);
  } else if (ev.key === "t") goToday();
  else if (ev.key === "d") setView("day");
  else if (ev.key === "w") setView("week");
  else if (ev.key === "n") {
    ev.preventDefault();
    newEvent();
  } else if ((ev.key === "Delete" || ev.key === "Backspace") && selected.value) {
    ev.preventDefault();
    void removeSelected();
  } else if (ev.key === "Escape") quick.value = null;
}

function bindVault(): void {
  stopEvents?.();
  stopDaily?.();
  const rootName = files.agentRoot || "Pawn";
  stopEvents = files.onVaultChange(`${rootName}/Events`, () => void load());
  stopDaily = files.onVaultChange(`${rootName}/Daily`, () => void load());
}

watch([date, week], () => void load());
watch(() => files.agentRoot, bindVault);
onMounted(() => {
  tick();
  clockTimer = window.setInterval(tick, 30_000);
  bindVault();
  void load();
});
onBeforeUnmount(() => {
  window.clearInterval(clockTimer);
  stopEvents?.();
  stopDaily?.();
});
</script>

<template>
  <div ref="root" class="day column no-wrap full-height" tabindex="0" :data-tab="tabId" @keydown="onKey">
    <div class="toolbar row items-center no-wrap q-px-sm q-py-xs border-b">
      <q-btn flat dense round icon="chevron_left" aria-label="Previous" @click="shift(-1)" />
      <q-btn flat dense no-caps label="Today" aria-label="Today" @click="goToday" />
      <q-btn flat dense round icon="chevron_right" aria-label="Next" @click="shift(1)" />
      <q-btn flat dense no-caps icon-right="arrow_drop_down" :label="heading" class="date-btn" aria-label="Choose date">
        <q-menu>
          <q-date
            minimal
            flat
            mask="YYYY-MM-DD"
            :model-value="date"
            @update:model-value="(value: string) => value && openDay({ date: value })"
          />
        </q-menu>
      </q-btn>
      <SegmentedControl
        :model-value="week ? 'week' : 'day'"
        size="sm"
        aria-label="Day or week"
        :options="[
          { value: 'day', label: 'Day', icon: 'today' },
          { value: 'week', label: 'Week', icon: 'date_range' },
        ]"
        @update:model-value="(value: string) => setView(value === 'week' ? 'week' : 'day')"
      />
      <q-space />
      <q-btn outline dense no-caps icon="auto_awesome" label="Plan my day" aria-label="Plan my day" @click="planDay" />
      <q-btn flat dense no-caps icon="add" label="Event" aria-label="New event" @click="newEvent" />
      <q-btn
        flat
        dense
        round
        :icon="agendaOpen ? 'view_sidebar' : 'view_agenda'"
        :class="{ 'toggle-on': agendaOpen }"
        aria-label="Toggle agenda"
        @click="agendaOpen = !agendaOpen"
      >
        <q-tooltip>{{ agendaOpen ? "Hide agenda" : "Show agenda" }}</q-tooltip>
      </q-btn>
      <q-btn flat dense round icon="more_vert" aria-label="Day menu">
        <q-menu>
          <q-list dense style="min-width: 220px">
            <q-item clickable @click="density = density === 'compact' ? 'comfortable' : 'compact'">
              <q-item-section>{{ density === "compact" ? "Comfortable density" : "Compact density" }}</q-item-section>
            </q-item>
            <q-item v-close-popup clickable @click="load">
              <q-item-section>Refresh</q-item-section>
            </q-item>
            <q-separator />
            <q-item-label header>Keys</q-item-label>
            <q-item><q-item-section>← → previous / next</q-item-section></q-item>
            <q-item><q-item-section>t today · d day · w week</q-item-section></q-item>
            <q-item><q-item-section>n new event · Del delete</q-item-section></q-item>
          </q-list>
        </q-menu>
      </q-btn>
    </div>
    <q-linear-progress v-if="loading" indeterminate size="2px" />
    <div class="row col no-wrap day-body">
      <TimeGrid
        class="col"
        :columns="columns"
        :px-per-hour="pxPerHour"
        :now-minutes="nowMinutes"
        :selected-id="dayRoute.event"
        @select="onSelect"
        @open="onOpen"
        @pick="onPick"
        @create="onCreate"
        @commit="onCommit"
      >
        <template #card="{ item, compact }">
          <div
            v-if="item.kind === 'event' && eventById(item.id)"
            v-target="eventTarget(eventById(item.id)!)"
            class="card-body"
            data-pawn-anchor-root
          >
            <div class="card-line">
              <span class="card-title">{{ eventById(item.id)!.title }}</span>
              <q-spinner v-if="eventById(item.id)!.prep_job_status === 'running'" size="12px" />
              <q-btn
                flat
                dense
                round
                size="xs"
                :icon="eventById(item.id)!.focus ? 'star' : 'star_border'"
                :color="eventById(item.id)!.focus ? 'amber-8' : undefined"
                aria-label="Toggle focus"
                @pointerdown.stop
                @click.stop="toggleFocus(eventById(item.id)!)"
              />
              <a
                v-if="eventById(item.id)!.meeting_url"
                class="join"
                :href="eventById(item.id)!.meeting_url"
                target="_blank"
                rel="noopener"
                @pointerdown.stop
                @click.stop
              >Join</a>
              <q-btn
                v-if="sessionOf(eventById(item.id)!)"
                flat
                dense
                round
                size="xs"
                icon="graphic_eq"
                aria-label="Open linked session"
                @pointerdown.stop
                @click.stop="openSession(sessionOf(eventById(item.id)!)!.session_id, { where: 'split' })"
              />
            </div>
            <div class="card-meta">{{ rangeLabel(item.start, item.end) }}</div>
            <div v-if="!compact && eventById(item.id)!.attendees?.length" class="card-meta ellipsis-1">
              {{ eventById(item.id)!.attendees!.join(", ") }}
            </div>
          </div>
          <div
            v-else-if="sessionById(item.id)"
            v-target="{ kind: 'session', ref: item.id, label: sessionById(item.id)!.title }"
            class="card-body"
          >
            <div class="card-line">
              <q-icon name="graphic_eq" size="14px" />
              <span class="card-title">{{ sessionById(item.id)!.title }}</span>
            </div>
            <div class="card-meta">{{ rangeLabel(item.start, item.end) }}</div>
          </div>
        </template>
      </TimeGrid>
      <EventDrawer
        v-if="selected"
        :event="selected"
        :sessions="focused?.sessions ?? []"
        :suggestions="focused?.suggestions ?? []"
        @close="openDay({ eventId: '' })"
        @saved="adopt"
        @deleted="onDrawerDeleted"
        @refresh="load"
      />
      <aside v-else-if="agendaOpen" class="rail column no-wrap">
        <div class="q-pa-sm col pawn-scroll">
          <button type="button" class="rail-head" @click="focused?.plan_path && openFile(focused.plan_path)">
            Plan
            <span v-if="focused?.plan_path" class="text-muted">{{ focused.plan_path }}</span>
          </button>
          <MarkdownView v-if="focused?.plan" :source="focused.plan" />
          <div v-else class="text-caption text-muted q-mb-md">No plan yet.</div>
          <div class="rail-head">Untimed ({{ focused?.events.filter((ev) => minutesOf(ev.start) == null).length ?? 0 }})</div>
          <button
            v-for="ev in focused?.events.filter((item) => minutesOf(item.start) == null) ?? []"
            :key="ev.id"
            type="button"
            class="rail-row"
            @click="openDay({ eventId: ev.id })"
          >
            {{ ev.title }}
          </button>
          <div class="rail-head">Open TODOs ({{ todos.length }})</div>
          <ItemCard
            v-for="item in todos"
            :key="item.id"
            :item="item"
            :checkbox="false"
            @action="(action, arg) => inbox.act(item, action, arg)"
          />
          <div v-if="!todos.length" class="text-caption text-muted q-mb-sm">Nothing open.</div>
          <div class="rail-head">Unlinked sessions ({{ unlinked.length }})</div>
          <div v-for="session in unlinked" :key="session.session_id" class="q-mb-sm">
            <button type="button" class="rail-row" @click="openSession(session.session_id, { where: 'split' })">
              {{ session.title }}
            </button>
            <q-btn
              v-if="suggestionFor(session.session_id)"
              flat
              dense
              no-caps
              size="sm"
              icon="link"
              :label="linkTitle(session.session_id)"
              @click="linkSuggestion(suggestionFor(session.session_id)!)"
            />
          </div>
        </div>
      </aside>
    </div>
    <div v-if="quick" class="quick-back" @pointerdown="quick = null">
      <div
        class="quick-pop"
        :style="{ left: `${quick.x}px`, top: `${quick.y}px` }"
        @pointerdown.stop
      >
        <div class="text-caption text-muted q-mb-xs">{{ quick.date }} · {{ rangeLabel(quick.start, quick.end) }}</div>
        <q-input
          v-model="quick.title"
          dense
          outlined
          autofocus
          placeholder="Event title"
          aria-label="Event title"
          @keyup.enter="saveQuick"
          @keyup.esc="quick = null"
        />
        <div class="row q-mt-xs">
          <q-btn flat dense no-caps label="More…" @click="moreQuick" />
          <q-space />
          <q-btn flat dense no-caps color="primary" label="Add" :disable="!quick.title.trim()" @click="saveQuick" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.day {
  min-height: 0;
  outline: none;
  position: relative;
}
.day-body {
  min-height: 0;
}
.toolbar {
  gap: 4px;
  min-width: 0;
}
.date-btn {
  font-variant-numeric: tabular-nums;
  margin-right: 4px;
}
.toggle-on {
  color: var(--q-primary);
}
.rail {
  width: 300px;
  flex: none;
  border-left: 1px solid var(--pawn-border);
  min-height: 0;
  background: var(--pawn-bg);
}
.rail-head {
  display: flex;
  gap: 8px;
  align-items: baseline;
  width: 100%;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--pawn-muted);
  text-align: left;
  padding: 10px 0 4px;
  cursor: pointer;
}
.rail-head .text-muted {
  text-transform: none;
  letter-spacing: 0;
  font-weight: 400;
}
.rail-row:hover {
  color: var(--q-primary);
}
.rail-row {
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  padding: 2px 0;
  cursor: pointer;
}
.card-body {
  height: 100%;
  padding: 2px 4px 4px;
  overflow: hidden;
}
.card-line {
  display: flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
}
.card-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-size: 0.75rem;
  font-weight: 650;
}
.card-meta {
  font-size: 0.68rem;
  color: var(--pawn-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.join {
  font-size: 0.68rem;
  color: var(--q-primary);
  text-decoration: none;
}
.quick-back {
  position: fixed;
  inset: 0;
  z-index: 20;
}
.quick-pop {
  position: fixed;
  width: 260px;
  padding: 10px;
  border-radius: 8px;
  background: var(--pawn-card);
  border: 1px solid var(--pawn-border);
  box-shadow: var(--pawn-shadow);
}
</style>

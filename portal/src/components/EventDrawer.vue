<script setup lang="ts">
// Event editor. Autosaves each field. Prep, snippets and the note body stay
// on the server; this drawer only sends the fields it owns.
import { computed, onBeforeUnmount, reactive, ref, watch } from "vue";
import { api, type PlannerEvent, type PlannerEventPatch, type PlannerSuggestion, type SessionSummary } from "@/api";
import { notifyError } from "@/lib/errors";
import { confirmAction } from "@/lib/dialogs";
import { clockOf, minutesOf } from "@/lib/planner";
import { openFile, openSession } from "@/router";
import { useJobsStore } from "@/stores/jobs";
import MarkdownView from "@/components/MarkdownView.vue";
import StatusBadge from "@/components/StatusBadge.vue";

const props = defineProps<{
  event: PlannerEvent;
  sessions: SessionSummary[];
  suggestions: PlannerSuggestion[];
}>();

const emit = defineEmits<{
  close: [];
  saved: [event: PlannerEvent];
  deleted: [];
  refresh: [];
}>();

const jobs = useJobsStore();
const draft = reactive({
  title: "",
  date: "",
  start: "",
  end: "",
  allDay: false,
  attendees: [] as string[],
  organizer: "",
  location: "",
  meetingUrl: "",
  focus: false,
  focusBefore: 15,
  sessionId: "",
});
const attendee = ref("");
const saved = ref(false);
const prepping = ref(false);
let hydrating = false;
let timer = 0;
let savedTimer = 0;
let lastSaved = "";

const running = computed(
  () => prepping.value || props.event.prep_job_status === "queued" || props.event.prep_job_status === "running",
);
const sessionOptions = computed(() => {
  const suggested = new Set(
    props.suggestions.filter((row) => row.event_id === props.event.id).map((row) => row.session_id),
  );
  return [...props.sessions]
    .sort((a, b) => Number(suggested.has(b.session_id)) - Number(suggested.has(a.session_id)))
    .map((row) => ({
      label: suggested.has(row.session_id) ? `${row.title} · suggested` : row.title,
      value: row.session_id,
    }));
});

function payload(): PlannerEventPatch {
  const startClock = (draft.start || "09:00").slice(0, 5);
  const endClock = draft.end ? draft.end.slice(0, 5) : "";
  const start = draft.allDay ? draft.date : `${draft.date}T${startClock}:00`;
  const end = draft.allDay ? "" : endClock ? `${draft.date}T${endClock}:00` : "";
  return {
    title: draft.title.trim(),
    start,
    end,
    attendees: [...draft.attendees],
    organizer: draft.organizer,
    location: draft.location,
    meeting_url: draft.meetingUrl,
    focus: draft.focus,
    focus_before: draft.focus ? draft.focusBefore || 15 : draft.focusBefore,
    session_id: draft.sessionId || null,
  };
}

function hydrate(): void {
  hydrating = true;
  const ev = props.event;
  const startMin = minutesOf(ev.start);
  draft.title = ev.title || "";
  draft.date = (ev.day || ev.start || "").slice(0, 10);
  draft.allDay = startMin == null;
  draft.start = startMin == null ? "09:00" : clockOf(startMin);
  const endMin = minutesOf(ev.end);
  draft.end = endMin == null ? "" : clockOf(endMin);
  draft.attendees = [...(ev.attendees ?? [])];
  draft.organizer = ev.organizer || "";
  draft.location = ev.location || "";
  draft.meetingUrl = ev.meeting_url || "";
  draft.focus = Boolean(ev.focus);
  draft.focusBefore = ev.focus_before || 15;
  draft.sessionId = ev.session_id || "";
  lastSaved = JSON.stringify(payload());
  queueMicrotask(() => {
    hydrating = false;
  });
}

function schedule(): void {
  window.clearTimeout(timer);
  timer = window.setTimeout(() => void save(), 400);
}

async function save(): Promise<void> {
  if (hydrating) return;
  const body = payload();
  if (!body.title) return;
  const sig = JSON.stringify(body);
  if (sig === lastSaved) return;
  try {
    const ev = await api().patchPlannerEvent(props.event.id, body);
    lastSaved = sig;
    saved.value = true;
    window.clearTimeout(savedTimer);
    savedTimer = window.setTimeout(() => {
      saved.value = false;
    }, 1200);
    emit("saved", ev);
  } catch (e) {
    notifyError(e, "Could not save the event");
  }
}

function addAttendee(): void {
  const name = attendee.value.trim();
  if (!name || draft.attendees.includes(name)) {
    attendee.value = "";
    return;
  }
  draft.attendees.push(name);
  attendee.value = "";
  schedule();
}

function removeAttendee(index: number): void {
  draft.attendees.splice(index, 1);
  schedule();
}

function onFocus(): void {
  draft.focus = !draft.focus;
  if (draft.focus && !draft.focusBefore) draft.focusBefore = 15;
  schedule();
}

async function generate(): Promise<void> {
  prepping.value = true;
  try {
    const job = await api().prepPlannerEvent(props.event.id);
    await jobs.watchJob(job.id);
    emit("refresh");
  } catch (e) {
    notifyError(e, "Could not generate prep");
  } finally {
    prepping.value = false;
  }
}

async function remove(): Promise<void> {
  const ok = await confirmAction("Delete event", `Delete “${props.event.title}”?`);
  if (!ok) return;
  try {
    await api().deletePlannerEvent(props.event.id);
    emit("deleted");
  } catch (e) {
    notifyError(e, "Could not delete the event");
  }
}

function editNote(): void {
  if (props.event.path) openFile(props.event.path, { where: "split", edit: true });
}

watch(() => props.event.id, hydrate, { immediate: true });
watch(draft, () => {
  if (!hydrating) schedule();
});
onBeforeUnmount(() => {
  window.clearTimeout(timer);
  window.clearTimeout(savedTimer);
});
</script>

<template>
  <aside class="drawer column no-wrap">
    <div class="row items-center q-px-sm q-pt-sm">
      <div class="section-label col q-pa-none">Event</div>
      <StatusBadge v-if="saved" kind="ok" icon="check" label="saved" class="q-mr-sm" />
      <q-btn flat dense round icon="close" aria-label="Close editor" @click="emit('close')" />
    </div>
    <div class="col q-pa-sm pawn-scroll">
      <q-input v-model="draft.title" dense outlined label="Title" class="q-mb-sm" />
      <q-input v-model="draft.date" dense outlined type="date" label="Date" class="q-mb-sm" />
      <div class="row q-gutter-xs q-mb-sm">
        <q-input v-model="draft.start" dense outlined type="time" label="Start" class="col" :disable="draft.allDay" />
        <q-input v-model="draft.end" dense outlined type="time" label="End" class="col" :disable="draft.allDay" />
      </div>
      <q-toggle v-model="draft.allDay" dense label="All day" class="q-mb-sm" />
      <div class="q-mb-sm">
        <div class="sub-label">Attendees</div>
        <div class="row q-gutter-xs q-mb-xs">
          <button
            v-for="(name, index) in draft.attendees"
            :key="name"
            type="button"
            class="pawn-pill"
            :aria-label="`Remove ${name}`"
            @click="removeAttendee(index)"
          >
            {{ name }} <q-icon name="close" class="x" />
          </button>
        </div>
        <q-input v-model="attendee" dense outlined placeholder="Add attendee" @keyup.enter="addAttendee" />
      </div>
      <q-input v-model="draft.organizer" dense outlined label="Organizer" class="q-mb-sm" />
      <q-input v-model="draft.location" dense outlined label="Location" class="q-mb-sm" />
      <div class="row items-end q-gutter-xs q-mb-sm">
        <q-input v-model="draft.meetingUrl" dense outlined label="Meeting URL" class="col" />
        <q-btn
          v-if="draft.meetingUrl"
          flat
          dense
          no-caps
          icon="open_in_new"
          label="Join"
          :href="draft.meetingUrl"
          target="_blank"
        />
      </div>
      <div class="row items-center q-mb-sm">
        <q-btn
          flat
          dense
          round
          :icon="draft.focus ? 'star' : 'star_border'"
          :color="draft.focus ? 'amber-8' : undefined"
          aria-label="Toggle focus"
          @click="onFocus"
        />
        <span class="q-mr-sm">Focus</span>
        <q-input
          v-model.number="draft.focusBefore"
          dense
          outlined
          type="number"
          label="Prep minutes"
          style="width: 120px"
          :disable="!draft.focus"
        />
      </div>
      <q-select
        v-model="draft.sessionId"
        dense
        outlined
        emit-value
        map-options
        clearable
        label="Linked session"
        :options="sessionOptions"
        class="q-mb-sm"
      />
      <q-btn
        v-if="draft.sessionId"
        flat
        dense
        no-caps
        icon="graphic_eq"
        label="Open session"
        class="q-mb-sm"
        @click="openSession(draft.sessionId, { where: 'split' })"
      />
      <div class="sub-label">Prep</div>
      <div v-if="running" class="row items-center q-mb-sm text-caption">
        <q-spinner size="16px" class="q-mr-xs" /> Generating prep…
      </div>
      <MarkdownView v-else-if="event.prep" :source="event.prep" class="q-mb-sm" />
      <div v-else class="text-caption text-muted q-mb-sm">No prep yet.</div>
      <q-btn
        flat
        dense
        no-caps
        icon="auto_awesome"
        :label="event.prep ? 'Regenerate' : 'Generate'"
        :loading="running"
        class="q-mb-md"
        @click="generate"
      />
      <template v-if="event.body">
        <div class="sub-label">Notes</div>
        <MarkdownView :source="event.body" class="q-mb-sm" />
      </template>
      <q-btn
        v-if="event.path"
        flat
        dense
        no-caps
        icon="edit_note"
        label="Edit note"
        class="q-mb-sm"
        @click="editNote"
      />
      <div v-if="event.source_url" class="q-mb-sm">
        <a :href="event.source_url" target="_blank" rel="noopener">{{ event.source || "Source" }}</a>
      </div>
      <q-btn flat dense no-caps color="negative" icon="delete" label="Delete" @click="remove" />
    </div>
  </aside>
</template>

<style scoped>
.drawer {
  width: 320px;
  flex: none;
  border-left: 1px solid var(--pawn-border);
  min-height: 0;
  background: var(--pawn-bg);
}
</style>

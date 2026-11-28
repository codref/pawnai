<script setup lang="ts">
// Speakers gallery. The selected person is `?id=` on this tool tab.
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { Notify } from "quasar";
import { api, type SpeakerDetail, type SpeakerEnrollmentRow, type SpeakerPatch, type SpeakerSummary } from "@/api";
import { notifyError } from "@/lib/errors";
import { confirmAction } from "@/lib/dialogs";
import { fileAt, formatTalk } from "@/lib/audio";
import { relativeTime } from "@/lib/time";
import { vTarget, type Target } from "@/lib/targets";
import { openFile, openSession, openSpeakers, speakersRoute } from "@/router";
import { useJobsStore } from "@/stores/jobs";
import { usePlayerStore } from "@/stores/player";
import { useTabsStore } from "@/stores/tabs";
import SpeakerChip from "@/components/SpeakerChip.vue";
import StatusBadge from "@/components/StatusBadge.vue";

const props = defineProps<{ tabId: string }>();

type SpeakerSort = "name" | "seen" | "sessions";
type SpeakerScope = "active" | "all";

const tabs = useTabsStore();
const jobs = useJobsStore();
const player = usePlayerStore();

const searchEl = ref<{ focus: () => void } | null>(null);
const audio = ref<HTMLAudioElement | null>(null);
const rows = ref<SpeakerSummary[]>([]);
const detail = ref<SpeakerDetail | null>(null);
const query = ref("");
const scope = ref<SpeakerScope>("active");
const sort = ref<SpeakerSort>("name");
const cursor = ref(0);
const loading = ref(false);
const nameDraft = ref("");
const notesDraft = ref("");
const aliasDraft = ref("");
const reidentify = ref(false);
const saved = ref(false);
const playing = ref("");
const mergeOpen = ref(false);
const mergeInto = ref("");
const mergeReidentify = ref(false);
let queryTimer = 0;
let notesTimer = 0;
let savedTimer = 0;
let clipEnd: number | null = null;
let quiet = false;

const sorted = computed(() => {
  const list = [...rows.value];
  list.sort((a, b) => {
    if (sort.value === "seen") return (b.last_seen_at ?? "").localeCompare(a.last_seen_at ?? "");
    if (sort.value === "sessions") return b.sessions - a.sessions || a.display_name.localeCompare(b.display_name);
    return a.display_name.localeCompare(b.display_name);
  });
  return list;
});

const others = computed(() => sorted.value.filter((row) => row.id !== detail.value?.id && row.active));

const target = computed<Target | null>(() =>
  detail.value ? { kind: "speaker", ref: detail.value.id, label: detail.value.display_name } : null,
);

const mergeTarget = computed(() => rows.value.find((row) => row.id === mergeInto.value) ?? null);

function clock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function flashSaved(): void {
  saved.value = true;
  window.clearTimeout(savedTimer);
  savedTimer = window.setTimeout(() => {
    saved.value = false;
  }, 1200);
}

function noteJob(jobId: string | null | undefined): void {
  if (!jobId) return;
  Notify.create({ message: "Reidentify queued", position: "bottom-right", timeout: 1500 });
  void jobs.watchJob(jobId);
}

async function reload(): Promise<void> {
  loading.value = true;
  try {
    rows.value = await api().listSpeakers({
      q: query.value.trim(),
      include_inactive: scope.value === "all",
    });
  } catch (e) {
    notifyError(e, "Could not load speakers");
  } finally {
    loading.value = false;
  }
}

async function loadDetail(id: string): Promise<void> {
  if (!id) {
    detail.value = null;
    return;
  }
  try {
    const row = await api().getSpeaker(id);
    quiet = true;
    detail.value = row;
    nameDraft.value = row.display_name;
    notesDraft.value = row.notes ?? "";
    aliasDraft.value = "";
    await nextTick();
    quiet = false;
  } catch (e) {
    detail.value = null;
    notifyError(e, "Could not open the speaker");
  }
}

function select(id: string): void {
  openSpeakers(id);
}

async function saveName(): Promise<void> {
  const row = detail.value;
  const name = nameDraft.value.trim();
  if (!row || !name || name === row.display_name) return;
  const body: SpeakerPatch = { display_name: name };
  if (reidentify.value) body.reidentify = true;
  try {
    const next = await api().patchSpeaker(row.id, body);
    detail.value = next;
    nameDraft.value = next.display_name;
    flashSaved();
    if (next.referenced_sessions) {
      Notify.create({
        message: `${next.referenced_sessions} sessions still use the previous name`,
        position: "bottom-right",
        timeout: 2000,
      });
    }
    noteJob(next.job_id);
    await reload();
  } catch (e) {
    notifyError(e, "Could not rename");
  }
}

async function saveAliases(aliases: string[]): Promise<void> {
  const row = detail.value;
  if (!row) return;
  try {
    const next = await api().patchSpeaker(row.id, { aliases });
    detail.value = next;
    flashSaved();
    await reload();
  } catch (e) {
    notifyError(e, "Could not update aliases");
  }
}

function addAlias(): void {
  const row = detail.value;
  const alias = aliasDraft.value.trim();
  if (!row || !alias) return;
  if (row.aliases.some((item) => item.toLowerCase() === alias.toLowerCase())) {
    aliasDraft.value = "";
    return;
  }
  aliasDraft.value = "";
  void saveAliases([...row.aliases, alias]);
}

function dropAlias(alias: string): void {
  const row = detail.value;
  if (!row) return;
  void saveAliases(row.aliases.filter((item) => item !== alias));
}

async function saveNotes(): Promise<void> {
  const row = detail.value;
  if (!row || quiet) return;
  if ((row.notes ?? "") === notesDraft.value) return;
  try {
    const next = await api().patchSpeaker(row.id, { notes: notesDraft.value });
    detail.value = { ...row, notes: next.notes };
    flashSaved();
  } catch (e) {
    notifyError(e, "Could not save notes");
  }
}

async function openPeople(): Promise<void> {
  const row = detail.value;
  if (!row) return;
  try {
    if (row.people_path) {
      openFile(row.people_path);
      return;
    }
    const created = await api().ensureSpeakerNote(row.id);
    detail.value = { ...row, people_path: created.path };
    const listed = rows.value.find((item) => item.id === row.id);
    if (listed) listed.people_path = created.path;
    openFile(created.path);
  } catch (e) {
    notifyError(e, "Could not open the people note");
  }
}

function stopClip(): void {
  const owner = playing.value ? `enroll:${playing.value}` : "";
  audio.value?.pause();
  playing.value = "";
  clipEnd = null;
  if (owner) player.release(owner);
}

async function toggleClip(row: SpeakerEnrollmentRow): Promise<void> {
  if (playing.value === row.id) {
    stopClip();
    return;
  }
  const el = audio.value;
  if (!el || !row.source_session_id) return;
  try {
    const urls = await api().sessionAudioUrl(row.source_session_id);
    if (!urls.files.length) return;
    // Enrollment times are session-global; map them into the right chunk.
    const hit = fileAt(urls.files, row.start_time ?? 0);
    const file = urls.files.find((item) => item.index === hit.index) ?? urls.files[0];
    clipEnd = row.end_time != null ? row.end_time - file.offset : null;
    player.claim(`enroll:${row.id}`);
    playing.value = row.id;
    el.src = file.url;
    const begin = hit.local;
    const start = (): void => {
      if (Number.isFinite(el.duration) && begin > 0 && begin < el.duration) el.currentTime = begin;
      void el.play();
    };
    if (el.readyState >= 1) start();
    else el.onloadedmetadata = start;
  } catch (e) {
    stopClip();
    notifyError(e, "Could not play the enrollment");
  }
}

function onClipTime(): void {
  const el = audio.value;
  if (!el || clipEnd == null) return;
  if (el.currentTime >= clipEnd) stopClip();
}

function onClipError(): void {
  if (!playing.value) return;
  stopClip();
  Notify.create({ type: "negative", message: "Could not play the enrollment audio", position: "bottom-right" });
}

async function removeEnrollment(row: SpeakerEnrollmentRow): Promise<void> {
  const person = detail.value;
  if (!person) return;
  const ok = await confirmAction("Remove enrollment", "This voiceprint is deleted.", "Remove");
  if (!ok) return;
  try {
    if (playing.value === row.id) stopClip();
    await api().deleteEnrollment(person.id, row.id);
    await loadDetail(person.id);
    await reload();
  } catch (e) {
    notifyError(e, "Could not remove the enrollment");
  }
}

function askMerge(): void {
  const first = others.value[0];
  if (!first || !detail.value) return;
  mergeInto.value = first.id;
  mergeReidentify.value = false;
  mergeOpen.value = true;
}

async function doMerge(): Promise<void> {
  const row = detail.value;
  const into = mergeInto.value;
  if (!row || !into) return;
  try {
    const result = await api().mergeSpeakers(row.id, into, mergeReidentify.value);
    mergeOpen.value = false;
    flashSaved();
    noteJob(result.job_id);
    openSpeakers(result.into);
    await reload();
    await loadDetail(result.into);
  } catch (e) {
    notifyError(e, "Could not merge");
  }
}

async function deactivate(): Promise<void> {
  const row = detail.value;
  if (!row) return;
  const ok = await confirmAction(
    `Deactivate ${row.display_name}`,
    "The person leaves the active gallery. Enrollments stay until you remove them.",
    "Deactivate",
  );
  if (!ok) return;
  try {
    await api().deactivateSpeaker(row.id);
    await reload();
    if (scope.value === "active") {
      detail.value = null;
      openSpeakers("");
      return;
    }
    await loadDetail(row.id);
  } catch (e) {
    notifyError(e, "Could not deactivate");
  }
}

function onKey(ev: KeyboardEvent): void {
  const targetEl = ev.target as HTMLElement | null;
  if (targetEl?.closest("input, textarea, select, [contenteditable], .cm-content")) return;
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const list = sorted.value;
  if (ev.key === "/") {
    ev.preventDefault();
    searchEl.value?.focus();
    return;
  }
  if (ev.key === "j" || ev.key === "ArrowDown") {
    ev.preventDefault();
    cursor.value = Math.min(list.length - 1, cursor.value + 1);
    const row = list[cursor.value];
    if (row) select(row.id);
    return;
  }
  if (ev.key === "k" || ev.key === "ArrowUp") {
    ev.preventDefault();
    cursor.value = Math.max(0, cursor.value - 1);
    const row = list[cursor.value];
    if (row) select(row.id);
    return;
  }
  if (ev.key === "Enter") {
    const row = list[cursor.value];
    if (row) select(row.id);
  }
}

watch(notesDraft, () => {
  if (quiet) return;
  window.clearTimeout(notesTimer);
  notesTimer = window.setTimeout(() => void saveNotes(), 400);
});

watch(
  () => speakersRoute.id,
  (id) => {
    const title = detail.value && detail.value.id === id ? detail.value.display_name : id;
    tabs.setStatus(props.tabId, { title: title ? `Speakers · ${title}` : "Speakers" });
    if (id) void loadDetail(id);
    else detail.value = null;
    const index = sorted.value.findIndex((row) => row.id === id);
    if (index >= 0) cursor.value = index;
  },
  { immediate: true },
);

watch(detail, (row) => {
  const id = speakersRoute.id;
  const title = row && row.id === id ? row.display_name : id;
  tabs.setStatus(props.tabId, { title: title ? `Speakers · ${title}` : "Speakers" });
});

watch([query, scope], () => {
  window.clearTimeout(queryTimer);
  queryTimer = window.setTimeout(() => void reload(), 250);
});

watch(
  () => player.owner,
  (owner) => {
    if (playing.value && owner !== `enroll:${playing.value}`) {
      audio.value?.pause();
      playing.value = "";
      clipEnd = null;
    }
  },
);

onMounted(() => {
  void reload();
});
</script>

<template>
  <div class="speakers column no-wrap full-height" tabindex="0" @keydown="onKey">
    <div class="row items-center q-pa-xs q-gutter-xs border-b">
      <q-input
        ref="searchEl"
        v-model="query"
        dense
        outlined
        clearable
        class="col"
        placeholder="Search speakers"
        aria-label="Search speakers"
        @clear="query = ''"
      >
        <template #prepend><q-icon name="search" /></template>
      </q-input>
      <q-btn-toggle
        v-model="scope"
        dense
        no-caps
        toggle-color="primary"
        :options="[
          { label: 'Active', value: 'active' },
          { label: 'All', value: 'all' },
        ]"
      />
      <q-btn-dropdown dense no-caps outline :label="sort === 'name' ? 'Name' : sort === 'seen' ? 'Last seen' : 'Sessions'">
        <q-list dense>
          <q-item v-close-popup clickable :active="sort === 'name'" @click="sort = 'name'"><q-item-section>Name</q-item-section></q-item>
          <q-item v-close-popup clickable :active="sort === 'seen'" @click="sort = 'seen'"><q-item-section>Last seen</q-item-section></q-item>
          <q-item v-close-popup clickable :active="sort === 'sessions'" @click="sort = 'sessions'"><q-item-section>Sessions</q-item-section></q-item>
        </q-list>
      </q-btn-dropdown>
      <q-btn flat dense round icon="refresh" aria-label="Refresh" :loading="loading" @click="reload()" />
      <q-btn flat dense round icon="help_outline" aria-label="Keyboard shortcuts">
        <q-menu anchor="bottom right" self="top right">
          <q-list dense style="min-width: 180px">
            <q-item><q-item-section>j / k move</q-item-section></q-item>
            <q-item><q-item-section>Enter select</q-item-section></q-item>
            <q-item><q-item-section>/ search</q-item-section></q-item>
          </q-list>
        </q-menu>
      </q-btn>
      <StatusBadge v-if="saved" kind="ok" icon="check" label="saved" />
    </div>

    <div class="row col no-wrap body">
      <aside class="list col-auto pawn-scroll">
        <button
          v-for="(row, index) in sorted"
          :key="row.id"
          type="button"
          class="person"
          :class="{ picked: row.id === speakersRoute.id, cursor: index === cursor, inactive: !row.active }"
          @click="select(row.id)"
        >
          <SpeakerChip :name="row.display_name" :size="28" />
          <span class="col meta">
            <span v-if="row.aliases.length" class="text-caption text-muted ellipsis">{{ row.aliases.join(", ") }}</span>
            <span class="text-caption text-muted">
              {{ row.enrollments }} voiceprints · {{ row.sessions }} sessions
              <template v-if="row.last_seen_at"> · {{ relativeTime(row.last_seen_at) }}</template>
            </span>
          </span>
        </button>
        <div v-if="!sorted.length && !loading" class="q-pa-md text-caption text-muted">No speakers</div>
      </aside>

      <section v-if="detail" v-target="target" class="detail col pawn-scroll q-pa-md">
        <div class="row items-end q-gutter-sm">
          <q-input
            v-model="nameDraft"
            class="col"
            dense
            outlined
            label="Name"
            @blur="saveName"
            @keyup.enter="saveName"
          />
          <q-checkbox v-model="reidentify" dense label="Reidentify on rename" />
        </div>
        <div class="text-caption text-muted q-mt-xs">
          {{ detail.enrollments }} voiceprints · {{ formatTalk(detail.seconds) }} enrolled
          <StatusBadge v-if="!detail.active" kind="neutral" label="inactive" class="q-ml-xs" />
        </div>

        <div class="q-mt-md sub-label">Aliases</div>
        <div class="row items-center q-gutter-xs q-mt-xs">
          <button
            v-for="alias in detail.aliases"
            :key="alias"
            type="button"
            class="pawn-pill"
            :aria-label="`Remove alias ${alias}`"
            @click="dropAlias(alias)"
          >
            {{ alias }} <q-icon name="close" class="x" />
          </button>
          <q-input
            v-model="aliasDraft"
            dense
            outlined
            placeholder="Add alias"
            aria-label="Add alias"
            style="width: 10rem"
            @keyup.enter="addAlias"
          />
        </div>

        <div class="row q-gutter-sm q-mt-md">
          <q-btn dense no-caps outline icon="person" :label="detail.people_path ? 'People note' : 'Create people note'" @click="openPeople" />
          <q-btn dense no-caps outline icon="merge" label="Merge into…" :disable="!others.length" @click="askMerge" />
          <q-btn dense no-caps flat color="negative" label="Deactivate" :disable="!detail.active" @click="deactivate" />
        </div>

        <div class="q-mt-lg sub-label">Notes</div>
        <q-input v-model="notesDraft" class="q-mt-xs" type="textarea" autogrow outlined dense aria-label="Notes" />

        <div class="q-mt-lg sub-label">Enrollments</div>
        <div v-if="!detail.enrollment_rows.length" class="text-caption text-muted q-mt-xs">No voiceprints</div>
        <div v-for="row in detail.enrollment_rows" :key="row.id" class="row items-center no-wrap q-gutter-x-sm q-mt-xs">
          <q-btn
            flat
            dense
            round
            :icon="playing === row.id ? 'pause' : 'play_arrow'"
            :aria-label="playing === row.id ? 'Pause enrollment' : 'Play enrollment'"
            :disable="!row.source_session_id"
            @click="toggleClip(row)"
          />
          <span class="clock">{{ clock(row.start_time) }}–{{ clock(row.end_time) }}</span>
          <span class="text-caption text-muted ellipsis">{{ row.source_session_id }}</span>
          <q-space />
          <q-btn flat dense round icon="close" aria-label="Remove enrollment" @click="removeEnrollment(row)" />
        </div>

        <div class="q-mt-lg sub-label">Sessions</div>
        <q-list dense class="q-mt-xs">
          <q-item
            v-for="session in detail.recent_sessions"
            :key="session.session_id"
            clickable
            @click="openSession(session.session_id)"
          >
            <q-item-section>{{ session.title }}</q-item-section>
            <q-item-section side class="text-caption">{{ relativeTime(session.created_at) }}</q-item-section>
          </q-item>
        </q-list>
      </section>
      <section v-else class="detail col flex-center text-muted">Select a speaker</section>
    </div>

    <audio
      ref="audio"
      class="clip"
      @timeupdate="onClipTime"
      @ended="stopClip"
      @error="onClipError"
    />

    <q-dialog v-model="mergeOpen">
      <q-card style="min-width: 22rem">
        <q-card-section class="text-h6">Merge {{ detail?.display_name }}</q-card-section>
        <q-card-section class="q-pt-none">
          <q-select
            v-model="mergeInto"
            dense
            outlined
            emit-value
            map-options
            label="Merge into"
            :options="others.map((row) => ({ label: row.display_name, value: row.id }))"
          />
          <ul v-if="detail && mergeTarget" class="effects q-mt-sm">
            <li>{{ detail.enrollments }} voiceprints move to {{ mergeTarget.display_name }}</li>
            <li>{{ detail.display_name }} is added as an alias</li>
            <li>{{ detail.display_name }} is deactivated</li>
            <li>{{ detail.sessions }} sessions are repointed</li>
          </ul>
          <q-checkbox v-model="mergeReidentify" dense label="Reidentify those sessions" />
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat no-caps label="Cancel" v-close-popup />
          <q-btn flat no-caps color="primary" label="Merge" :disable="!mergeInto" @click="doMerge" />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<style scoped>
.body {
  min-height: 0;
}
.list {
  width: 320px;
  border-right: 1px solid var(--pawn-border);
}
.person {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid var(--pawn-border);
  background: transparent;
  color: inherit;
  padding: 8px;
  cursor: pointer;
}
.person:hover,
.person.cursor {
  background: var(--pawn-hover);
}
.person.picked {
  background: var(--pawn-active);
}
.person.inactive {
  opacity: 0.55;
}
.meta {
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.detail {
  min-width: 0;
}
.clock {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.clip {
  display: none;
}
.effects {
  margin: 0;
  padding-left: 1.2rem;
}
@media (max-width: 800px) {
  .body {
    flex-direction: column;
  }
  .list {
    width: 100%;
    max-height: 40%;
    border-right: 0;
    border-bottom: 1px solid var(--pawn-border);
  }
}
</style>

<script lang="ts">
let paintedNow = "";
let paintedFind = "";
</script>

<script setup lang="ts">
// One recording: player, transcript, summary, items, and the speakers rail.
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Notify } from "quasar";
import {
  api,
  type InboxItem,
  type ItemAction,
  type SessionDetail,
  type SessionSample,
  type SessionTranscript,
  type TranscriptSegment,
} from "@/api";
import { errorText, notifyError } from "@/lib/errors";
import { confirmAction } from "@/lib/dialogs";
import { formatClock, formatTalk } from "@/lib/audio";
import { filterTurns } from "@/lib/transcript";
import { speakerColor } from "@/lib/speakers";
import { vTarget, type Target } from "@/lib/targets";
import { askAbout, type ActionContext } from "@/lib/actions";
import { linkFor, openChat, openDay, openFile, router, sessionRoute } from "@/router";
import { useFilesStore } from "@/stores/files";
import { useInboxStore } from "@/stores/inbox";
import { useJobsStore } from "@/stores/jobs";
import { useSessionsStore } from "@/stores/sessions";
import { useTabsStore } from "@/stores/tabs";
import AudioPlayer, { type PlayableFile } from "@/components/AudioPlayer.vue";
import ItemCard from "@/components/ItemCard.vue";
import MarkdownView from "@/components/MarkdownView.vue";
import SessionActionDialogs from "@/components/SessionActionDialogs.vue";
import SpeakerChip from "@/components/SpeakerChip.vue";
import SpeakerRail from "@/components/SpeakerRail.vue";
import SegmentedControl from "@/components/SegmentedControl.vue";
import StatusBadge from "@/components/StatusBadge.vue";

const props = defineProps<{ id: string; tabId: string }>();

const files = useFilesStore();
const inbox = useInboxStore();
const jobs = useJobsStore();
const sessions = useSessionsStore();
const tabs = useTabsStore();

const detail = ref<SessionDetail | null>(null);
const transcript = ref<SessionTranscript | null>(null);
const items = ref<InboxItem[]>([]);
const analysisMd = ref("");
const audioFiles = ref<PlayableFile[]>([]);
const error = ref("");
const section = ref<"transcript" | "summary" | "items">("transcript");
const find = ref("");
const speakerFilter = ref("");
const time = ref(0);
const follow = ref(true);
const narrow = ref(false);
const railOpen = ref(true);
const showExtract = ref(false);
const showReprocess = ref(false);
const banner = ref(0);
const transcriptEl = ref<HTMLElement | null>(null);
const findEl = ref<{ focus: () => void } | null>(null);
const playerRef = ref<{
  seek: (seconds: number) => Promise<void>;
  play: () => Promise<void>;
  toggle: () => void;
  playRange: (start: number, end: number) => Promise<void>;
  stopRange: () => void;
} | null>(null);
/** Start time of the sample set being played (the rail shows ■ for it). */
const activeSample = ref<number | null>(null);
const audioError = ref("");
const railRef = ref<{ focusFirst: () => void } | null>(null);

let stopTranscript: (() => void) | null = null;
let stopAnalysis: (() => void) | null = null;
let poll = 0;
let programmatic = false;
let sampleToken = 0;

const target = computed<Target>(() => ({
  kind: "session",
  ref: props.id,
  label: detail.value?.title || props.id,
}));
const sideChats = computed(() => detail.value?.side_chats ?? []);
const duration = computed(() => transcript.value?.duration ?? detail.value?.duration_seconds ?? 0);
const flatSegments = computed(() => {
  const out: { label: string; start: number; end: number }[] = [];
  for (const turn of transcript.value?.turns ?? []) {
    for (const seg of turn.segments) out.push({ label: turn.label, start: seg.start, end: seg.end });
  }
  return out;
});
const visibleTurns = computed(() =>
  filterTurns(transcript.value?.turns ?? [], { speaker: speakerFilter.value, find: find.value }),
);
const currentId = computed(() => segmentAt(time.value)?.id ?? "");
/** Mapped display name for a session label (falls back to the label). */
const nameOf = computed(() => {
  const map = new Map<string, string>();
  for (const spk of transcript.value?.speakers ?? []) if (spk.name) map.set(spk.label, spk.name);
  for (const label of detail.value?.labels ?? []) if (label.mapped_name) map.set(label.local_label, label.mapped_name);
  return (label: string) => map.get(label) ?? label;
});

function segmentAt(seconds: number): TranscriptSegment | null {
  let last: TranscriptSegment | null = null;
  for (const turn of transcript.value?.turns ?? []) {
    for (const seg of turn.segments) {
      last = seg;
      if (seconds >= seg.start && seconds < seg.end) return seg;
    }
  }
  return last && seconds >= last.start ? last : null;
}

function highlights(): { set: (name: string, value: Highlight) => void; delete: (name: string) => void } | null {
  const css = CSS as unknown as { highlights?: { set: (name: string, value: Highlight) => void; delete: (name: string) => void } };
  return typeof Highlight === "undefined" ? null : (css.highlights ?? null);
}

function paintNow(): void {
  const apiHighlights = highlights();
  if (!apiHighlights || !transcriptEl.value) return;
  apiHighlights.delete("pawn-now");
  const id = currentId.value;
  if (!id) return;
  const el = transcriptEl.value.querySelector(`[data-seg="${CSS.escape(id)}"]`);
  if (!el) return;
  const range = document.createRange();
  range.selectNodeContents(el);
  apiHighlights.set("pawn-now", new Highlight(range));
  paintedNow = props.id;
}

function paintFind(): void {
  const apiHighlights = highlights();
  if (!apiHighlights || !transcriptEl.value) return;
  apiHighlights.delete("pawn-find");
  const needle = find.value.trim().toLowerCase();
  if (!needle) return;
  const mark = new Highlight();
  transcriptEl.value.querySelectorAll<HTMLElement>("[data-seg]").forEach((el) => {
    const text = el.textContent ?? "";
    const at = text.toLowerCase().indexOf(needle);
    const node = el.firstChild;
    if (at < 0 || !node) return;
    const range = document.createRange();
    range.setStart(node, at);
    range.setEnd(node, Math.min(text.length, at + needle.length));
    mark.add(range);
  });
  if (mark.size) {
    apiHighlights.set("pawn-find", mark);
    paintedFind = props.id;
  }
}

function scrollToCurrent(): void {
  const el = transcriptEl.value?.querySelector("[data-current='1']");
  if (!el) return;
  programmatic = true;
  el.scrollIntoView({ block: "nearest" });
  window.setTimeout(() => {
    programmatic = false;
  }, 60);
}

function onTranscriptScroll(): void {
  if (!programmatic) follow.value = false;
}

/** Detail, transcript, items and analysis. Never touches the player. */
async function reload(): Promise<void> {
  error.value = "";
  try {
    const [next, spoken, page] = await Promise.all([
      api().getSession(props.id),
      api().sessionTranscript(props.id).catch((e: unknown) => {
        if (e && typeof e === "object" && "status" in e && (e as { status: number }).status === 404) return null;
        throw e;
      }),
      api().listItems({ source_ref: props.id, limit: 100 }),
    ]);
    detail.value = next;
    transcript.value = spoken;
    items.value = page.items;
    analysisMd.value = "";
    if (next.analysis_path) {
      try {
        analysisMd.value = (await api().readFile(next.analysis_path)).content;
      } catch {
        analysisMd.value = "";
      }
    }
    // Re-sign the audio only when the recording's file list changed (first
    // load, reprocess, a new chunk). A plain detail refresh keeps playing.
    const layout = (spoken?.files ?? []).map((f) => `${f.index}:${f.audio_file}`).join("|");
    if (layout !== audioLayout || !audioFiles.value.length) {
      audioLayout = layout;
      await reloadAudio();
    }
  } catch (e) {
    error.value = errorText(e);
  }
}

let audioLayout = "";

async function signedFiles(): Promise<PlayableFile[] | null> {
  try {
    const urls = await api().sessionAudioUrl(props.id);
    const spans = transcript.value?.files ?? [];
    audioError.value = "";
    return urls.files.map((file) => {
      const span = spans.find((item) => item.index === file.index);
      return {
        index: file.index,
        url: file.url,
        exp: file.exp,
        offset: file.offset ?? span?.offset ?? 0,
        duration: file.duration ?? span?.duration ?? 0,
        estimated: file.estimated ?? span?.estimated,
      };
    });
  } catch (e) {
    audioError.value = errorText(e);
    return null;
  }
}

async function reloadAudio(): Promise<void> {
  audioFiles.value = (await signedFiles()) ?? [];
}

function chatAbout(ev: Event): void {
  const point = ev instanceof MouseEvent ? ev : null;
  const ctx: ActionContext = { x: point?.clientX ?? 0, y: point?.clientY ?? 0 };
  void askAbout(target.value, ctx);
}

function copyLink(): void {
  void navigator.clipboard.writeText(linkFor("session", props.id)).then(() => {
    Notify.create({ message: "Link copied", position: "bottom-right", timeout: 1500 });
  });
}

function localStamp(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso.slice(0, 10);
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

function when(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso.slice(0, 16);
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function openEvent(): void {
  const event = detail.value?.event;
  if (!event) return;
  openDay({ date: localStamp(detail.value?.created_at), eventId: event.id });
}

async function seekPlay(seconds: number): Promise<void> {
  follow.value = true;
  await playerRef.value?.seek(seconds);
  await playerRef.value?.play();
}

function speakerLabelOf(label: string): string {
  return nameOf.value(label);
}

function toggleSpeaker(label: string): void {
  speakerFilter.value = speakerFilter.value === label ? "" : label;
}

function stepTurn(delta: number): void {
  const turns = visibleTurns.value;
  if (!turns.length) return;
  const current = turns.findIndex((turn) => time.value >= turn.start && time.value < turn.end);
  const next = Math.min(turns.length - 1, Math.max(0, (current < 0 ? 0 : current) + delta));
  void seekPlay(turns[next].start);
}

function onStrip(ev: MouseEvent): void {
  const el = ev.currentTarget as HTMLElement;
  const rect = el.getBoundingClientRect();
  if (!rect.width || !duration.value) return;
  const ratio = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
  void seekPlay(ratio * duration.value);
}

/** Play a label's samples back to back, each stopping at its end. Clicking
 * the same label again stops it. */
async function playSamples(samples: SessionSample[]): Promise<void> {
  const first = samples[0]?.start ?? null;
  const player = playerRef.value;
  if (!player || first == null) return;
  if (activeSample.value === first) {
    sampleToken += 1;
    activeSample.value = null;
    player.stopRange();
    return;
  }
  const token = ++sampleToken;
  activeSample.value = first;
  follow.value = true;
  for (const sample of samples) {
    if (token !== sampleToken) return;
    await player.playRange(sample.start, sample.end);
  }
  if (token === sampleToken) activeSample.value = null;
}

async function onItem(item: InboxItem, action: ItemAction, arg?: string): Promise<void> {
  if (action === "delete") {
    const ok = await confirmAction("Delete item", `Delete “${item.text.slice(0, 80)}”?`);
    if (!ok) return;
  }
  if (await inbox.act(item, action, arg)) await reload();
}

async function onEnrolled(): Promise<void> {
  await reload();
  const page = await api().listSessions({ needs_speakers: true, limit: 50 });
  const total = page.facets.total ?? page.data.length;
  const includesSelf = page.data.some((row) => row.session_id === props.id);
  banner.value = Math.max(0, total - (includesSelf ? 1 : 0));
}

async function reidentifyOthers(): Promise<void> {
  try {
    const result = await api().reidentifySessions({ needs_speakers: true, limit: 50 });
    if (result.job_id) void jobs.watchJob(result.job_id);
    banner.value = 0;
    Notify.create({ message: "Reidentify queued.", position: "bottom-right" });
  } catch (e) {
    notifyError(e, "Reidentify failed");
  }
}

function typing(ev: KeyboardEvent): boolean {
  const el = ev.target instanceof HTMLElement ? ev.target : null;
  return !!el?.closest("input, textarea, select, [contenteditable]");
}

function onKey(ev: KeyboardEvent): void {
  if (ev.metaKey || ev.ctrlKey || ev.altKey || typing(ev)) return;
  if (ev.key === " " ) {
    ev.preventDefault();
    playerRef.value?.toggle();
  } else if (ev.key === "n") {
    stepTurn(1);
  } else if (ev.key === "p") {
    stepTurn(-1);
  } else if (ev.key === "/") {
    ev.preventDefault();
    section.value = "transcript";
    void nextTick(() => findEl.value?.focus());
  }
}

function armPoll(processing: boolean | undefined): void {
  window.clearInterval(poll);
  if (!processing) return;
  poll = window.setInterval(() => void reload(), 5000);
}

watch(
  () => (sessionRoute.id === props.id ? sessionRoute.section : ""),
  (value) => {
    if (value === "items" || value === "summary") section.value = value;
  },
  { immediate: true },
);
// A board or dossier asked to jump to a moment: play from there once the player exists.
watch(
  () => [sessionRoute.id === props.id ? sessionRoute.seek : null, playerRef.value] as const,
  ([seek, player]) => {
    if (seek === null || !player) return;
    sessionRoute.seek = null;
    section.value = "transcript";
    void seekPlay(seek);
  },
  { immediate: true },
);
watch(section, (value) => {
  if (sessionRoute.id !== props.id || sessionRoute.section === value) return;
  sessionRoute.section = value === "transcript" ? "" : value;
  if (tabs.activeId !== props.tabId) return;
  void router.replace({
    name: "session",
    params: { id: props.id },
    query: sessionRoute.section ? { section: sessionRoute.section } : {},
  });
});
watch(
  () => detail.value?.title,
  (title) => tabs.setStatus(props.tabId, { title: title || props.id }),
  { immediate: true },
);
watch(() => props.id, () => void reload());
watch(() => detail.value?.processing, armPoll);
watch([time, currentId, visibleTurns], async () => {
  await nextTick();
  paintNow();
  paintFind();
  if (follow.value) scrollToCurrent();
});
watch(find, async () => {
  await nextTick();
  paintFind();
});

onMounted(() => {
  const mq = window.matchMedia("(max-width: 900px)");
  const apply = () => {
    narrow.value = mq.matches;
    if (mq.matches) railOpen.value = false;
  };
  apply();
  mq.addEventListener("change", apply);
  // Only this session's notes: other sessions' chunk pushes must not reload.
  const root = files.agentRoot || "Pawn";
  const mine = (paths: string[]) =>
    paths.some((path) => path === detail.value?.transcript_path || path === detail.value?.analysis_path);
  stopTranscript = files.onVaultChange(`${root}/Transcripts/`, (paths) => {
    if (mine(paths)) void reload();
  });
  stopAnalysis = files.onVaultChange(`${root}/Analyses/`, (paths) => {
    if (mine(paths)) void reload();
  });
  void reload().then(async () => {
    if (sessions.focusRail === props.id) {
      sessions.focusRail = "";
      railOpen.value = true;
      await nextTick();
      railRef.value?.focusFirst();
    }
  });
});

onBeforeUnmount(() => {
  stopTranscript?.();
  stopAnalysis?.();
  window.clearInterval(poll);
  sampleToken += 1;
  const apiHighlights = highlights();
  if (paintedNow === props.id) {
    apiHighlights?.delete("pawn-now");
    paintedNow = "";
  }
  if (paintedFind === props.id) {
    apiHighlights?.delete("pawn-find");
    paintedFind = "";
  }
});
</script>

<template>
  <div v-if="error && !detail" class="q-pa-lg text-center">
    <q-icon name="graphic_eq" size="48px" class="text-muted" />
    <div class="q-mt-sm">{{ error }}</div>
  </div>
  <div v-else class="column no-wrap full-height" tabindex="0" :data-tab="tabId" @keydown="onKey">
    <div v-target="target" class="row items-center q-px-md q-py-sm border-b">
      <div class="col" style="min-width: 0">
        <div class="text-subtitle1 ellipsis">{{ detail?.title || id }}</div>
        <div class="text-caption text-muted row items-center q-gutter-x-sm">
          <span>{{ when(detail?.created_at) }}</span>
          <span>{{ formatTalk(detail?.duration_seconds ?? duration) }}</span>
          <StatusBadge v-if="detail?.processing" kind="info" busy label="processing" />
          <StatusBadge v-else-if="detail?.needs_speakers" kind="warn" label="needs training" />
          <StatusBadge v-else-if="detail?.speakers_trained" kind="ok" label="trained" />
          <q-btn
            v-if="detail?.event"
            flat
            dense
            no-caps
            size="sm"
            icon="event"
            :label="detail.event.title"
            @click="openEvent"
          />
        </div>
      </div>
      <q-btn-dropdown
        split
        dense
        no-caps
        outline
        icon="chat"
        label="Chat about this"
        class="q-mr-xs"
        @click="chatAbout"
      >
        <q-list dense style="min-width: 200px">
          <q-item v-if="!sideChats.length" dense>
            <q-item-section class="text-muted">No side chats yet</q-item-section>
          </q-item>
          <q-item v-for="chat in sideChats" :key="chat.id" v-close-popup clickable @click="openChat(chat.id, { where: 'float' })">
            <q-item-section>{{ chat.title || "Untitled chat" }}</q-item-section>
          </q-item>
        </q-list>
      </q-btn-dropdown>
      <q-btn flat dense no-caps icon="playlist_add_check" label="Extract…" @click="showExtract = true" />
      <q-btn flat dense no-caps icon="refresh" label="Reprocess…" @click="showReprocess = true" />
      <q-btn flat dense round icon="more_vert" aria-label="More">
        <q-menu>
          <q-list dense style="min-width: 160px">
            <q-item v-close-popup clickable :disable="!detail?.transcript_path" @click="detail?.transcript_path && openFile(detail.transcript_path)">
              <q-item-section>Open note</q-item-section>
            </q-item>
            <q-item v-close-popup clickable @click="copyLink">
              <q-item-section>Copy link</q-item-section>
            </q-item>
          </q-list>
        </q-menu>
      </q-btn>
    </div>

    <q-banner v-if="banner > 0" dense class="pawn-banner warn">
      Reidentify {{ banner }} other session{{ banner === 1 ? "" : "s" }} with unknown speakers
      <template #action>
        <q-btn flat no-caps label="Reidentify" @click="reidentifyOthers" />
        <q-btn flat no-caps label="Dismiss" @click="banner = 0" />
      </template>
    </q-banner>

    <AudioPlayer
      ref="playerRef"
      :owner-id="id"
      :files="audioFiles"
      :duration="duration"
      :refresh="signedFiles"
      compact
      @time="time = $event"
    />
    <div v-if="audioError" class="audio-error text-caption q-px-md" role="alert">
      <q-icon name="error_outline" size="16px" /> Audio unavailable: {{ audioError }}
    </div>

    <div class="strip" :title="formatClock(time)" @click="onStrip">
      <div
        v-for="(seg, index) in flatSegments"
        :key="index"
        class="bit"
        :style="{
          left: duration ? `${(seg.start / duration) * 100}%` : '0',
          width: duration ? `${Math.max(((seg.end - seg.start) / duration) * 100, 0.3)}%` : '0',
          background: speakerColor(seg.label),
        }"
      />
      <div class="playhead" :style="{ left: duration ? `${(time / duration) * 100}%` : '0' }" />
    </div>

    <div class="row items-center q-px-sm border-b">
      <SegmentedControl
        v-model="section"
        size="sm"
        class="q-my-xs"
        aria-label="Session section"
        :options="[
          { value: 'transcript', label: 'Transcript', icon: 'subject' },
          { value: 'summary', label: 'Summary', icon: 'summarize' },
          { value: 'items', label: 'Items', icon: 'checklist', count: items.length },
        ]"
      />
      <q-space />
      <q-input
        v-if="section === 'transcript'"
        ref="findEl"
        v-model="find"
        dense
        borderless
        placeholder="Find"
        aria-label="Find in transcript"
        class="find"
        clearable
      />
      <q-btn
        flat
        dense
        no-caps
        size="sm"
        :icon="follow ? 'vertical_align_center' : 'pause'"
        :label="follow ? 'Follow' : 'Follow off'"
        @click="follow = !follow"
      />
      <q-btn v-if="narrow" flat dense no-caps size="sm" label="Speakers" @click="railOpen = !railOpen" />
    </div>

    <div class="col row no-wrap body">
      <q-tab-panels v-model="section" animated class="col" style="min-width: 0; background: transparent">
        <q-tab-panel name="transcript" class="q-pa-none column">
          <div v-if="speakerFilter" class="q-px-sm q-pt-xs">
            <button type="button" class="pawn-pill active" @click="speakerFilter = ''">
              {{ speakerLabelOf(speakerFilter) }}
              <q-icon name="close" class="x" />
            </button>
          </div>
          <div
            ref="transcriptEl"
            v-target="target"
            class="col pawn-scroll q-pa-sm"
            @scroll="onTranscriptScroll"
          >
            <div v-if="!transcript" class="text-muted">No transcript yet.</div>
            <div v-else-if="!visibleTurns.length" class="text-muted">No matching turns.</div>
            <div v-for="turn in visibleTurns" :key="turn.id" class="turn row no-wrap">
              <div class="meta column items-end">
                <button type="button" class="stamp" @click="seekPlay(turn.start)">{{ formatClock(turn.start) }}</button>
                <button type="button" class="chip-btn" @click="toggleSpeaker(turn.label)">
                  <SpeakerChip :name="nameOf(turn.label)" :color-key="turn.label" :size="18" />
                </button>
              </div>
              <div class="utter" data-pawn-anchor-root>
                <span
                  v-for="seg in turn.segments"
                  :key="seg.id"
                  class="tx-seg"
                  :data-seg="seg.id"
                  :data-t="seg.start"
                  :data-current="seg.id === currentId ? '1' : undefined"
                >{{ seg.text }} </span>
              </div>
            </div>
          </div>
        </q-tab-panel>
        <q-tab-panel name="summary" class="pawn-scroll">
          <div v-if="detail?.analysis" class="q-gutter-sm">
            <p>{{ detail.analysis.summary }}</p>
            <div v-if="detail.analysis.key_points.length">
              <div class="sub-label">Key points</div>
              <ul>
                <li v-for="point in detail.analysis.key_points" :key="point">{{ point }}</li>
              </ul>
            </div>
            <div class="row items-center q-gutter-xs">
              <span v-for="tag in detail.analysis.tags" :key="tag" class="tag-chip">#{{ tag }}</span>
              <span v-for="topic in detail.analysis.topics" :key="topic" class="tag-chip">{{ topic }}</span>
              <span v-if="detail.analysis.sentiment" class="text-caption text-muted">{{ detail.analysis.sentiment }}</span>
            </div>
          </div>
          <div v-else class="text-muted">No analysis yet.</div>
          <MarkdownView v-if="analysisMd" class="q-mt-md" :source="analysisMd" :from-key="detail?.analysis_path ?? ''" />
        </q-tab-panel>
        <q-tab-panel name="items" class="pawn-scroll">
          <div v-if="!items.length" class="text-muted">No open items from this session.</div>
          <ItemCard
            v-for="item in items"
            :key="item.id"
            :item="item"
            :checkbox="false"
            :source-link="false"
            :session-link="false"
            @action="(action, arg) => onItem(item, action, arg)"
          />
        </q-tab-panel>
      </q-tab-panels>
      <SpeakerRail
        v-show="!narrow || railOpen"
        ref="railRef"
        :session-id="id"
        :labels="detail?.labels ?? []"
        :gallery="detail?.gallery ?? []"
        :trained="!!detail?.speakers_trained"
        :focus="sessions.focusRail === id"
        :active-sample="activeSample"
        @refresh="reload()"
        @enrolled="onEnrolled"
        @play="playSamples"
      />
    </div>

    <SessionActionDialogs
      :session-id="id"
      :transcript-path="detail?.transcript_path"
      :extract="showExtract"
      :reprocess="showReprocess"
      @update:extract="showExtract = $event"
      @update:reprocess="showReprocess = $event"
      @done="reload()"
    />
  </div>
</template>

<style scoped>
.border-b {
  border-bottom: 1px solid var(--pawn-border);
}
.strip {
  position: relative;
  height: 28px;
  margin: 0 8px 6px;
  background: var(--pawn-track);
  border-radius: 4px;
  cursor: pointer;
  overflow: hidden;
}
.bit {
  position: absolute;
  top: 4px;
  bottom: 4px;
  border-radius: 2px;
  opacity: 0.85;
}
.playhead {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--q-primary);
}
.find {
  width: 160px;
}
.body {
  min-height: 0;
}
.turn {
  gap: 8px;
  margin-bottom: 10px;
}
.meta {
  width: 7.5rem;
  flex: none;
}
.stamp,
.chip-btn {
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  padding: 0;
}
.stamp {
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  color: var(--pawn-muted);
}
.stamp:hover {
  color: var(--q-primary);
}
.audio-error {
  color: var(--pawn-danger);
  padding-bottom: 4px;
}
.utter {
  flex: 1 1 auto;
  min-width: 0;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
@media (max-width: 900px) {
  .body {
    flex-direction: column;
  }
}
</style>

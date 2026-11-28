<script setup lang="ts">
// Session audio over signed URLs (the element requests Range itself).
// One load path: `load()` sets `src` once and waits for metadata, so a seek
// across a file boundary keeps its position. `playRange()` plays a span and
// pauses at its end (speaker samples). Expired links are re-signed through
// `refresh` and playback resumes where it was. Starting playback claims the
// global player and pauses the others.
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { fileAt, formatClock, globalTime, spansKnown, totalDuration, type AudioSpan } from "@/lib/audio";
import { usePlayerStore } from "@/stores/player";

export interface PlayableFile extends AudioSpan {
  url: string;
  /** Signature expiry (epoch seconds). */
  exp?: number;
  estimated?: boolean;
}

const props = defineProps<{
  ownerId: string;
  files: PlayableFile[];
  duration?: number;
  /** Re-sign the URLs (expired link or 403). */
  refresh?: () => Promise<PlayableFile[] | null>;
  /** Hide the slider when the parent draws its own scrubber. */
  compact?: boolean;
}>();

const emit = defineEmits<{ time: [seconds: number]; playing: [on: boolean] }>();

const player = usePlayerStore();
const audio = ref<HTMLAudioElement | null>(null);
const rate = ref(1);
const global = ref(0);
const playing = ref(false);
const problem = ref("");
const rates = [1, 1.25, 1.5, 2];

/** URLs can be replaced by a refresh without touching the span layout. */
const urls = ref<PlayableFile[]>([]);
let loadedIndex = -1;
let loadedUrl = "";
let stopAt: number | null = null;
let rangeDone: (() => void) | null = null;
let retried = false;

const length = computed(() => totalDuration(urls.value, props.duration ?? 0));
const known = computed(() => spansKnown(urls.value));
const estimated = computed(() => urls.value.some((file) => file.estimated));

function layout(files: PlayableFile[]): string {
  return files.map((file) => `${file.index}:${file.offset}:${file.duration}`).join("|");
}

watch(
  () => props.files,
  (next, prev) => {
    const sameLayout = prev && layout(next) === layout(prev);
    urls.value = next.map((file) => ({ ...file }));
    if (!sameLayout) {
      // A different recording (or a reprocess): start over.
      audio.value?.pause();
      loadedIndex = -1;
      loadedUrl = "";
      global.value = 0;
      problem.value = "";
    }
  },
  { immediate: true },
);

watch(
  () => player.owner,
  (owner) => {
    if (owner !== props.ownerId) audio.value?.pause();
  },
);

watch(rate, (value) => {
  if (audio.value) audio.value.playbackRate = value;
});

function fileByIndex(index: number): PlayableFile | null {
  return urls.value.find((file) => file.index === index) ?? null;
}

function expiresSoon(file: PlayableFile): boolean {
  return Boolean(file.exp && file.exp * 1000 - Date.now() < 60_000);
}

async function resign(): Promise<boolean> {
  if (!props.refresh) return false;
  const fresh = await props.refresh().catch(() => null);
  if (!fresh?.length) return false;
  urls.value = fresh.map((file) => ({ ...file }));
  return true;
}

function waitMetadata(el: HTMLAudioElement): Promise<boolean> {
  if (el.readyState >= 1) return Promise.resolve(true);
  return new Promise((resolve) => {
    const done = (ok: boolean) => {
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("error", onErr);
      resolve(ok);
    };
    const onMeta = () => done(true);
    const onErr = () => done(false);
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("error", onErr);
  });
}

/** Make *index* the loaded source. Returns false when it cannot be played. */
async function load(index: number): Promise<boolean> {
  const el = audio.value;
  let file = fileByIndex(index);
  if (!el || !file) return false;
  if (expiresSoon(file) && (await resign())) file = fileByIndex(index) ?? file;
  if (loadedIndex === index && loadedUrl === file.url && el.readyState >= 1) return true;
  loadedIndex = index;
  loadedUrl = file.url;
  el.src = file.url;
  el.playbackRate = rate.value;
  return waitMetadata(el);
}

function publish(index: number, local: number): void {
  global.value = globalTime(urls.value, index, local);
  emit("time", global.value);
}

async function seek(seconds: number): Promise<void> {
  const el = audio.value;
  if (!el || !urls.value.length) return;
  const hit = fileAt(urls.value, Math.min(Math.max(0, seconds), length.value || seconds));
  if (!(await load(hit.index))) return;
  const cap = Number.isFinite(el.duration) ? Math.max(0, el.duration - 0.05) : hit.local;
  el.currentTime = Math.min(Math.max(0, hit.local), cap);
  publish(hit.index, el.currentTime);
}

async function play(): Promise<void> {
  const el = audio.value;
  if (!el || !urls.value.length) return;
  if (loadedIndex < 0 && !(await load(urls.value[0].index))) return;
  player.claim(props.ownerId);
  try {
    await el.play();
  } catch {
    /* blocked by the browser, or the source failed (reported by onError) */
  }
}

function pause(): void {
  audio.value?.pause();
}

function toggle(): void {
  if (audio.value?.paused !== false) void play();
  else pause();
}

function finishRange(): void {
  stopAt = null;
  const done = rangeDone;
  rangeDone = null;
  done?.();
}

/** Play [start, end) and pause at the end. Resolves when it stops for any reason. */
async function playRange(start: number, end: number): Promise<void> {
  finishRange();
  await seek(start);
  stopAt = end;
  const done = new Promise<void>((resolve) => {
    rangeDone = resolve;
  });
  await play();
  if (audio.value?.paused) finishRange();
  return done;
}

function stopRange(): void {
  if (stopAt == null) return;
  pause();
  finishRange();
}

function skip(delta: number): void {
  void seek(Math.min(length.value, Math.max(0, global.value + delta)));
}

function onTime(): void {
  const el = audio.value;
  if (!el || loadedIndex < 0) return;
  publish(loadedIndex, el.currentTime);
  if (stopAt != null && global.value >= stopAt) stopRange();
}

function onPlay(): void {
  playing.value = true;
  problem.value = "";
  retried = false;
  player.claim(props.ownerId);
  emit("playing", true);
}

function onPause(): void {
  playing.value = false;
  if (player.owner === props.ownerId) player.release(props.ownerId);
  emit("playing", false);
  if (stopAt != null) finishRange();
}

async function onEnded(): Promise<void> {
  const pos = urls.value.findIndex((file) => file.index === loadedIndex);
  const next = urls.value[pos + 1];
  if (!next || !known.value) {
    finishRange();
    return;
  }
  const keepRange = stopAt;
  await seek(next.offset);
  stopAt = keepRange;
  await play();
}

async function httpStatus(url: string): Promise<{ status: number; detail: string }> {
  try {
    const resp = await fetch(url, { headers: { Range: "bytes=0-0" } });
    let detail = "";
    if (!resp.ok) {
      try {
        detail = String((await resp.json()).detail ?? "");
      } catch {
        detail = resp.statusText;
      }
    }
    return { status: resp.status, detail };
  } catch {
    return { status: 0, detail: "" };
  }
}

async function onError(): Promise<void> {
  const el = audio.value;
  if (!el || !loadedUrl) return;
  const resume = global.value;
  const wasPlaying = playing.value || stopAt != null;
  const { status, detail } = await httpStatus(loadedUrl);
  if ((status === 403 || status === 401) && !retried && (await resign())) {
    retried = true;
    loadedIndex = -1;
    await seek(resume);
    if (wasPlaying) await play();
    return;
  }
  finishRange();
  if (status === 415) problem.value = detail || "Unsupported audio format";
  else if (status === 404) problem.value = "Audio file missing";
  else if (status === 403 || status === 401) problem.value = "Audio link expired. Reload the session.";
  else if (el.error?.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED) problem.value = "This browser cannot play this audio format";
  else problem.value = "Audio could not be loaded";
}

onBeforeUnmount(() => {
  finishRange();
  player.release(props.ownerId);
});

defineExpose({ seek, play, pause, toggle, playRange, stopRange });
</script>

<template>
  <div class="player row items-center no-wrap q-gutter-x-xs">
    <audio
      ref="audio"
      preload="metadata"
      @timeupdate="onTime"
      @play="onPlay"
      @pause="onPause"
      @ended="onEnded"
      @error="onError"
    />
    <q-btn flat dense round icon="replay_10" aria-label="Back 10 seconds" :disable="!urls.length" @click="skip(-10)" />
    <q-btn
      flat
      dense
      round
      :icon="playing ? 'pause' : 'play_arrow'"
      aria-label="Play or pause"
      :disable="!urls.length"
      @click="toggle"
    />
    <q-btn flat dense round icon="forward_10" aria-label="Forward 10 seconds" :disable="!urls.length" @click="skip(10)" />
    <q-slider
      v-if="!compact"
      class="col"
      :min="0"
      :max="Math.max(length, 0.01)"
      :step="0.1"
      :model-value="Math.min(global, length)"
      color="primary"
      @change="(value: number) => seek(value)"
    />
    <q-space v-else />
    <span v-if="problem" class="problem text-caption" role="alert">
      <q-icon name="error_outline" size="16px" /> {{ problem }}
    </span>
    <span v-else-if="!urls.length" class="text-caption text-muted">No audio</span>
    <span v-else-if="!known || estimated" class="text-caption text-muted" :title="known ? 'File offsets estimated from segment times' : 'File durations unknown; seeking stays in the first file'">
      ≈ timing
    </span>
    <span class="clock text-caption text-muted">{{ formatClock(global) }} / {{ formatClock(length) }}</span>
    <div class="rates row no-wrap" role="group" aria-label="Playback speed">
      <button
        v-for="value in rates"
        :key="value"
        type="button"
        class="rate"
        :class="{ active: rate === value }"
        :aria-pressed="rate === value"
        @click="rate = value"
      >
        {{ value }}×
      </button>
    </div>
  </div>
</template>

<style scoped>
.player {
  padding: 4px 8px;
}
.clock {
  min-width: 7.5rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.problem {
  color: var(--pawn-danger);
  white-space: nowrap;
}
.rates {
  border: 1px solid var(--pawn-border);
  border-radius: 4px;
  overflow: hidden;
}
.rate {
  border: 0;
  background: transparent;
  color: var(--pawn-muted);
  font: 500 11px/1 var(--pawn-mono);
  padding: 4px 6px;
  cursor: pointer;
}
.rate + .rate {
  border-left: 1px solid var(--pawn-border);
}
.rate:hover {
  background: var(--pawn-hover);
}
.rate.active {
  background: var(--pawn-active);
  color: var(--q-primary);
}
</style>

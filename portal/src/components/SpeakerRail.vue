<script setup lang="ts">
// Speakers rail for one session: name each diarization label, enroll into the
// gallery, or relabel this session only.
import { computed, ref } from "vue";
import { api, type GallerySpeaker, type SessionLabel, type SessionSample } from "@/api";
import { notifyError } from "@/lib/errors";
import { formatTalk } from "@/lib/audio";
import { speakerColor, speakerLabel } from "@/lib/speakers";
import { openSpeakers } from "@/router";
import SpeakerChip from "./SpeakerChip.vue";

const props = defineProps<{
  sessionId: string;
  labels: SessionLabel[];
  gallery: GallerySpeaker[];
  trained: boolean;
  focus?: boolean;
  /** Start of the sample set now playing (its button shows stop). */
  activeSample?: number | null;
}>();

const emit = defineEmits<{ refresh: []; enrolled: []; play: [samples: SessionSample[]] }>();

function sampleIcon(label: SessionLabel): string {
  const first = label.samples?.[0]?.start;
  return first != null && first === props.activeSample ? "stop" : "play_arrow";
}

const names = ref<Record<string, string>>({});
const confirming = ref("");
const busy = ref("");
const filters = ref<Record<string, string>>({});
const root = ref<HTMLElement | null>(null);

function galleryName(id: string | null | undefined): string {
  if (!id) return "";
  return props.gallery.find((speaker) => speaker.id === id)?.display_name ?? "";
}

function draft(label: SessionLabel): string {
  if (names.value[label.local_label] !== undefined) return names.value[label.local_label];
  return label.mapped_name || galleryName(label.suggested_speaker_id) || "";
}

function setDraft(label: SessionLabel, value: string): void {
  const created = /^Create '(.+)'$/.exec(value);
  names.value = { ...names.value, [label.local_label]: created ? created[1] : value };
}

function suggested(label: SessionLabel): boolean {
  const name = galleryName(label.suggested_speaker_id);
  return !label.mapped_speaker_id && !!name && draft(label) === name && label.suggested_score != null;
}

function optionsFor(label: SessionLabel): string[] {
  const q = (filters.value[label.local_label] ?? "").trim().toLowerCase();
  const all = props.gallery.map((speaker) => speaker.display_name);
  const matched = q ? all.filter((name) => name.toLowerCase().includes(q)) : all;
  const typed = (filters.value[label.local_label] ?? "").trim();
  if (typed && !all.some((name) => name.toLowerCase() === typed.toLowerCase())) matched.push(`Create '${typed}'`);
  return matched;
}

function talkOf(label: SessionLabel): string {
  return formatTalk(label.talk_seconds ?? label.duration_s);
}

async function enroll(label: SessionLabel): Promise<void> {
  const speaker = draft(label).trim();
  if (!speaker) return;
  busy.value = label.local_label;
  try {
    await api().enrollSession(props.sessionId, { speaker, from_label: label.local_label });
    confirming.value = "";
    emit("enrolled");
  } catch (e) {
    notifyError(e, "Enroll failed");
  } finally {
    busy.value = "";
  }
}

async function relabel(label: SessionLabel): Promise<void> {
  const to = draft(label).trim();
  if (!to) return;
  busy.value = label.local_label;
  try {
    await api().relabelSession(props.sessionId, { from: label.local_label, to });
    emit("refresh");
  } catch (e) {
    notifyError(e, "Relabel failed");
  } finally {
    busy.value = "";
  }
}

async function ignore(label: SessionLabel): Promise<void> {
  busy.value = label.local_label;
  try {
    await api().ignoreSessionSpeaker(props.sessionId, { from: label.local_label, ignore: !label.ignored });
    emit("refresh");
  } catch (e) {
    notifyError(e, "Could not update the speaker");
  } finally {
    busy.value = "";
  }
}

async function reidentify(): Promise<void> {
  busy.value = "*";
  try {
    await api().reidentifySession(props.sessionId);
    emit("refresh");
  } catch (e) {
    notifyError(e, "Reidentify failed");
  } finally {
    busy.value = "";
  }
}

async function markTrained(): Promise<void> {
  busy.value = "*";
  try {
    await api().setSessionTrained(props.sessionId, !props.trained);
    emit("refresh");
  } catch (e) {
    notifyError(e, "Could not update trained");
  } finally {
    busy.value = "";
  }
}

const galleryId = computed(() => {
  const mapped = props.labels.find((label) => label.mapped_speaker_id)?.mapped_speaker_id;
  return mapped || props.labels.find((label) => label.suggested_speaker_id)?.suggested_speaker_id || "";
});

defineExpose({
  focusFirst() {
    root.value?.querySelector<HTMLElement>("input")?.focus();
  },
});
</script>

<template>
  <aside ref="root" class="rail pawn-rail column no-wrap" :class="{ focus }">
    <div class="section-label">Speakers</div>
    <div class="col pawn-scroll q-px-sm">
      <div v-for="label in labels" :key="label.local_label" class="label" :class="{ ignored: label.ignored }">
        <div class="row items-center no-wrap q-gutter-x-xs">
          <span class="swatch" :style="{ background: speakerColor(label.local_label) }" />
          <button
            v-if="label.mapped_speaker_id"
            type="button"
            class="name-link"
            :aria-label="`Open ${label.mapped_name || speakerLabel(label.local_label)} in the gallery`"
            @click="openSpeakers(label.mapped_speaker_id)"
          >
            <SpeakerChip :name="label.mapped_name || label.local_label" :color-key="label.local_label" :size="18" />
          </button>
          <SpeakerChip v-else :name="label.local_label" :color-key="label.local_label" :size="18" />
          <span class="text-caption text-muted">{{ talkOf(label) }}</span>
          <q-space />
          <q-btn
            v-if="label.samples?.length"
            flat
            dense
            round
            size="sm"
            :icon="sampleIcon(label)"
            :aria-label="sampleIcon(label) === 'stop' ? 'Stop sample' : 'Play samples'"
            @click="emit('play', label.samples)"
          />
        </div>
        <div class="share q-mt-xs">
          <div class="share-fill" :style="{ width: `${Math.round((label.share ?? 0) * 100)}%`, background: speakerColor(label.local_label) }" />
        </div>
        <q-select
          class="q-mt-xs"
          dense
          outlined
          use-input
          fill-input
          hide-selected
          input-debounce="0"
          :model-value="draft(label)"
          :options="optionsFor(label)"
          label="Name"
          @filter="(val: string, update: (fn: () => void) => void) => update(() => (filters[label.local_label] = val))"
          @update:model-value="(value: string | null) => setDraft(label, value ?? '')"
        />
        <div v-if="suggested(label)" class="suggest text-caption text-muted">
          suggested {{ label.suggested_score?.toFixed(2) }}
        </div>
        <div v-if="confirming === label.local_label" class="confirm q-mt-xs">
          <div class="confirm-text">
            Enroll {{ talkOf(label) }} of {{ speakerLabel(label.local_label) }} as <b>{{ draft(label) }}</b>?
          </div>
          <div class="actions row items-center no-wrap q-mt-xs">
            <q-btn
              dense
              no-caps
              outline
              size="sm"
              color="primary"
              icon="how_to_reg"
              label="Confirm"
              :loading="busy === label.local_label"
              @click="enroll(label)"
            />
            <q-btn dense flat no-caps size="sm" label="Cancel" @click="confirming = ''" />
          </div>
        </div>
        <div v-else class="actions row items-center q-mt-xs">
          <q-btn
            dense
            no-caps
            outline
            size="sm"
            icon="how_to_reg"
            label="Enroll"
            :disable="!draft(label).trim() || label.ignored"
            @click="confirming = label.local_label"
          >
            <q-tooltip>Add this voice to the speakers gallery</q-tooltip>
          </q-btn>
          <q-btn dense flat no-caps size="sm" icon="label" label="Session only" :disable="!draft(label).trim()" @click="relabel(label)">
            <q-tooltip>Rename this label in this session only</q-tooltip>
          </q-btn>
          <q-space />
          <q-btn
            dense
            flat
            round
            size="sm"
            :icon="label.ignored ? 'visibility' : 'visibility_off'"
            :aria-label="label.ignored ? 'Unignore' : 'Ignore'"
            :loading="busy === label.local_label && confirming !== label.local_label"
            @click="ignore(label)"
          >
            <q-tooltip>{{ label.ignored ? "Unignore" : "Ignore this speaker" }}</q-tooltip>
          </q-btn>
          <q-btn
            v-if="label.mapped_speaker_id"
            dense
            flat
            round
            size="sm"
            icon="open_in_new"
            aria-label="Open in gallery"
            @click="openSpeakers(label.mapped_speaker_id)"
          >
            <q-tooltip>Open in gallery</q-tooltip>
          </q-btn>
        </div>
      </div>
      <div v-if="!labels.length" class="text-caption text-muted q-pa-sm">No speaker labels.</div>
    </div>
    <div class="actions footer row items-center no-wrap q-px-sm q-py-xs border-t">
      <q-btn dense flat no-caps size="sm" icon="refresh" label="Reidentify" :loading="busy === '*'" @click="reidentify" />
      <q-space />
      <q-btn
        dense
        no-caps
        size="sm"
        :outline="!trained"
        :flat="trained"
        :icon="trained ? 'undo' : 'check'"
        :label="trained ? 'Trained · undo' : 'Mark trained'"
        @click="markTrained"
      />
      <q-btn v-if="galleryId" dense flat round size="sm" icon="record_voice_over" aria-label="Open speakers" @click="openSpeakers(galleryId)">
        <q-tooltip>Speakers</q-tooltip>
      </q-btn>
    </div>
  </aside>
</template>

<style scoped>
.rail {
  width: 300px;
  border-left: 1px solid var(--pawn-border);
  min-height: 0;
}
.rail.focus {
  box-shadow: inset 2px 0 0 var(--q-primary);
}
.swatch {
  width: 8px;
  height: 18px;
  border-radius: 2px;
}
.share {
  height: 4px;
  background: var(--pawn-border);
  border-radius: 2px;
}
.share-fill {
  height: 100%;
  border-radius: 2px;
}
.label {
  padding: 8px 0 10px;
  border-bottom: 1px solid var(--pawn-border);
}
.label:last-child {
  border-bottom: 0;
}
.label.ignored {
  opacity: 0.55;
}
.actions {
  gap: 4px;
}
.confirm {
  font-size: 12px;
  padding: 6px 8px;
  border-left: 2px solid var(--q-primary);
  background: color-mix(in srgb, var(--q-primary) 8%, transparent);
}
.suggest {
  font-variant-numeric: tabular-nums;
}
.name-link {
  border: 0;
  padding: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
}
.border-t {
  border-top: 1px solid var(--pawn-border);
}
@media (max-width: 900px) {
  .rail {
    width: 100%;
    border-left: 0;
    border-top: 1px solid var(--pawn-border);
    max-height: 46%;
  }
}
</style>

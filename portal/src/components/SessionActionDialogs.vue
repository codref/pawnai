<script setup lang="ts">
// Extract and reprocess dialogs shared by the session list and the session tab.
import { ref, watch } from "vue";
import { Notify } from "quasar";
import { api } from "@/api";
import { notifyError } from "@/lib/errors";
import { openFile } from "@/router";
import { useJobsStore } from "@/stores/jobs";

const props = defineProps<{
  sessionId: string;
  transcriptPath?: string | null;
  extract: boolean;
  reprocess: boolean;
}>();

const emit = defineEmits<{
  "update:extract": [open: boolean];
  "update:reprocess": [open: boolean];
  done: [];
}>();

const jobs = useJobsStore();
const busy = ref(false);
const doItems = ref(true);
const doPeople = ref(false);
const mode = ref<"rediarize" | "retranscribe">("rediarize");
const backend = ref("nemo");
const language = ref("");
const threshold = ref(0.7);
const crossFile = ref(0.55);
const margin = ref("");

watch(
  () => props.extract,
  (open) => {
    if (!open) return;
    doItems.value = true;
    doPeople.value = false;
  },
);

watch(
  () => props.reprocess,
  (open) => {
    if (!open) return;
    mode.value = "rediarize";
    backend.value = "nemo";
    language.value = "";
    threshold.value = 0.7;
    crossFile.value = 0.55;
    margin.value = "";
  },
);

function follow(jobId: unknown): void {
  if (typeof jobId === "string" && jobId) void jobs.watchJob(jobId);
}

async function runExtract(): Promise<void> {
  if (busy.value || (!doItems.value && !doPeople.value)) return;
  busy.value = true;
  try {
    const result = await api().extractSession(props.sessionId, { items: doItems.value, people: doPeople.value });
    const count = result.items?.items;
    Notify.create({
      message: typeof count === "number" ? `Extracted ${count} item(s).` : "Extract finished.",
      position: "bottom-right",
    });
    emit("update:extract", false);
    emit("done");
  } catch (e) {
    notifyError(e, "Extract failed");
  } finally {
    busy.value = false;
  }
}

async function runReprocess(): Promise<void> {
  if (busy.value) return;
  const wipe = mode.value === "retranscribe" ? "\n\nThis will wipe and rebuild the transcript text." : "";
  if (!window.confirm(`Queue ${mode.value} for ${props.sessionId}?${wipe}`)) return;
  busy.value = true;
  try {
    const body: {
      confirm: true;
      mode: "rediarize" | "retranscribe";
      threshold: number;
      cross_file_threshold: number;
      backend?: string;
      language?: string;
      margin?: number;
    } = {
      confirm: true,
      mode: mode.value,
      threshold: threshold.value,
      cross_file_threshold: crossFile.value,
    };
    if (mode.value === "retranscribe") {
      body.backend = backend.value;
      if (backend.value === "whisper" && language.value.trim()) body.language = language.value.trim();
    }
    if (mode.value === "rediarize" && margin.value.trim()) {
      const parsed = Number(margin.value);
      if (!Number.isNaN(parsed)) body.margin = parsed;
    }
    const result = await api().reprocessSession(props.sessionId, body);
    follow(result.job_id);
    Notify.create({ message: `Queued ${mode.value}.`, position: "bottom-right" });
    emit("update:reprocess", false);
    emit("done");
  } catch (e) {
    notifyError(e, "Reprocess failed");
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <q-dialog :model-value="extract" @update:model-value="emit('update:extract', $event)">
    <q-card style="width: 460px; max-width: 92vw">
      <q-card-section class="text-subtitle1">Extract · {{ sessionId }}</q-card-section>
      <q-card-section class="q-pt-none text-body2">
        Re-runs the same analysis that runs at the end of a recording. Edit ## Annotations on the
        transcript first to add people, relations, or what to ignore — those notes are included in
        the extract. Untriaged items for this session are replaced; already-triaged fingerprints stay
        suppressed.
      </q-card-section>
      <q-card-section v-if="transcriptPath" class="q-pt-none row items-center">
        <div class="col">
          <div class="sub-label">Transcript</div>
          <div class="ellipsis">{{ transcriptPath }}</div>
        </div>
        <q-btn flat no-caps label="Open" @click="openFile(transcriptPath)" />
      </q-card-section>
      <q-card-section class="q-pt-none">
        <q-toggle v-model="doItems" label="Items" />
        <div class="text-caption text-muted q-ml-lg">Extract decisions, commitments, questions → Pawn/Items</div>
        <q-toggle v-model="doPeople" label="People refresh" class="q-mt-sm" />
        <div class="text-caption text-muted q-ml-lg">Update People/ bios for gallery speakers in this session</div>
      </q-card-section>
      <q-card-actions align="right">
        <q-btn flat no-caps label="Cancel" :disable="busy" @click="emit('update:extract', false)" />
        <q-btn
          color="primary"
          no-caps
          label="Extract"
          :loading="busy"
          :disable="!doItems && !doPeople"
          @click="runExtract"
        />
      </q-card-actions>
    </q-card>
  </q-dialog>

  <q-dialog :model-value="reprocess" @update:model-value="emit('update:reprocess', $event)">
    <q-card style="width: 460px; max-width: 92vw">
      <q-card-section class="text-subtitle1">Reprocess · {{ sessionId }}</q-card-section>
      <q-card-section class="q-pt-none column q-gutter-sm">
        <q-btn-toggle
          v-model="mode"
          no-caps
          spread
          toggle-color="primary"
          :options="[
            { label: 'Rediarize', value: 'rediarize' },
            { label: 'Retranscribe', value: 'retranscribe' },
          ]"
        />
        <template v-if="mode === 'retranscribe'">
          <q-select
            v-model="backend"
            dense
            outlined
            label="ASR"
            :options="[
              { label: 'NeMo', value: 'nemo' },
              { label: 'Whisper', value: 'whisper' },
            ]"
            emit-value
            map-options
          />
          <q-input
            v-if="backend === 'whisper'"
            v-model="language"
            dense
            outlined
            label="Language"
            hint="Optional Whisper language code"
          />
        </template>
        <q-input v-model.number="threshold" dense outlined type="number" step="0.05" label="Identify threshold" />
        <q-input
          v-model.number="crossFile"
          dense
          outlined
          type="number"
          step="0.05"
          label="Cross-file threshold"
        />
        <q-input
          v-if="mode === 'rediarize'"
          v-model="margin"
          dense
          outlined
          label="Identify margin"
          hint="Optional. Empty uses the server default."
        />
      </q-card-section>
      <q-card-actions align="right">
        <q-btn flat no-caps label="Cancel" :disable="busy" @click="emit('update:reprocess', false)" />
        <q-btn color="primary" no-caps :label="mode === 'rediarize' ? 'Queue rediarize' : 'Queue retranscribe'" :loading="busy" @click="runReprocess" />
      </q-card-actions>
    </q-card>
  </q-dialog>
</template>

<script setup lang="ts">
// New board: a blank corkboard, a lens (saved query) or a question the night
// shift works on. Opens the board when it is made.
import { ref } from "vue";
import type { BoardCreate } from "@/api";
import { useBoardsStore } from "@/stores/boards";
import { openBrainmap } from "@/router";

const props = defineProps<{ folderId?: string | null; tags?: string[] }>();
const open = defineModel<boolean>({ default: false });
const boards = useBoardsStore();

type Kind = "manual" | "lens" | "question";
const kind = ref<Kind>("manual");
const title = ref("");
const query = ref("");
const people = ref("");
const question = ref("");
const busy = ref(false);

const KINDS: { value: Kind; label: string; hint: string; icon: string }[] = [
  { value: "manual", label: "Blank board", hint: "Pin cards from other boards, or ask the board chat to propose some.", icon: "dashboard" },
  { value: "lens", label: "Lens", hint: "A saved query that keeps itself up to date: threads, items and notes near it.", icon: "filter_alt" },
  { value: "question", label: "Question", hint: "Pawn works on it overnight: it finds evidence, dreams links, and tells you in the morning.", icon: "bedtime" },
];

function canSubmit(): boolean {
  if (kind.value === "question") return !!question.value.trim();
  if (kind.value === "lens") return !!(title.value.trim() && (query.value.trim() || people.value.trim()));
  return !!title.value.trim();
}

async function submit(): Promise<void> {
  if (!canSubmit() || busy.value) return;
  busy.value = true;
  const body: BoardCreate = {
    title: title.value.trim() || question.value.trim(),
    kind: kind.value,
    folder_id: props.folderId ?? null,
    tags: props.tags ?? [],
  };
  if (kind.value === "lens") {
    body.spec = {
      q: query.value.trim() || undefined,
      people: people.value
        .split(",")
        .map((p) => p.trim())
        .filter(Boolean),
    };
  }
  if (kind.value === "question") body.question = question.value.trim();
  const b = await boards.create(body);
  busy.value = false;
  if (b) {
    open.value = false;
    title.value = query.value = people.value = question.value = "";
    openBrainmap(b.name);
  }
}
</script>

<template>
  <q-dialog v-model="open">
    <q-card style="width: 460px; max-width: 92vw">
      <q-card-section class="text-h6">New board</q-card-section>
      <q-card-section class="q-pt-none">
        <q-list dense>
          <q-item v-for="k in KINDS" :key="k.value" v-ripple tag="label" class="rounded-borders">
            <q-item-section side><q-radio v-model="kind" :val="k.value" /></q-item-section>
            <q-item-section avatar><q-icon :name="k.icon" /></q-item-section>
            <q-item-section>
              <q-item-label>{{ k.label }}</q-item-label>
              <q-item-label caption>{{ k.hint }}</q-item-label>
            </q-item-section>
          </q-item>
        </q-list>
        <div class="q-mt-md column q-gutter-sm">
          <q-input
            v-if="kind === 'question'"
            v-model="question"
            outlined
            dense
            autofocus
            autogrow
            label="The question"
            placeholder="Why does team B keep slipping?"
            @keydown.enter.exact.prevent="submit"
          />
          <q-input
            v-model="title"
            outlined
            dense
            :autofocus="kind !== 'question'"
            :label="kind === 'question' ? 'Board name (optional)' : 'Board name'"
            @keydown.enter="submit"
          />
          <template v-if="kind === 'lens'">
            <q-input v-model="query" outlined dense label="Look for" placeholder="vendor contract renewal" @keydown.enter="submit" />
            <q-input v-model="people" outlined dense label="People (comma separated, optional)" @keydown.enter="submit" />
          </template>
        </div>
      </q-card-section>
      <q-card-actions align="right">
        <q-btn flat no-caps label="Cancel" @click="open = false" />
        <q-btn unelevated no-caps color="primary" label="Create" :loading="busy" :disable="!canSubmit()" @click="submit" />
      </q-card-actions>
    </q-card>
  </q-dialog>
</template>

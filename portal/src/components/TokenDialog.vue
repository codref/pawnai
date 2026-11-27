<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{ submit: (token: string) => Promise<string | null> }>();

const token = ref("");
const error = ref("");
const busy = ref(false);

async function save(): Promise<void> {
  if (!token.value.trim()) return;
  busy.value = true;
  error.value = (await props.submit(token.value.trim())) ?? "";
  busy.value = false;
}
</script>

<template>
  <q-dialog :model-value="true" persistent>
    <q-card style="width: 420px; max-width: 92vw">
      <q-card-section>
        <div class="text-h6">Connect to Pawn</div>
        <div class="text-body2 text-muted q-mt-xs">
          Enter the pawn-server API token (<code>api.token</code> in pawnai.yaml). It is kept in this
          browser's local storage.
        </div>
      </q-card-section>
      <q-card-section class="q-pt-none">
        <q-input
          v-model="token"
          type="password"
          label="API token"
          autofocus
          outlined
          dense
          :error="Boolean(error)"
          :error-message="error"
          @keyup.enter="save"
        />
      </q-card-section>
      <q-card-actions align="right">
        <q-btn color="primary" label="Connect" :loading="busy" :disable="!token.trim()" @click="save" />
      </q-card-actions>
    </q-card>
  </q-dialog>
</template>

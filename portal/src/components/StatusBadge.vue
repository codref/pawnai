<script setup lang="ts">
// A small status pill that follows the skin: ok / warn / danger / info /
// neutral. Use it instead of `<q-badge color="orange">` so colors come from
// the theme tokens (`--pawn-ok`, `--pawn-warn`, …).
export type StatusKind = "ok" | "warn" | "danger" | "info" | "neutral";

withDefaults(defineProps<{ kind?: StatusKind; label: string; busy?: boolean; icon?: string }>(), {
  kind: "neutral",
  busy: false,
  icon: "",
});
</script>

<template>
  <span class="pawn-status" :class="[kind, { busy }]">
    <q-icon v-if="icon" :name="icon" size="12px" />
    {{ label }}
  </span>
</template>

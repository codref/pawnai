<script setup lang="ts">
// Speaker avatar + name. `colorKey` (the session label) keeps one color per
// speaker across the strip, the rail and the transcript, even after mapping.
import { computed } from "vue";
import { speakerColor, speakerInitials, speakerLabel } from "@/lib/speakers";

const props = withDefaults(defineProps<{ name: string; size?: number; colorKey?: string }>(), {
  size: 22,
  colorKey: "",
});

const label = computed(() => speakerLabel(props.name));
const initials = computed(() => speakerInitials(label.value));
const color = computed(() => speakerColor(props.colorKey || props.name));
</script>

<template>
  <span class="chip row no-wrap items-center" :title="name">
    <span
      class="spk-avatar row flex-center"
      :style="{
        '--spk': color,
        background: color,
        width: `${size}px`,
        height: `${size}px`,
        fontSize: `${Math.max(10, size * 0.42)}px`,
      }"
    >
      {{ initials }}
    </span>
    <span class="q-ml-xs">{{ label }}</span>
  </span>
</template>

<style scoped>
.spk-avatar {
  border-radius: 50%;
  color: var(--pawn-spk-ink);
  font-weight: 600;
  line-height: 1;
  flex: 0 0 auto;
}
</style>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { Notify } from "quasar";
import { renderMarkdown } from "@/lib/markdown";
import { useFilesStore } from "@/stores/files";
import { openFile } from "@/router";

const props = defineProps<{ source: string; fromKey?: string }>();
const files = useFilesStore();
const root = ref<HTMLElement | null>(null);
const html = computed(() => renderMarkdown(props.source));

async function onClick(ev: MouseEvent): Promise<void> {
  const a = (ev.target as HTMLElement).closest("a.wikilink") as HTMLAnchorElement | null;
  if (!a) return;
  ev.preventDefault();
  const target = a.dataset.wikilink ?? "";
  const key = await files.resolveWikilink(target, props.fromKey ?? "");
  if (key) {
    files.markSeen(key);
    openFile(key);
  } else {
    Notify.create({ message: `No file found for [[${target}]]`, position: "bottom-right" });
  }
}

/** Dim links whose target doesn't exist in the vault. */
async function markMissing(): Promise<void> {
  await nextTick();
  const links = Array.from(root.value?.querySelectorAll<HTMLAnchorElement>("a.wikilink") ?? []);
  for (const a of links) {
    const key = await files.resolveWikilink(a.dataset.wikilink ?? "", props.fromKey ?? "").catch(() => null);
    a.classList.toggle("wikilink-missing", !key);
    if (key) a.title = key;
  }
}

onMounted(markMissing);
watch(html, markMissing);
</script>

<template>
  <!-- The rendered body is what annotation anchors are measured on. -->
  <!-- eslint-disable-next-line vue/no-v-html -->
  <div ref="root" class="markdown" data-pawn-anchor-root @click="onClick" v-html="html" />
</template>

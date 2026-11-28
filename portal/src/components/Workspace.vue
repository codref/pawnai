<script setup lang="ts">
// Center area: the docked split tree, floating windows above it, and every
// open tab mounted exactly once. Each panel is teleported into its group's
// body; changing a tab's group moves the DOM without remounting, so a move
// between splits or into a float keeps scroll, drafts and live streams.
//
// Targets are the group body *elements* (lib/groupSlots.ts), so a group view
// that remounts still gets its panels moved into the new element.
import { defineAsyncComponent } from "vue";
import { groupSlots } from "@/lib/groupSlots";
import { useTabsStore, type Tab } from "@/stores/tabs";
import SplitView from "./SplitView.vue";
import FloatWindow from "./FloatWindow.vue";

const ChatPanel = defineAsyncComponent(() => import("@/panels/ChatPanel.vue"));
const FilePanel = defineAsyncComponent(() => import("@/panels/FilePanel.vue"));
const ClipPanel = defineAsyncComponent(() => import("@/panels/ClipPanel.vue"));
const InboxPanel = defineAsyncComponent(() => import("@/panels/InboxPanel.vue"));
const SettingsPanel = defineAsyncComponent(() => import("@/panels/SettingsPanel.vue"));
const SessionsPanel = defineAsyncComponent(() => import("@/panels/SessionsPanel.vue"));
const SessionPanel = defineAsyncComponent(() => import("@/panels/SessionPanel.vue"));
const DayPanel = defineAsyncComponent(() => import("@/panels/DayPanel.vue"));
const SpeakersPanel = defineAsyncComponent(() => import("@/panels/SpeakersPanel.vue"));
const AssistantsPanel = defineAsyncComponent(() => import("@/panels/AssistantsPanel.vue"));
const BrainmapPanel = defineAsyncComponent(() => import("@/panels/BrainmapPanel.vue"));

const tabs = useTabsStore();

function shown(t: Tab): boolean {
  return tabs.group(t.groupId)?.activeId === t.id;
}
</script>

<template>
  <div class="workspace relative-position full-height">
    <SplitView :node="tabs.root" />
    <div class="float-layer">
      <FloatWindow v-for="g in tabs.floats" :key="g.id" :group-id="g.id" />
    </div>
    <!-- Panels whose group body isn't mounted yet wait here (hidden). -->
    <div id="panel-park" class="panel-park" />
    <Teleport v-for="t in tabs.tabs" :key="t.id" defer :to="groupSlots.get(t.groupId) ?? '#panel-park'">
      <div v-show="shown(t)" class="panel-slot">
        <ChatPanel v-if="t.kind === 'chat'" :id="t.ref" :tab-id="t.id" />
        <ClipPanel v-else-if="t.kind === 'clip'" :id="t.ref" :tab-id="t.id" />
        <InboxPanel v-else-if="t.kind === 'inbox'" :tab-id="t.id" />
        <SettingsPanel v-else-if="t.kind === 'settings'" :tab-id="t.id" />
        <SessionsPanel v-else-if="t.kind === 'sessions'" :tab-id="t.id" />
        <SessionPanel v-else-if="t.kind === 'session'" :id="t.ref" :tab-id="t.id" />
        <DayPanel v-else-if="t.kind === 'day'" :tab-id="t.id" />
        <SpeakersPanel v-else-if="t.kind === 'speakers'" :tab-id="t.id" />
        <AssistantsPanel v-else-if="t.kind === 'assistants'" :tab-id="t.id" />
        <BrainmapPanel v-else-if="t.kind === 'brainmap'" :board="t.ref" :tab-id="t.id" :active="shown(t)" />
        <FilePanel v-else :file-key="t.ref" :tab-id="t.id" :edit="t.edit" />
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.workspace {
  overflow: hidden;
}
.float-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 20;
}
.panel-park {
  display: none;
}
.panel-slot {
  position: absolute;
  inset: 0;
}
</style>

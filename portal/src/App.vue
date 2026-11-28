<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { api, MOCK_ENABLED } from "@/api";
import { useAuthStore } from "@/stores/auth";
import { useChatsStore } from "@/stores/chats";
import { useFilesStore } from "@/stores/files";
import { useInboxStore } from "@/stores/inbox";
import { useJobsStore } from "@/stores/jobs";
import { useSessionsStore } from "@/stores/sessions";
import { openAssistants, openBrainmap, openDay, openInbox, openSessions, openSettings, openSpeakers } from "@/router";
import { useTabsStore } from "@/stores/tabs";
import { loadString, saveString } from "@/lib/storage";
import { applyAppearance, cycleMode, mode, skin, skinLabel } from "@/lib/theme";
import { pageStyle } from "@/lib/layout";
import { ICONS } from "@/lib/icons";
import ChatSidebar from "@/components/ChatSidebar.vue";
import ContextMenu from "@/components/ContextMenu.vue";
import FilesSidebar from "@/components/FilesSidebar.vue";
import ResizeHandle from "@/components/ResizeHandle.vue";
import TokenDialog from "@/components/TokenDialog.vue";
import Workspace from "@/components/Workspace.vue";

const auth = useAuthStore();
const chats = useChatsStore();
const files = useFilesStore();
const tabs = useTabsStore();
const inbox = useInboxStore();
const sessions = useSessionsStore();
const jobs = useJobsStore();

const leftOpen = ref(loadString("layout.left", "1") === "1");
const rightOpen = ref(loadString("layout.right", "1") === "1");
const ready = ref(false);

watch(leftOpen, (v) => saveString("layout.left", v ? "1" : "0"));
watch(rightOpen, (v) => saveString("layout.right", v ? "1" : "0"));

const DEFAULT_WIDTH = 300;
function loadWidth(key: string): number {
  const n = Number(loadString(key, String(DEFAULT_WIDTH)));
  return Number.isFinite(n) && n >= 200 && n <= 640 ? n : DEFAULT_WIDTH;
}
const leftWidth = ref(loadWidth("layout.leftWidth"));
const rightWidth = ref(loadWidth("layout.rightWidth"));
watch(leftWidth, (v) => saveString("layout.leftWidth", String(v)));
watch(rightWidth, (v) => saveString("layout.rightWidth", String(v)));

applyAppearance();

async function start(): Promise<void> {
  if (ready.value) return;
  ready.value = true;
  await Promise.all([chats.refreshAll(), chats.loadModels(), files.load(""), tabs.hydrate()]);
  void files.startEvents();
  inbox.watch();
  void inbox.refreshCounts();
  sessions.watch();
  void sessions.refreshCount();
}

let chord = false;
let chordTimer: ReturnType<typeof setTimeout> | undefined;

function typingTarget(target: EventTarget | null): boolean {
  const el = target instanceof HTMLElement ? target : null;
  return Boolean(el?.closest("input, textarea, select, [contenteditable='true'], .cm-content"));
}

function onGlobalKey(ev: KeyboardEvent): void {
  if (ev.metaKey || ev.ctrlKey || ev.altKey || typingTarget(ev.target)) {
    chord = false;
    return;
  }
  const key = ev.key.toLowerCase();
  if (!chord) {
    if (key === "g") {
      chord = true;
      clearTimeout(chordTimer);
      chordTimer = setTimeout(() => {
        chord = false;
      }, 800);
    }
    return;
  }
  chord = false;
  clearTimeout(chordTimer);
  if (!ready.value) return;
  const go: Record<string, () => void> = {
    d: () => openDay(),
    s: () => openSessions(),
    i: () => openInbox(),
    p: () => openSpeakers(),
    a: () => openAssistants(),
    b: () => openBrainmap(),
  };
  if (!go[key]) return;
  ev.preventDefault();
  go[key]();
}

onMounted(() => {
  window.addEventListener("keydown", onGlobalKey);
  if (MOCK_ENABLED || !auth.needsToken) void start();
});
onBeforeUnmount(() => {
  window.removeEventListener("keydown", onGlobalKey);
  clearTimeout(chordTimer);
  files.stopEvents();
  jobs.stop();
});

async function onToken(token: string): Promise<string | null> {
  auth.setToken(token);
  try {
    await api().checkToken();
  } catch (e) {
    auth.rejected();
    return e instanceof Error ? e.message : String(e);
  }
  ready.value = false;
  await start();
  return null;
}

function signOut(): void {
  files.stopEvents();
  jobs.stop();
  ready.value = false;
  auth.signOut();
}
</script>

<template>
  <q-layout view="hHh LpR fFf">
    <q-header bordered class="bg-primary text-white">
      <q-toolbar dense>
        <q-btn flat dense round :icon="ICONS.leftPanel" aria-label="Chats" @click="leftOpen = !leftOpen">
          <q-tooltip>{{ leftOpen ? "Hide" : "Show" }} chats</q-tooltip>
        </q-btn>
        <q-toolbar-title class="text-subtitle1 brand">
          <span class="brand-mark" aria-hidden="true">♟</span>Pawn
          <q-badge v-if="MOCK_ENABLED" color="orange" class="q-ml-sm" label="mock" />
        </q-toolbar-title>
        <q-btn flat dense round :icon="ICONS.today" aria-label="Day" :disable="!ready" @click="openDay()">
          <q-tooltip>Day · g d</q-tooltip>
        </q-btn>
        <q-btn flat dense round :icon="ICONS.sessions" aria-label="Sessions" :disable="!ready" @click="openSessions()">
          <q-badge
            v-if="sessions.needsSpeakers"
            floating
            color="orange"
            rounded
            :label="sessions.needsSpeakers > 99 ? '99+' : sessions.needsSpeakers"
          />
          <q-tooltip>Sessions · {{ sessions.needsSpeakers }} need speakers · g s</q-tooltip>
        </q-btn>
        <q-btn flat dense round :icon="ICONS.inbox" aria-label="Inbox" :disable="!ready" @click="openInbox()">
          <q-badge v-if="inbox.total" floating color="orange" rounded :label="inbox.total > 99 ? '99+' : inbox.total" />
          <q-tooltip>
            Inbox · {{ inbox.counts.items }} items · {{ inbox.counts.ideas }} ideas · {{ inbox.counts.captures }} captures · g i
          </q-tooltip>
        </q-btn>
        <q-btn flat dense round :icon="ICONS.more" aria-label="More" :disable="!ready">
          <q-tooltip>More</q-tooltip>
          <q-menu anchor="bottom right" self="top right">
            <q-list dense style="min-width: 200px">
              <q-item v-close-popup clickable @click="openBrainmap()">
                <q-item-section avatar><q-icon :name="ICONS.brainmap" /></q-item-section>
                <q-item-section>Brainmap</q-item-section>
                <q-item-section side class="text-caption text-muted">g b</q-item-section>
              </q-item>
              <q-item v-close-popup clickable @click="openSpeakers()">
                <q-item-section avatar><q-icon :name="ICONS.speakers" /></q-item-section>
                <q-item-section>Speakers</q-item-section>
                <q-item-section side class="text-caption text-muted">g p</q-item-section>
              </q-item>
              <q-item v-close-popup clickable @click="openAssistants()">
                <q-item-section avatar><q-icon :name="ICONS.assistants" /></q-item-section>
                <q-item-section>Assistants</q-item-section>
                <q-item-section side class="text-caption text-muted">g a</q-item-section>
              </q-item>
              <q-item v-close-popup clickable @click="openSettings()">
                <q-item-section avatar><q-icon :name="ICONS.settings" /></q-item-section>
                <q-item-section>Settings</q-item-section>
              </q-item>
            </q-list>
          </q-menu>
        </q-btn>
        <q-btn
          flat
          dense
          round
          :icon="mode === 'auto' ? ICONS.modeAuto : mode === 'dark' ? ICONS.modeDark : ICONS.modeLight"
          :aria-label="`Theme: ${skinLabel(skin)}, ${mode}`"
          @click="cycleMode"
        >
          <q-tooltip>Theme: {{ skinLabel(skin) }} · {{ mode }}</q-tooltip>
        </q-btn>
        <q-btn v-if="!MOCK_ENABLED" flat dense round :icon="ICONS.key" aria-label="Change token" @click="signOut">
          <q-tooltip>Change token</q-tooltip>
        </q-btn>
        <q-btn flat dense round :icon="ICONS.rightPanel" aria-label="Files" @click="rightOpen = !rightOpen">
          <q-tooltip>{{ rightOpen ? "Hide" : "Show" }} files</q-tooltip>
        </q-btn>
      </q-toolbar>
    </q-header>

    <q-drawer v-model="leftOpen" side="left" bordered :width="leftWidth" show-if-above>
      <ChatSidebar v-if="ready" />
      <ResizeHandle v-model:width="leftWidth" side="left" />
    </q-drawer>

    <q-drawer v-model="rightOpen" side="right" bordered :width="rightWidth" show-if-above>
      <FilesSidebar v-if="ready" />
      <ResizeHandle v-model:width="rightWidth" side="right" />
    </q-drawer>

    <q-page-container>
      <q-page :style-fn="pageStyle">
        <Workspace v-if="ready" />
      </q-page>
    </q-page-container>

    <ContextMenu v-if="ready" />
    <TokenDialog v-if="!MOCK_ENABLED && auth.needsToken" :submit="onToken" />
  </q-layout>
</template>

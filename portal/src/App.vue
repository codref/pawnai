<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Dark } from "quasar";
import { api, MOCK_ENABLED } from "@/api";
import { useAuthStore } from "@/stores/auth";
import { useChatsStore } from "@/stores/chats";
import { useFilesStore } from "@/stores/files";
import { useInboxStore } from "@/stores/inbox";
import { openInbox, openSettings } from "@/router";
import { useTabsStore } from "@/stores/tabs";
import { loadString, saveString } from "@/lib/storage";
import { pageStyle } from "@/lib/layout";
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

const theme = ref(loadString("theme", "auto"));
function applyTheme(): void {
  Dark.set(theme.value === "auto" ? "auto" : theme.value === "dark");
}
function cycleTheme(): void {
  theme.value = theme.value === "auto" ? "dark" : theme.value === "dark" ? "light" : "auto";
  saveString("theme", theme.value);
  applyTheme();
}
applyTheme();

async function start(): Promise<void> {
  if (ready.value) return;
  ready.value = true;
  await Promise.all([chats.refreshAll(), chats.loadModels(), files.load(""), tabs.hydrate()]);
  void files.startEvents();
  inbox.watch();
  void inbox.refreshCounts();
}

onMounted(() => {
  if (MOCK_ENABLED || !auth.needsToken) void start();
});
onBeforeUnmount(() => files.stopEvents());

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
  ready.value = false;
  auth.signOut();
}
</script>

<template>
  <q-layout view="hHh LpR fFf">
    <q-header bordered class="bg-primary text-white">
      <q-toolbar dense>
        <q-btn flat dense round icon="menu" aria-label="Chats" @click="leftOpen = !leftOpen" />
        <q-toolbar-title class="text-subtitle1">
          Pawn
          <q-badge v-if="MOCK_ENABLED" color="orange" class="q-ml-sm" label="mock" />
        </q-toolbar-title>
        <q-btn flat dense round icon="inbox" aria-label="Inbox" :disable="!ready" @click="openInbox()">
          <q-badge v-if="inbox.total" floating color="orange" rounded :label="inbox.total > 99 ? '99+' : inbox.total" />
          <q-tooltip>
            Inbox · {{ inbox.counts.items }} items · {{ inbox.counts.ideas }} ideas · {{ inbox.counts.captures }} captures
          </q-tooltip>
        </q-btn>
        <q-btn flat dense round icon="settings" aria-label="Settings" :disable="!ready" @click="openSettings()">
          <q-tooltip>Settings</q-tooltip>
        </q-btn>
        <q-btn
          flat
          dense
          round
          :icon="theme === 'auto' ? 'brightness_auto' : theme === 'dark' ? 'dark_mode' : 'light_mode'"
          :aria-label="`Theme: ${theme}`"
          @click="cycleTheme"
        >
          <q-tooltip>Theme: {{ theme }}</q-tooltip>
        </q-btn>
        <q-btn v-if="!MOCK_ENABLED" flat dense round icon="key" aria-label="Change token" @click="signOut">
          <q-tooltip>Change token</q-tooltip>
        </q-btn>
        <q-btn flat dense round icon="folder_open" aria-label="Files" @click="rightOpen = !rightOpen" />
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

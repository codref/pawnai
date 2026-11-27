<script setup lang="ts">
import { useChatsStore } from "@/stores/chats";
import { openChat } from "@/router";

const chats = useChatsStore();

async function newChat(): Promise<void> {
  const chat = await chats.createChat();
  if (chat) openChat(chat.id, { pin: true });
}
</script>

<template>
  <div class="flex flex-center column q-pa-lg text-center full-height">
    <q-icon name="forum" size="56px" class="text-muted" />
    <div class="text-h6 q-mt-md">Pick a chat or a file</div>
    <div class="text-body2 text-muted q-mb-md">Chats are on the left, the vault is on the right.</div>
    <q-btn color="primary" icon="add" label="New chat" unelevated @click="newChat" />
  </div>
</template>

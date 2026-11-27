import { defineStore } from "pinia";
import { computed, ref } from "vue";
import {
  api,
  type Chat,
  type ChatOrigin,
  type ChatPatch,
  type ChatSort,
  type Clip,
  type FolderNode,
  type ModelCatalog,
  type SortOrder,
  type TagNode,
} from "@/api";
import { loadJson, loadString, saveJson, saveString } from "@/lib/storage";
import { notifyError } from "@/lib/errors";

export type ChatView = "folders" | "tags" | "all";

/** A side chat ("Ask about this") that has not been promoted. */
export function isSideChat(c: Pick<Chat, "origin">): boolean {
  return Boolean(c.origin && !c.origin.promoted);
}

export interface SortChoice {
  sort: ChatSort;
  order: SortOrder;
}

const DEFAULT_SORTS: Record<ChatView, SortChoice> = {
  folders: { sort: "updated", order: "desc" },
  tags: { sort: "tag", order: "asc" },
  all: { sort: "updated", order: "desc" },
};

export const useChatsStore = defineStore("chats", () => {
  const view = ref<ChatView>((loadString("chat.view") as ChatView) || "folders");
  const sorts = ref<Record<ChatView, SortChoice>>({
    ...DEFAULT_SORTS,
    ...loadJson<Partial<Record<ChatView, SortChoice>>>("chat.sorts", {}),
  });
  const chats = ref<Chat[]>([]);
  const archived = ref<Chat[]>([]);
  const archivedLoaded = ref(false);
  const folders = ref<FolderNode[]>([]);
  const unfiledCount = ref(0);
  const tagTree = ref<TagNode[]>([]);
  /** Every tag in use: chats, clips, annotations and vault files. */
  const allTags = ref<TagNode[]>([]);
  const clips = ref<Clip[]>([]);
  const loading = ref(false);
  const models = ref<ModelCatalog | null>(null);
  const model = ref(loadString("chat.model"));
  /** Bumped after a chat changes so open panels can refetch. */
  const revision = ref(0);

  const sort = computed(() => sorts.value[view.value]);
  const pinned = computed(() => chats.value.filter((c) => c.pinned));
  const byId = computed(() => {
    const m = new Map<string, Chat>();
    for (const c of [...chats.value, ...archived.value]) m.set(c.id, c);
    return m;
  });
  /** Side chats by parent chat id (unarchived list only). */
  const sideChildren = computed(() => {
    const m = new Map<string, Chat[]>();
    for (const c of chats.value) {
      if (!isSideChat(c) || !c.parent_chat_id) continue;
      m.set(c.parent_chat_id, [...(m.get(c.parent_chat_id) ?? []), c]);
    }
    return m;
  });
  /** Chat lists hide side chats that show nested under a listed parent. */
  function nested(c: Chat): boolean {
    return isSideChat(c) && Boolean(c.parent_chat_id && chats.value.some((p) => p.id === c.parent_chat_id));
  }
  /** "Scroll to this message" request for an open ChatPanel. */
  const focusRequest = ref<{ chatId: string; messageId: string; at: number } | null>(null);
  function requestFocus(chatId: string, messageId: string): void {
    focusRequest.value = { chatId, messageId, at: Date.now() };
  }

  function setView(v: ChatView): void {
    view.value = v;
    saveString("chat.view", v);
    void refreshChats();
  }

  function setSort(choice: Partial<SortChoice>): void {
    sorts.value = { ...sorts.value, [view.value]: { ...sort.value, ...choice } };
    saveJson("chat.sorts", sorts.value);
    void refreshChats();
  }

  function setModel(id: string): void {
    model.value = id;
    saveString("chat.model", id || null);
  }

  async function refreshChats(): Promise<void> {
    try {
      chats.value = await api().listChats({ sort: sort.value.sort, order: sort.value.order });
      if (archivedLoaded.value) await loadArchived();
    } catch (e) {
      notifyError(e, "Could not load chats");
    }
  }

  async function refreshFolders(): Promise<void> {
    try {
      const tree = await api().listFolders();
      folders.value = tree.folders;
      unfiledCount.value = tree.unfiled_count;
    } catch (e) {
      notifyError(e, "Could not load folders");
    }
  }

  async function refreshTags(): Promise<void> {
    try {
      const trees = await api().listTags();
      tagTree.value = trees.chats;
      allTags.value = trees.tree;
    } catch (e) {
      notifyError(e, "Could not load tags");
    }
  }

  async function refreshClips(): Promise<void> {
    try {
      clips.value = await api().listClips();
    } catch (e) {
      notifyError(e, "Could not load clips");
    }
  }

  async function loadArchived(): Promise<void> {
    archived.value = await api().listChats({
      archived: true,
      sort: sort.value.sort,
      order: sort.value.order,
    });
    archivedLoaded.value = true;
  }

  async function loadModels(): Promise<void> {
    try {
      models.value = await api().listModels();
      const ids = models.value.models.map((m) => m.id);
      if (!model.value || !ids.includes(model.value)) model.value = models.value.default;
    } catch (e) {
      notifyError(e, "Could not load models");
    }
  }

  async function refreshAll(): Promise<void> {
    loading.value = true;
    try {
      await Promise.all([refreshChats(), refreshFolders(), refreshTags(), refreshClips()]);
    } finally {
      loading.value = false;
    }
  }

  async function createChat(
    opts: {
      folder_id?: string | null;
      tags?: string[];
      title?: string;
      parent_chat_id?: string | null;
      origin?: ChatOrigin | null;
    } = {},
  ): Promise<Chat | null> {
    try {
      const chat = await api().createChat(opts);
      chats.value = [chat, ...chats.value];
      if (opts.origin || opts.parent_chat_id) revision.value += 1;
      void refreshFolders();
      if (opts.tags?.length) void refreshTags();
      return chat;
    } catch (e) {
      notifyError(e, "Could not create chat");
      return null;
    }
  }

  async function patchChat(id: string, body: ChatPatch): Promise<Chat | null> {
    try {
      const chat = await api().patchChat(id, body);
      revision.value += 1;
      await refreshChats();
      if (body.folder_id !== undefined || body.unfile || body.archived !== undefined)
        void refreshFolders();
      if (body.tags !== undefined || body.archived !== undefined) void refreshTags();
      return chat;
    } catch (e) {
      notifyError(e, "Could not update chat");
      return null;
    }
  }

  async function deleteChat(id: string): Promise<boolean> {
    try {
      await api().deleteChat(id);
      chats.value = chats.value.filter((c) => c.id !== id);
      archived.value = archived.value.filter((c) => c.id !== id);
      revision.value += 1;
      void refreshFolders();
      void refreshTags();
      return true;
    } catch (e) {
      notifyError(e, "Could not delete chat");
      return false;
    }
  }

  /** Move a chat into a folder; null means Unfiled. */
  async function moveChat(id: string, folderId: string | null): Promise<void> {
    await patchChat(id, folderId ? { folder_id: folderId } : { unfile: true });
  }

  async function createFolder(name: string, parentId: string | null = null): Promise<void> {
    try {
      await api().createFolder({ name, parent_id: parentId });
      await refreshFolders();
    } catch (e) {
      notifyError(e, "Could not create folder");
    }
  }

  async function renameFolder(id: string, name: string): Promise<void> {
    try {
      await api().patchFolder(id, { name });
      await refreshFolders();
    } catch (e) {
      notifyError(e, "Could not rename folder");
    }
  }

  async function moveFolder(id: string, parentId: string | null): Promise<void> {
    try {
      await api().patchFolder(id, parentId ? { parent_id: parentId } : { to_root: true });
      await refreshFolders();
    } catch (e) {
      notifyError(e, "Could not move folder");
    }
  }

  async function deleteFolder(id: string): Promise<void> {
    try {
      await api().deleteFolder(id);
      await Promise.all([refreshFolders(), refreshChats()]);
    } catch (e) {
      notifyError(e, "Could not delete folder");
    }
  }

  async function renameTag(from: string, to: string): Promise<void> {
    try {
      await api().renameTag(from, to);
      revision.value += 1;
      await Promise.all([refreshTags(), refreshChats(), refreshClips()]);
    } catch (e) {
      notifyError(e, "Could not rename tag");
    }
  }

  /** Called after a chat turn: titles and last_message_at may have changed. */
  function touched(): void {
    void refreshChats();
  }

  return {
    view,
    sorts,
    sort,
    chats,
    archived,
    archivedLoaded,
    folders,
    unfiledCount,
    tagTree,
    allTags,
    clips,
    loading,
    models,
    model,
    revision,
    pinned,
    byId,
    sideChildren,
    nested,
    focusRequest,
    requestFocus,
    setView,
    setSort,
    setModel,
    refreshChats,
    refreshFolders,
    refreshTags,
    refreshClips,
    refreshAll,
    loadArchived,
    loadModels,
    createChat,
    patchChat,
    deleteChat,
    moveChat,
    createFolder,
    renameFolder,
    moveFolder,
    deleteFolder,
    renameTag,
    touched,
  };
});

// User boards in the sidebar: list, create, rename, tag, file, archive,
// delete. The graph of one board lives in its brainmap store.
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api, type Board, type BoardCreate, type BoardPatch } from "@/api";
import { notifyError } from "@/lib/errors";

export const useBoardsStore = defineStore("boards", () => {
  const boards = ref<Board[]>([]);
  const archived = ref<Board[]>([]);
  const loaded = ref(false);
  /** A first message a new board's chat sends on open (board from a chat). */
  const pendingPrompt = ref<Record<string, string>>({});

  const byName = computed(() => new Map([...boards.value, ...archived.value].map((b) => [b.name, b])));
  /** User boards, newest first (the built-in Items board is listed apart). */
  const userBoards = computed(() => boards.value.filter((b) => b.kind !== "items"));
  const pinned = computed(() => userBoards.value.filter((b) => b.pinned));

  async function refresh(): Promise<void> {
    try {
      boards.value = await api().listBoards();
      loaded.value = true;
    } catch (e) {
      notifyError(e, "Could not load boards");
    }
  }

  async function loadArchived(): Promise<void> {
    try {
      archived.value = await api().listBoards(true);
    } catch (e) {
      notifyError(e, "Could not load archived boards");
    }
  }

  function upsert(b: Board): void {
    const others = boards.value.filter((x) => x.name !== b.name);
    if (b.archived) {
      boards.value = others;
      archived.value = [b, ...archived.value.filter((x) => x.name !== b.name)];
    } else {
      archived.value = archived.value.filter((x) => x.name !== b.name);
      boards.value = [b, ...others];
    }
  }

  async function create(body: BoardCreate, firstPrompt?: string): Promise<Board | null> {
    try {
      const b = await api().createBoard(body);
      upsert(b);
      if (firstPrompt) pendingPrompt.value = { ...pendingPrompt.value, [b.name]: firstPrompt };
      return b;
    } catch (e) {
      notifyError(e, "Could not create the board");
      return null;
    }
  }

  async function patch(name: string, body: BoardPatch): Promise<Board | null> {
    try {
      const b = await api().patchBoard(name, body);
      upsert(b);
      return b;
    } catch (e) {
      notifyError(e, "Could not update the board");
      return null;
    }
  }

  async function remove(name: string): Promise<boolean> {
    try {
      await api().deleteBoard(name);
      boards.value = boards.value.filter((b) => b.name !== name);
      archived.value = archived.value.filter((b) => b.name !== name);
      return true;
    } catch (e) {
      notifyError(e, "Could not delete the board");
      return false;
    }
  }

  async function pin(name: string, nodeIds: string[]): Promise<boolean> {
    try {
      upsert(await api().pinToBoard(name, nodeIds));
      return true;
    } catch (e) {
      notifyError(e, "Could not pin to the board");
      return false;
    }
  }

  function takePrompt(name: string): string {
    const p = pendingPrompt.value[name] ?? "";
    if (p) {
      const next = { ...pendingPrompt.value };
      delete next[name];
      pendingPrompt.value = next;
    }
    return p;
  }

  return {
    boards,
    archived,
    loaded,
    byName,
    userBoards,
    pinned,
    refresh,
    loadArchived,
    upsert,
    create,
    patch,
    remove,
    pin,
    takePrompt,
  };
});

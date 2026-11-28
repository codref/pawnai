import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api, type FileListing } from "@/api";
import { loadJson, saveJson } from "@/lib/storage";
import { isAbort, notifyError } from "@/lib/errors";

export interface RecentEntry {
  path: string;
  at: number;
  source: string;
  deleted?: boolean;
}

const RECENT_MAX = 100;
const POLL_TIMEOUT = 25;

function parentOf(key: string): string {
  const i = key.lastIndexOf("/");
  return i >= 0 ? key.slice(0, i) : "";
}

export const useFilesStore = defineStore("files", () => {
  /** One level of the vault per prefix ("" = root), filled lazily. */
  const listings = ref<Record<string, FileListing>>({});
  const scratchDir = ref("Pawn/Scratch");
  const agentRoot = ref("Pawn");
  /** Keys written since the viewer last opened them (per-browser). */
  const unseen = ref<string[]>(loadJson<string[]>("files.unseen", []));
  const recent = ref<RecentEntry[]>(loadJson<RecentEntry[]>("files.recent", []));
  /** Bumped per key when it changes on the server (open FilePanels reload). */
  const changed = ref<Record<string, number>>({});
  let allKeys: string[] | null = null;
  let polling: AbortController | null = null;
  /** Subscribers to vault writes under some key prefixes (Inbox refresh). */
  const keyWatchers = new Set<{ match: (key: string) => boolean; cb: (paths: string[]) => void }>();

  /** Call *cb* with the changed keys whenever a vault event touches a key *match* accepts. */
  function onKeys(match: (key: string) => boolean, cb: (paths: string[]) => void): () => void {
    const w = { match, cb };
    keyWatchers.add(w);
    return () => keyWatchers.delete(w);
  }

  /**
   * Debounced listener for vault writes under *prefix* (the prefix itself or a
   * descendant). Sessions watch transcripts; Day watches events and daily notes.
   */
  function onVaultChange(prefix: string, fn: (paths: string[]) => void): () => void {
    const root = prefix.replace(/\/+$/, "");
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pending: string[] = [];
    const stop = onKeys(
      (key) => key === root || key.startsWith(`${root}/`),
      (paths) => {
        pending.push(...paths);
        clearTimeout(timer);
        timer = setTimeout(() => {
          const batch = pending;
          pending = [];
          fn(batch);
        }, 500);
      },
    );
    return () => {
      clearTimeout(timer);
      stop();
    };
  }

  function notifyWatchers(paths: string[]): void {
    for (const w of keyWatchers) {
      const hit = paths.filter(w.match);
      if (hit.length) w.cb(hit);
    }
  }

  const unseenSet = computed(() => new Set(unseen.value));

  function persist(): void {
    saveJson("files.unseen", unseen.value);
    saveJson("files.recent", recent.value);
  }

  function isUnseen(path: string): boolean {
    return unseenSet.value.has(path);
  }

  /** True when any unseen key sits under the folder *prefix*. */
  function folderUnseen(prefix: string): boolean {
    const p = prefix + "/";
    return unseen.value.some((k) => k.startsWith(p));
  }

  function markSeen(path: string): void {
    if (!unseenSet.value.has(path)) return;
    unseen.value = unseen.value.filter((k) => k !== path);
    persist();
  }

  function markAllSeen(): void {
    unseen.value = [];
    persist();
  }

  async function load(prefix: string): Promise<FileListing | null> {
    try {
      const listing = await api().listFiles(prefix);
      listings.value = { ...listings.value, [prefix]: listing };
      if (listing.scratch_dir) scratchDir.value = listing.scratch_dir;
      if (listing.agent_root) agentRoot.value = listing.agent_root;
      return listing;
    } catch (e) {
      notifyError(e, `Could not list ${prefix || "vault"}`);
      return null;
    }
  }

  /** Reload every loaded folder that is an ancestor of *key*. */
  async function refreshFor(keys: string[]): Promise<void> {
    const prefixes = new Set<string>();
    for (const key of keys) {
      let p = parentOf(key);
      for (;;) {
        if (p in listings.value) prefixes.add(p);
        if (!p) break;
        p = parentOf(p);
      }
    }
    await Promise.all([...prefixes].map((p) => load(p)));
  }

  async function exists(key: string): Promise<boolean> {
    const listing = listings.value[parentOf(key)] ?? (await load(parentOf(key)));
    return Boolean(listing?.files.some((f) => f.path === key));
  }

  function recordEvent(paths: string[], source: string, markUnseen: boolean): void {
    const at = Date.now();
    const existing = recent.value.filter((r) => !paths.includes(r.path));
    recent.value = [...paths.map((path) => ({ path, at, source })), ...existing].slice(0, RECENT_MAX);
    if (markUnseen && source !== "portal") {
      unseen.value = [...new Set([...paths, ...unseen.value])];
    }
    const bump = { ...changed.value };
    for (const p of paths) bump[p] = (bump[p] ?? 0) + 1;
    changed.value = bump;
    allKeys = null;
    persist();
  }

  /** Long-poll /v1/vault/events and feed Recent, unseen dots and the tree. */
  async function startEvents(): Promise<void> {
    if (polling) return;
    const ctl = new AbortController();
    polling = ctl;
    let since = 0;
    let first = true;
    while (!ctl.signal.aborted) {
      try {
        const res = await api().vaultEvents(since, first ? 0 : POLL_TIMEOUT, ctl.signal);
        if (!first && res.events.length) {
          for (const ev of res.events) recordEvent(ev.paths, ev.source, true);
          const changed = res.events.flatMap((e) => e.paths);
          notifyWatchers(changed);
          await refreshFor(changed);
        }
        if (res.resync) await Promise.all(Object.keys(listings.value).map((p) => load(p)));
        since = res.seq;
        first = false;
      } catch (e) {
        if (isAbort(e)) return;
        await new Promise((r) => setTimeout(r, 5000));
      }
    }
  }

  function stopEvents(): void {
    polling?.abort();
    polling = null;
  }

  async function listAll(): Promise<string[]> {
    if (!allKeys) allKeys = await api().listAllFiles("");
    return allKeys;
  }

  /** Resolve an Obsidian-style [[target]] to a vault key, or null. */
  async function resolveWikilink(target: string, fromKey = ""): Promise<string | null> {
    const clean = target.split("#")[0].split("|")[0].trim().replace(/^\/+/, "");
    if (!clean) return null;
    const keys = await listAll();
    const withMd = /\.[a-z0-9]+$/i.test(clean) ? [clean] : [`${clean}.md`, clean];
    for (const cand of withMd) if (keys.includes(cand)) return cand;
    const dir = parentOf(fromKey);
    if (dir) for (const cand of withMd) if (keys.includes(`${dir}/${cand}`)) return `${dir}/${cand}`;
    const lower = withMd.map((c) => "/" + c.toLowerCase());
    const hit = keys.find((k) => lower.some((c) => ("/" + k.toLowerCase()).endsWith(c)));
    return hit ?? null;
  }

  async function upload(file: File, folder: string): Promise<string | null> {
    try {
      const stat = await api().uploadFile(file, folder);
      recordEvent([stat.key], "portal", false);
      await refreshFor([stat.key]);
      return stat.key;
    } catch (e) {
      notifyError(e, `Upload of ${file.name} failed`);
      return null;
    }
  }

  async function createNote(key: string, content: string): Promise<boolean> {
    try {
      const stat = await api().writeFile(key, content, "");
      recordEvent([stat.key], "portal", false);
      await refreshFor([stat.key]);
      return true;
    } catch (e) {
      notifyError(e, "Could not create note");
      return false;
    }
  }

  async function move(src: string, dst: string, overwrite = false): Promise<string | null> {
    try {
      const stat = await api().moveFile(src, dst, overwrite);
      recordEvent([src, stat.key], "portal", false);
      markSeen(src);
      await refreshFor([src, stat.key]);
      return stat.key;
    } catch (e) {
      notifyError(e, "Move failed");
      return null;
    }
  }

  async function remove(key: string): Promise<boolean> {
    try {
      await api().deleteFile(key);
      markSeen(key);
      recordEvent([key], "portal", false);
      recent.value = recent.value.map((r) => (r.path === key ? { ...r, deleted: true } : r));
      await refreshFor([key]);
      return true;
    } catch (e) {
      notifyError(e, "Delete failed");
      return false;
    }
  }

  /** Record a file a chat turn wrote (the `files` SSE event). */
  function noteWritten(paths: string[]): void {
    recordEvent(paths, "agent", true);
    void refreshFor(paths);
  }

  return {
    listings,
    scratchDir,
    agentRoot,
    unseen,
    recent,
    changed,
    isUnseen,
    folderUnseen,
    markSeen,
    markAllSeen,
    load,
    refreshFor,
    exists,
    startEvents,
    stopEvents,
    onKeys,
    onVaultChange,
    resolveWikilink,
    upload,
    createNote,
    move,
    remove,
    noteWritten,
  };
});

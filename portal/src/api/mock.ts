// In-browser mock of the portal API, enabled with VITE_PAWN_MOCK=1.
// State lives in memory only; a reload starts from the seed data again.

import {
  HttpError,
  type Annotation,
  type AnnotationCreate,
  type CaptureAtom,
  type CaptureCollection,
  type CaptureFileBody,
  type Idea,
  type IdeaActionKind,
  type IdeaActionResult,
  type InboxCounts,
  type InboxItem,
  type ItemAction,
  type ItemActionResult,
  type ItemListParams,
  type ItemsDeleteBody,
  type ItemsPage,
  type SettingField,
  type SettingsPayload,
  type SettingsStatus,
  type AnnotationListParams,
  type AnnotationPatch,
  type Chat,
  type ChatCreate,
  type ChatDetail,
  type ChatListParams,
  type ChatPatch,
  type ChatStreamHandlers,
  type Clip,
  type ClipCreate,
  type ClipPatch,
  type Fork,
  type FileContent,
  type FileListing,
  type FileHit,
  type FileStat,
  type FileTags,
  type Folder,
  type FolderCreate,
  type FolderNode,
  type FolderPatch,
  type FolderTree,
  type Message,
  type MessageCreate,
  type ModelCatalog,
  type PortalApi,
  type ReindexResult,
  type SearchResults,
  type TagNode,
  type TagTrees,
  type VaultEvent,
  type VaultEvents,
  type Workspace,
} from "./types";

const SCRATCH = "Pawn/Scratch";

let counter = 0;
const uid = () => `m${Date.now().toString(36)}${(counter++).toString(36)}`;
const nowIso = () => new Date().toISOString();
const minutesAgo = (m: number) => new Date(Date.now() - m * 60_000).toISOString();

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });
}

function cleanTags(tags: string[] | undefined): string[] {
  const out: string[] = [];
  for (const raw of tags ?? []) {
    const t = raw
      .split("/")
      .map((p) => p.trim().replace(/^#/, ""))
      .filter(Boolean)
      .join("/");
    if (t && !out.includes(t)) out.push(t);
  }
  return out;
}

function tagMatches(tags: string[], prefix: string): boolean {
  return tags.some((t) => t === prefix || t.startsWith(prefix + "/"));
}

function buildTagTree(lists: string[][]): TagNode[] {
  interface N {
    name: string;
    path: string;
    count: number;
    children: Map<string, N>;
  }
  const root: N = { name: "", path: "", count: 0, children: new Map() };
  lists.forEach((tags, idx) => {
    const seen = new Set<string>();
    for (const tag of tags) {
      let node = root;
      let path = "";
      for (const part of tag.split("/")) {
        path = path ? `${path}/${part}` : part;
        let child = node.children.get(part);
        if (!child) {
          child = { name: part, path, count: 0, children: new Map() };
          node.children.set(part, child);
        }
        const key = `${idx}:${path}`;
        if (!seen.has(key)) {
          child.count += 1;
          seen.add(key);
        }
        node = child;
      }
    }
  });
  const out = (n: N): TagNode[] =>
    [...n.children.values()]
      .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()))
      .map((c) => ({ name: c.name, path: c.path, count: c.count, children: out(c) }));
  return out(root);
}

function sortChats(chats: Chat[], sort: string, order: string): Chat[] {
  const dir = order === "asc" ? 1 : -1;
  const name = (c: Chat) => (c.title || "").toLowerCase();
  const cmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
  const list = [...chats];
  if (sort === "name") return list.sort((a, b) => dir * cmp(name(a), name(b)));
  if (sort === "created")
    return list.sort((a, b) => dir * cmp(a.created_at ?? "", b.created_at ?? ""));
  if (sort === "tag") {
    const first = (c: Chat) => [...c.tags].sort()[0]?.toLowerCase() ?? "";
    const tagged = list.filter((c) => c.tags.length);
    const untagged = list.filter((c) => !c.tags.length).sort((a, b) => cmp(name(a), name(b)));
    tagged.sort((a, b) => dir * (cmp(first(a), first(b)) || cmp(name(a), name(b))));
    return [...tagged, ...untagged];
  }
  const when = (c: Chat) => c.last_message_at ?? c.updated_at ?? c.created_at ?? "";
  return list.sort((a, b) => dir * cmp(when(a), when(b)));
}

const SEED_FILES: Record<string, string> = {
  "Pawn/Scratch/Welcome.md": `---
title: Welcome
pawn: editable
---
# Welcome to the scratchpad

This is a **mock** file. Edit it with the pencil button.

- Links: [[Ideas/Portal]] and [[Acme kickoff|the kickoff analysis]]
- A missing link: [[Nowhere]]
`,
  "Pawn/Scratch/Todo.md": "# Todo\n\n- [ ] Try the portal\n- [x] Read the plan\n",
  "Pawn/Analyses/Acme kickoff.md": `# Acme kickoff — analysis

## Summary
The team agreed on a **Q4** launch. See [[People/Davide]].

| Owner | Task |
|---|---|
| Davide | Draft plan |
| Ana | Budget |
`,
  "Pawn/Items/2026-11-26 Follow up with Ana.md": "---\nstatus: open\n---\nFollow up with Ana about the budget.\n",
  "Pawn/Today.md": "# Today\n\n- 09:30 Standup\n- 14:00 Acme sync\n",
  "People/Davide.md": "# Davide\n\nOwner of the portal project.\n",
  "Ideas/Portal.md": "---\nstatus: inbox\n---\n# Portal\n\nA chat-first workspace that replaces Obsidian.\n",
  "Goals.md": "# Goals\n\n1. Ship the portal\n",
};


// ── inbox / settings seed ─────────────────────────────────────────────────

const OPEN_ITEM = ["new", "notified", "snoozed"];
const IDEAS = "Ideas";
const CAPTURE_INBOX = "Pawn/Captures/Inbox";

function seedItem(i: number, kind: string, text: string, extra: Partial<InboxItem> = {}): InboxItem {
  const created = minutesAgo(30 + i * 60 * 9);
  return {
    id: `it${i}`,
    short_id: `i${i}`,
    source_kind: "session",
    source_ref: i % 2 ? "acme-kickoff" : "weekly-sync",
    kind,
    text,
    owner: null,
    due: null,
    quote: null,
    context: null,
    thread: null,
    interrupt: false,
    movement: false,
    reason: null,
    recurrence: 0,
    status: "new",
    snooze_until: null,
    note_key: `Pawn/Items/${created.slice(0, 10)} i${i}.md`,
    payload: {},
    created_at: created,
    updated_at: created,
    ...extra,
  };
}

function frontmatter(text: string): Record<string, string> {
  const m = /^---\n([\s\S]*?)\n---/.exec(text);
  const out: Record<string, string> = {};
  if (!m) return out;
  for (const line of m[1].split("\n")) {
    const kv = /^([A-Za-z_]+):\s*(.*)$/.exec(line);
    if (kv) out[kv[1]] = kv[2].trim().replace(/^["']|["']$/g, "");
  }
  return out;
}

function headingOf(text: string, fallback: string): string {
  const line = text.split("\n").find((l) => l.startsWith("# "));
  return line ? line.slice(2).trim() : fallback;
}

function stemOf(key: string): string {
  return key.split("/").pop()!.replace(/\.md$/, "");
}

function setStatus(text: string, status: string): string {
  if (!text.startsWith("---\n")) return `---\nstatus: ${status}\n---\n\n${text}`;
  return /^status:.*$/m.test(text)
    ? text.replace(/^status:.*$/m, `status: ${status}`)
    : text.replace(/^---\n/, `---\nstatus: ${status}\n`);
}

const MOCK_FIELDS: Omit<SettingField, "value" | "running" | "source" | "yaml_value" | "pending_restart">[] = [
  { path: "agent.default", label: "Background model", group: "models", type: "model", help: "Used by autonomous work and when a turn names no model.", apply: "live", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "capture.model", label: "Research capture model", group: "models", type: "model", help: "Empty uses the background model.", apply: "live", secret: false, choices: null, nullable: true, tz_path: null, min: null },
  { path: "agent.name", label: "Agent name", group: "agent", type: "str", help: "How Pawn calls itself.", apply: "new_chats", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "agent.strip_thinking", label: "Strip thinking", group: "agent", type: "bool", help: "Remove <think> blocks from answers.", apply: "new_chats", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "agent.sallm.max_steps", label: "Max tool steps", group: "agent", type: "int", help: "ReAct steps per turn.", apply: "new_chats", secret: false, choices: null, nullable: false, tz_path: null, min: 1 },
  { path: "coworker.enabled", label: "Coworker enabled", group: "coworker", type: "bool", help: "Extract items and run the crons.", apply: "restart", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "coworker.timezone", label: "Timezone", group: "coworker", type: "tz", help: "Day boundaries and cron times.", apply: "restart", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "coworker.briefing_cron", label: "Morning briefing", group: "coworker", type: "cron", help: "Cron for the Today.md briefing.", apply: "restart", secret: false, choices: null, nullable: false, tz_path: "coworker.timezone", min: null },
  { path: "coworker.me", label: "Me", group: "coworker", type: "list", help: "Names that mean you.", apply: "live", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "coworker.autonomy.mode", label: "Autonomy", group: "coworker", type: "enum", help: "How far the coworker may act on its own.", apply: "live", secret: false, choices: ["off", "suggest_only", "approve_writes", "limited_act"], nullable: false, tz_path: null, min: null },
  { path: "coworker.notify.token", label: "ntfy token", group: "coworker", type: "str", help: "Write-only.", apply: "live", secret: true, choices: null, nullable: false, tz_path: null, min: null },
  { path: "capture.instructions", label: "Research instructions", group: "capture", type: "text", help: "Extra guidance for research_capture.", apply: "live", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "matrix_bot.enabled", label: "Matrix bot enabled", group: "matrix", type: "bool", help: "Start the Matrix bot with serve.", apply: "restart", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "portal.auto_name_every", label: "Re-name every N messages", group: "portal", type: "int", help: "0 names a chat after its first answer only.", apply: "live", secret: false, choices: null, nullable: false, tz_path: null, min: 0 },
];

function mockCatalog(): ModelCatalog {
  return {
    default: "ollama@gemma4:4b",
    background: "ollama@gemma4:4b",
    models: [
      { id: "ollama@gemma4:4b", provider: "ollama", model: "gemma4:4b", vision: true },
      { id: "openrouter@qwen3-235b", provider: "openrouter", model: "qwen3-235b" },
    ],
  };
}

const MOCK_YAML: Record<string, unknown> = {
  "agent.default": "ollama@gemma4:4b",
  "capture.model": "",
  "agent.name": "Bob",
  "agent.strip_thinking": true,
  "agent.sallm.max_steps": 8,
  "coworker.enabled": true,
  "coworker.timezone": "Europe/Rome",
  "coworker.briefing_cron": "0 8 * * *",
  "coworker.me": ["Davide"],
  "coworker.autonomy.mode": "suggest_only",
  "coworker.notify.token": "secret",
  "capture.instructions": "",
  "matrix_bot.enabled": false,
  "portal.auto_name_every": 6,
};

export class MockApi implements PortalApi {
  readonly mock = true;
  private chats = new Map<string, Chat>();
  private messages = new Map<string, Message[]>();
  private folders = new Map<string, Folder>();
  private clips: Clip[] = [];
  private annotations: Annotation[] = [];
  private files = new Map<string, { content: string; etag: string; modified: string }>();
  private blobs = new Map<string, Blob>();
  private blobModified = new Map<string, string>();
  /** Stored tags for files (the mock keeps every file's tags here). */
  private fileTags = new Map<string, string[]>();
  private events: VaultEvent[] = [];
  private eventSeq = 0;
  private waiters: (() => void)[] = [];
  private workspace: Workspace = { name: "default", layout: null, updated_at: null };
  private etagCounter = 1;
  private items: InboxItem[] = [
    seedItem(1, "commitment", "Davide drafts the Q4 launch plan", { owner: "Davide", due: "2026-12-05", quote: "I'll have a draft by next Friday." }),
    seedItem(2, "decision", "Q4 launch is confirmed for 1 December", { reason: "Matches goal: Acme launch", interrupt: true }),
    seedItem(3, "open_question", "Who signs off the Acme budget?", { owner: "Ana" }),
    seedItem(4, "proposal", "Research competitor pricing for the launch", {}),
    seedItem(5, "block", "Design review is waiting on legal", { status: "snoozed", snooze_until: minutesAgo(-60 * 24) }),
    seedItem(6, "commitment", "Send the retro notes", { status: "filed" }),
  ];
  private overrides: Record<string, unknown> = {};
  private running: Record<string, unknown> = { ...MOCK_YAML };

  constructor() {
    this.seed();
  }

  private seed(): void {
    for (const [key, content] of Object.entries(SEED_FILES)) {
      this.files.set(key, { content, etag: this.nextEtag(), modified: minutesAgo(90) });
    }
    const idea = (day: string, title: string, line: string, status = "inbox") =>
      this.files.set(`${IDEAS}/${day} ${title}.md`, {
        content: `---\ntags: [idea]\nstatus: ${status}\n---\n\n# ${title}\n\n${line}\n`,
        etag: this.nextEtag(),
        modified: minutesAgo(120),
      });
    idea("2026-11-25", "Solar fruit dryer", "Build a solar fruit dryer for the balcony.");
    idea("2026-11-24", "Pawn weekly digest", "Email myself a weekly digest of filed items.");
    idea("2026-11-10", "Old idea", "Something parked.", "later");
    const atom = (stem: string, title: string, hint: string, type = "") =>
      this.files.set(`${CAPTURE_INBOX}/${stem}.md`, {
        content:
          `---\npawn: capture\nstatus: ${type ? "proposed" : "inbox"}\nkind: text\ncreated: ${minutesAgo(200)}\n` +
          `collection: ${type ? "books" : ""}\nhint: ${hint}\nentity: ""\ntype: ${type}\nsource_url: https://example.com/${stem}\n---\n\n# ${title}\n\n## Snippets\n`,
        etag: this.nextEtag(),
        modified: minutesAgo(200),
      });
    atom("2026-11-26 Thinking in systems-s1", "Thinking in Systems", "books", "book");
    atom("2026-11-26 Rust async notes-s2", "Rust async notes", "programming");
    const work = this.addFolder("Work", null, 0);
    const acme = this.addFolder("Acme", work.id, 0);
    this.addFolder("Personal", null, 1);
    const c1 = this.addChat("Acme Q4 planning", acme.id, ["work/acme/q4"], 30, true);
    this.addChat("Recipe ideas", null, ["home/cooking"], 240);
    const c3 = this.addChat("Weekly review", work.id, ["work", "review"], 60 * 26);
    this.addChat("Old brainstorm", null, [], 60 * 24 * 9, false, true);
    this.pushMessages(c1.id, [
      ["user", "What did we agree in the Acme kickoff?"],
      [
        "assistant",
        "We agreed on a **Q4 launch**. Ana owns the budget, Davide drafts the plan.\n\nI saved the analysis.",
        ["Pawn/Analyses/Acme kickoff.md"],
      ],
    ]);
    this.pushMessages(c3.id, [
      ["user", "Summarize my week"],
      ["assistant", "- Shipped the portal backend\n- Two Acme meetings\n- One open item: follow up with Ana"],
    ]);
    const msgs = this.messages.get(c1.id) ?? [];
    this.clips.push({
      id: uid(),
      chat_id: c1.id,
      message_ids: msgs.map((m) => m.id),
      title: "Acme agreement",
      tags: ["work/acme"],
      content_md: msgs.map((m) => `**${m.role === "user" ? "You" : "Pawn"}:**\n\n${m.content}`).join("\n\n"),
      vault_key: null,
      created_at: minutesAgo(20),
      updated_at: minutesAgo(20),
    });
    const answer = msgs.find((m) => m.role === "assistant");
    const ann = (a: Omit<Annotation, "id" | "created_at" | "updated_at" | "side_chat_id">) =>
      this.annotations.push({ ...a, id: uid(), side_chat_id: null, created_at: minutesAgo(15), updated_at: minutesAgo(15) });
    if (answer) {
      ann({
        target_kind: "message",
        target_ref: answer.id,
        anchor: { exact: "Ana owns the budget", prefix: "We agreed on a Q4 launch. ", suffix: ", Davide drafts the plan.", start: 26, end: 45 },
        body_md: "Check the budget numbers with Ana before Friday.",
        tags: ["work/acme"],
        color: "yellow",
      });
    }
    ann({
      target_kind: "file",
      target_ref: "Pawn/Analyses/Acme kickoff.md",
      anchor: { exact: "agreed on a Q4 launch", prefix: "Summary\nThe team ", suffix: ". See Davide.\n", start: 40, end: 61 },
      body_md: "Launch date still to confirm.",
      tags: ["work/acme/q4"],
      color: "green",
    });
    ann({
      target_kind: "file",
      target_ref: "Pawn/Analyses/Acme kickoff.md",
      anchor: { exact: "the old marketing budget", prefix: "We cut ", suffix: " in half.", start: 300, end: 324 },
      body_md: "This sentence was removed from the analysis: shows as orphaned.",
      tags: [],
      color: "pink",
    });
    this.publish(["Pawn/Today.md"], "coworker");
  }

  private nextEtag(): string {
    return `e${this.etagCounter++}`;
  }

  private addFolder(name: string, parent: string | null, position: number): Folder {
    const f: Folder = {
      id: uid(),
      parent_id: parent,
      name,
      position,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    this.folders.set(f.id, f);
    return f;
  }

  private addChat(
    title: string,
    folder: string | null,
    tags: string[],
    agoMinutes: number,
    pinned = false,
    archived = false,
  ): Chat {
    const id = uid();
    const c: Chat = {
      id,
      conversation_key: `chat:${id}`,
      title,
      folder_id: folder,
      position: 0,
      tags,
      pinned,
      archived,
      parent_chat_id: null,
      fork_from_message_id: null,
      origin: null,
      last_message_at: minutesAgo(agoMinutes),
      created_at: minutesAgo(agoMinutes + 60),
      updated_at: minutesAgo(agoMinutes),
    };
    this.chats.set(id, c);
    this.messages.set(id, []);
    return c;
  }

  private pushMessages(chatId: string, rows: [string, string, string[]?][]): void {
    const list = this.messages.get(chatId) ?? [];
    for (const [role, content, written] of rows) {
      list.push({
        id: uid(),
        chat_id: chatId,
        seq: list.length,
        role,
        content,
        context: written ? { written_files: written } : {},
        run_id: null,
        model: null,
        created_at: nowIso(),
      });
    }
    this.messages.set(chatId, list);
  }

  private chat(id: string): Chat {
    const c = this.chats.get(id);
    if (!c) throw new HttpError(404, "Chat not found");
    return c;
  }

  private publish(paths: string[], source: string, runId: string | null = null): void {
    this.eventSeq += 1;
    this.events.push({ seq: this.eventSeq, paths, source, run_id: runId });
    if (this.events.length > 200) this.events.shift();
    const w = this.waiters;
    this.waiters = [];
    w.forEach((fn) => fn());
  }

  private writeKey(key: string, content: string): FileStat {
    const etag = this.nextEtag();
    this.files.set(key, { content, etag, modified: nowIso() });
    return { key, etag, size: content.length };
  }

  private assertWritable(key: string, existing?: string): void {
    if (key.startsWith(".obsidian/")) throw new HttpError(403, "Writes under .obsidian/ are not allowed");
    if (key.startsWith("Pawn/")) return;
    if (existing !== undefined && /^pawn:\s*editable/m.test(existing)) return;
    throw new HttpError(403, `${key} is outside Pawn/ and not marked pawn: editable`);
  }

  // ── PortalApi ──────────────────────────────────────────────────────────

  async health(): Promise<boolean> {
    return true;
  }

  async checkToken(): Promise<void> {
    await sleep(50);
  }

  async listChats(p: ChatListParams = {}): Promise<Chat[]> {
    await sleep(60);
    let list = [...this.chats.values()].filter((c) => c.archived === Boolean(p.archived));
    if (p.folder_id) list = list.filter((c) => c.folder_id === p.folder_id);
    else if (p.unfiled) list = list.filter((c) => !c.folder_id);
    if (p.pinned !== undefined) list = list.filter((c) => c.pinned === p.pinned);
    if (p.tag) list = list.filter((c) => tagMatches(c.tags, p.tag!));
    if (p.origin_ref) list = list.filter((c) => c.origin?.ref === p.origin_ref);
    if (p.q) {
      const q = p.q.toLowerCase();
      list = list.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          (this.messages.get(c.id) ?? []).some((m) => m.content.toLowerCase().includes(q)),
      );
    }
    return sortChats(list, p.sort ?? "updated", p.order ?? "desc").map((c) => ({
      ...c,
      message_count: this.messages.get(c.id)?.length ?? 0,
    }));
  }

  async createChat(body: ChatCreate = {}): Promise<Chat> {
    await sleep(40);
    const c = this.addChat(body.title ?? "", body.folder_id ?? null, cleanTags(body.tags), 0);
    c.parent_chat_id = body.parent_chat_id ?? null;
    c.origin = body.origin ?? null;
    c.last_message_at = null;
    return { ...c, message_count: 0 };
  }

  async getChat(id: string): Promise<ChatDetail> {
    await sleep(60);
    const c = this.chat(id);
    const msgs = this.messages.get(id) ?? [];
    return {
      ...c,
      message_count: msgs.length,
      messages: msgs.map((m) => ({ ...m, context: { ...m.context } })),
      side_chats: [...this.chats.values()].filter((x) => x.parent_chat_id === id),
    };
  }

  async patchChat(id: string, body: ChatPatch): Promise<Chat> {
    await sleep(40);
    const c = this.chat(id);
    if (body.title !== undefined) {
      c.title = body.title;
      c.title_auto = false;
    }
    if (body.unfile) c.folder_id = null;
    else if (body.folder_id) {
      if (!this.folders.has(body.folder_id)) throw new HttpError(404, "Folder not found");
      c.folder_id = body.folder_id;
    }
    if (body.position !== undefined) c.position = body.position;
    if (body.tags !== undefined) {
      c.tags = cleanTags(body.tags);
      c.tags_auto = false;
    }
    if (body.pinned !== undefined) c.pinned = body.pinned;
    if (body.archived !== undefined) c.archived = body.archived;
    if (body.promoted !== undefined) {
      if (!c.origin) throw new HttpError(422, "Not a side chat");
      c.origin = { ...c.origin, promoted: body.promoted };
    }
    c.updated_at = nowIso();
    return { ...c };
  }

  async deleteChat(id: string): Promise<void> {
    this.chat(id);
    this.chats.delete(id);
    this.messages.delete(id);
    for (const a of this.annotations) if (a.side_chat_id === id) a.side_chat_id = null;
  }

  async moveChats(ids: string[], folderId: string | null): Promise<void> {
    for (const id of ids) {
      const c = this.chats.get(id);
      if (c) c.folder_id = folderId;
    }
  }

  async sendMessage(
    chatId: string,
    body: MessageCreate,
    h: ChatStreamHandlers,
    signal?: AbortSignal,
  ): Promise<void> {
    const chat = this.chat(chatId);
    const text = body.message.trim();
    if (!text && !body.images?.length) throw new HttpError(422, "conversation and message are required");
    const list = this.messages.get(chatId) ?? [];
    const ctx: Message["context"] = {};
    if (body.context?.length) ctx.notes = body.context.map((n) => n.path);
    if (body.selection) ctx.selection = body.selection;
    const user: Message = {
      id: uid(),
      chat_id: chatId,
      seq: list.length,
      role: "user",
      content: text,
      context: ctx,
      run_id: null,
      model: body.model ?? null,
      created_at: nowIso(),
    };
    list.push(user);
    this.messages.set(chatId, list);
    if (!chat.title.trim()) chat.title = text.split("\n")[0].slice(0, 60) || "New chat";
    chat.last_message_at = nowIso();
    h.onMessage?.({ id: user.id, seq: user.seq, role: "user" });

    try {
      for (const step of ["Thinking…", "Using notes skill", "Calling note_search", "Writing answer"]) {
        await sleep(450, signal);
        h.onProgress?.(step, "step");
      }
      if (/fail|error/i.test(text)) {
        h.onError?.("Mock error: the agent run failed.");
        h.onDone?.({ conversation: chat.conversation_key });
        return;
      }
      const written: string[] = [];
      if (/save|write|note/i.test(text)) {
        const key = `${SCRATCH}/Mock note ${new Date().toLocaleTimeString().replace(/:/g, "-")}.md`;
        this.writeKey(key, `# Mock note\n\nWritten for: ${text}\n`);
        written.push(key);
      }
      const notes = body.context?.length ? `\n\nContext notes: ${body.context.map((n) => `\`${n.path}\``).join(", ")}.` : "";
      const answer =
        `You said: **${text}**${notes}\n\n` +
        "This answer comes from the mock API. Try a message containing *save* to see a written-file chip, " +
        "or *fail* to see an error.\n\n```python\nprint('hello from Pawn')\n```";
      await sleep(300, signal);
      h.onAnswer?.(answer);
      const runId = uid();
      if (written.length) {
        h.onFiles?.(written, runId);
        this.publish(written, "agent", runId);
      }
      const reply: Message = {
        id: uid(),
        chat_id: chatId,
        seq: list.length,
        role: "assistant",
        content: answer,
        context: written.length ? { written_files: written } : {},
        run_id: runId,
        model: body.model ?? null,
        created_at: nowIso(),
      };
      list.push(reply);
      chat.last_message_at = nowIso();
      h.onMessage?.({ id: reply.id, seq: reply.seq, role: "assistant" });
      if (list.length === 2 && chat.title_auto !== false) {
        // Mimic background auto-naming after the first answer.
        await sleep(500, signal);
        const words = text.replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean).slice(0, 5);
        chat.title = words.map((w) => w[0].toUpperCase() + w.slice(1)).join(" ") || chat.title;
        if (chat.tags_auto !== false) chat.tags = cleanTags(["mock/" + (words[0] ?? "chat").toLowerCase()]);
        h.onChat?.({ id: chat.id, title: chat.title, tags: chat.tags });
      }
      h.onDone?.({ conversation: chat.conversation_key, run_id: runId });
    } catch (e) {
      if ((e as DOMException).name === "AbortError") return;
      throw e;
    }
  }

  async listFolders(): Promise<FolderTree> {
    await sleep(40);
    const counts = new Map<string | null, number>();
    for (const c of this.chats.values()) {
      if (c.archived) continue;
      counts.set(c.folder_id, (counts.get(c.folder_id) ?? 0) + 1);
    }
    const tree = (parent: string | null): FolderNode[] =>
      [...this.folders.values()]
        .filter((f) => f.parent_id === parent)
        .sort((a, b) => a.position - b.position || a.name.localeCompare(b.name))
        .map((f) => ({ ...f, chat_count: counts.get(f.id) ?? 0, children: tree(f.id) }));
    return { folders: tree(null), unfiled_count: counts.get(null) ?? 0 };
  }

  async createFolder(body: FolderCreate): Promise<Folder> {
    if (!body.name.trim()) throw new HttpError(422, "name is required");
    return { ...this.addFolder(body.name.trim(), body.parent_id ?? null, body.position ?? 0) };
  }

  private descendants(id: string): Set<string> {
    const out = new Set<string>();
    const stack = [id];
    while (stack.length) {
      const cur = stack.pop()!;
      for (const f of this.folders.values()) {
        if (f.parent_id === cur && !out.has(f.id)) {
          out.add(f.id);
          stack.push(f.id);
        }
      }
    }
    return out;
  }

  async patchFolder(id: string, body: FolderPatch): Promise<Folder> {
    const f = this.folders.get(id);
    if (!f) throw new HttpError(404, "Folder not found");
    if (body.name !== undefined) f.name = body.name.trim() || f.name;
    if (body.to_root) f.parent_id = null;
    else if (body.parent_id) {
      if (body.parent_id === id || this.descendants(id).has(body.parent_id))
        throw new HttpError(422, "Cannot move a folder into itself");
      f.parent_id = body.parent_id;
    }
    if (body.position !== undefined) f.position = body.position;
    f.updated_at = nowIso();
    return { ...f };
  }

  async deleteFolder(id: string): Promise<void> {
    const f = this.folders.get(id);
    if (!f) throw new HttpError(404, "Folder not found");
    for (const c of this.chats.values()) if (c.folder_id === id) c.folder_id = f.parent_id;
    for (const sub of this.folders.values()) if (sub.parent_id === id) sub.parent_id = f.parent_id;
    this.folders.delete(id);
  }

  async listTags(): Promise<TagTrees> {
    const chatTags = [...this.chats.values()].filter((c) => !c.archived).map((c) => c.tags);
    const fileTags = [...this.fileTags.values()];
    return {
      tree: buildTagTree([
        ...chatTags,
        ...this.clips.map((c) => c.tags),
        ...this.annotations.map((a) => a.tags),
        ...fileTags,
      ]),
      chats: buildTagTree(chatTags),
      files: buildTagTree(fileTags),
    };
  }

  async renameTag(from: string, to: string): Promise<number> {
    let changed = 0;
    const rewrite = (tags: string[]) => {
      let hit = false;
      const out = tags.map((t) => {
        if (t === from || t.startsWith(from + "/")) {
          hit = true;
          return to + t.slice(from.length);
        }
        return t;
      });
      if (hit) changed += 1;
      return cleanTags(out);
    };
    for (const c of this.chats.values()) c.tags = rewrite(c.tags);
    for (const c of this.clips) c.tags = rewrite(c.tags);
    for (const a of this.annotations) a.tags = rewrite(a.tags);
    for (const [k, t] of this.fileTags) this.fileTags.set(k, rewrite(t));
    return changed;
  }

  async search(q: string): Promise<SearchResults> {
    await sleep(80);
    const tags = q.split(/\s+/).filter((t) => t.startsWith("#") && t.length > 1).map((t) => t.slice(1));
    const needle = q
      .split(/\s+/)
      .filter((t) => !t.startsWith("#"))
      .join(" ")
      .trim()
      .toLowerCase();
    const tagged = (t: string[]) => tags.every((want) => tagMatches(t, want));
    const hit = (text: string) => !needle || text.toLowerCase().includes(needle);
    const snippet = (text: string) => {
      const idx = needle ? text.toLowerCase().indexOf(needle) : 0;
      const start = Math.max(0, idx - 60);
      return (start ? "…" : "") + text.slice(start, start + 180).replace(/\n/g, " ");
    };
    const msgs = needle
      ? [...this.messages.values()].flat().filter((m) => hit(m.content) && tagged(this.chats.get(m.chat_id)?.tags ?? []))
      : [];
    const files: FileHit[] = this.allKeys()
      .filter((k) => tagged(this.fileTags.get(k) ?? []))
      .filter((k) => hit(k) || hit(this.files.get(k)?.content ?? ""))
      .map((k) => ({
        key: k,
        title: k.split("/").pop() ?? k,
        kind: /\.md$/i.test(k) ? "markdown" : /\.(png|jpe?g|gif|webp)$/i.test(k) ? "image" : "other",
        snippet: snippet(this.files.get(k)?.content ?? ""),
        tags: this.fileTags.get(k) ?? [],
        modified: this.files.get(k)?.modified ?? this.blobModified.get(k) ?? null,
      }));
    return {
      query: needle,
      tags,
      chats: [...this.chats.values()].filter((c) => hit(c.title) && tagged(c.tags)),
      messages: msgs.map((m) => ({
        id: m.id,
        chat_id: m.chat_id,
        chat_title: this.chats.get(m.chat_id)?.title ?? "",
        role: m.role,
        seq: m.seq,
        snippet: snippet(m.content),
        created_at: m.created_at,
      })),
      files,
      clips: this.clips.filter((c) => (hit(c.title) || hit(c.content_md)) && tagged(c.tags)),
      annotations: this.annotations
        .filter((a) => (hit(a.body_md) || hit(a.anchor?.exact ?? "")) && tagged(a.tags))
        .map((a) => ({ ...a, context: this.annotationContext(a) })),
    };
  }

  async forkChat(chatId: string, messageId: string, title?: string): Promise<Fork> {
    await sleep(60);
    const src = this.chat(chatId);
    const msgs = this.messages.get(chatId) ?? [];
    const cut = msgs.find((m) => m.id === messageId);
    if (!cut) throw new HttpError(404, "Message not in this chat");
    const c = this.addChat(title?.trim() || `${src.title || "Chat"} (fork)`, src.folder_id, [...src.tags], 0);
    c.parent_chat_id = src.id;
    c.fork_from_message_id = cut.id;
    const kept = msgs.filter((m) => m.seq <= cut.seq);
    this.messages.set(
      c.id,
      kept.map((m, i) => ({ ...m, id: uid(), chat_id: c.id, seq: i, context: { ...m.context } })),
    );
    return { ...c, message_count: kept.length, sallm_seeded: kept.length };
  }

  private clip(id: string): Clip {
    const c = this.clips.find((x) => x.id === id);
    if (!c) throw new HttpError(404, "Clip not found");
    return c;
  }

  async getClip(id: string): Promise<Clip> {
    await sleep(30);
    return { ...this.clip(id) };
  }

  async createClip(body: ClipCreate): Promise<Clip> {
    await sleep(40);
    const all = [...this.messages.values()].flat();
    const msgs = (body.message_ids ?? []).map((id) => {
      const m = all.find((x) => x.id === id);
      if (!m) throw new HttpError(404, `Unknown messages: ${id}`);
      return m;
    });
    if (body.content_md === undefined && !msgs.length) {
      throw new HttpError(422, "message_ids or content_md required");
    }
    const content =
      body.content_md ??
      msgs.map((m) => `**${m.role === "user" ? "You" : "Pawn"}:**\n\n${m.content}`).join("\n\n");
    const clip: Clip = {
      id: uid(),
      chat_id: body.chat_id ?? msgs[0]?.chat_id ?? null,
      message_ids: msgs.map((m) => m.id),
      title: body.title?.trim() || content.replace(/[#*>_`]/g, "").trim().split("\n")[0].slice(0, 60) || "Clip",
      tags: cleanTags(body.tags),
      content_md: content,
      vault_key: null,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    this.clips.unshift(clip);
    return { ...clip };
  }

  async patchClip(id: string, body: ClipPatch): Promise<Clip> {
    await sleep(30);
    const c = this.clip(id);
    if (body.title !== undefined) c.title = body.title.trim();
    if (body.tags !== undefined) c.tags = cleanTags(body.tags);
    if (body.content_md !== undefined) c.content_md = body.content_md;
    c.updated_at = nowIso();
    return { ...c };
  }

  async deleteClip(id: string): Promise<void> {
    this.clip(id);
    this.clips = this.clips.filter((c) => c.id !== id);
  }

  async saveClip(id: string, path?: string): Promise<FileStat> {
    await sleep(40);
    const c = this.clip(id);
    const safe = (c.title || "Clip").replace(/[\\/:*?"<>|#^[\]]+/g, " ").trim() || "Clip";
    const key = path || `Pawn/Scratch/Clips/${safe}.md`;
    const tags = c.tags.length ? `tags:\n${c.tags.map((t) => `  - ${t}`).join("\n")}\n` : "";
    const body = `---\npawn: editable\nsource: portal-clip\nclip_id: ${c.id}\n${tags}---\n# ${c.title}\n\n${c.content_md}`;
    const etag = this.nextEtag();
    this.files.set(key, { content: body, etag, modified: nowIso() });
    c.vault_key = key;
    this.publish([key], "portal");
    return { key, etag };
  }

  async listClips(p: { tag?: string; chat_id?: string } = {}): Promise<Clip[]> {
    return this.clips.filter(
      (c) => (!p.chat_id || c.chat_id === p.chat_id) && (!p.tag || tagMatches(c.tags, p.tag)),
    );
  }

  private annotationContext(a: Annotation): Annotation["context"] {
    if (a.target_kind === "message") {
      const m = [...this.messages.values()].flat().find((x) => x.id === a.target_ref);
      const title = m ? (this.chats.get(m.chat_id)?.title ?? "") : "";
      return m ? { chat_id: m.chat_id, chat_title: title, label: title || "Chat" } : {};
    }
    if (a.target_kind === "clip") {
      const c = this.clips.find((x) => x.id === a.target_ref);
      return c ? { label: c.title || "Clip" } : {};
    }
    if (a.target_kind === "file") return { label: a.target_ref.split("/").pop() ?? a.target_ref };
    return {};
  }

  private annotation(id: string): Annotation {
    const a = this.annotations.find((x) => x.id === id);
    if (!a) throw new HttpError(404, "Annotation not found");
    return a;
  }

  async listAnnotations(p: AnnotationListParams = {}): Promise<Annotation[]> {
    await sleep(30);
    const refs = Array.isArray(p.target_ref) ? p.target_ref : p.target_ref ? [p.target_ref] : [];
    return this.annotations
      .filter(
        (a) =>
          (!p.target_kind || a.target_kind === p.target_kind) &&
          (!refs.length || refs.includes(a.target_ref)) &&
          (!p.tag || tagMatches(a.tags, p.tag)),
      )
      .map((a) => ({ ...a, tags: [...a.tags] }));
  }

  async createAnnotation(body: AnnotationCreate): Promise<Annotation> {
    await sleep(30);
    const kinds = ["message", "file", "item", "session", "clip", "capture"];
    if (!kinds.includes(body.target_kind)) throw new HttpError(422, `target_kind must be one of ${kinds.join(", ")}`);
    if (!body.target_ref?.trim()) throw new HttpError(422, "target_ref is required");
    const a: Annotation = {
      id: uid(),
      target_kind: body.target_kind,
      target_ref: body.target_ref.trim(),
      anchor: body.anchor ?? null,
      body_md: body.body_md ?? "",
      tags: cleanTags(body.tags),
      color: body.color ?? null,
      side_chat_id: body.side_chat_id ?? null,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    this.annotations.push(a);
    return { ...a };
  }

  async patchAnnotation(id: string, body: AnnotationPatch): Promise<Annotation> {
    await sleep(30);
    const a = this.annotation(id);
    if (body.anchor !== undefined) a.anchor = body.anchor;
    if (body.body_md !== undefined) a.body_md = body.body_md;
    if (body.tags !== undefined) a.tags = cleanTags(body.tags);
    if ("color" in body) a.color = body.color ?? null;
    if ("side_chat_id" in body) a.side_chat_id = body.side_chat_id ?? null;
    a.updated_at = nowIso();
    return { ...a, tags: [...a.tags] };
  }

  async deleteAnnotation(id: string): Promise<void> {
    this.annotation(id);
    this.annotations = this.annotations.filter((a) => a.id !== id);
  }

  private allKeys(): string[] {
    return [...new Set([...this.files.keys(), ...this.blobs.keys()])];
  }

  async listFiles(prefix = ""): Promise<FileListing> {
    await sleep(60);
    const base = prefix.replace(/^\/+|\/+$/g, "");
    const pre = base ? base + "/" : "";
    const folders = new Map<string, { count: number; modified: string }>();
    const files: string[] = [];
    for (const key of this.allKeys()) {
      if (!key.startsWith(pre)) continue;
      const rest = key.slice(pre.length);
      const modified = this.modifiedOf(key);
      if (rest.includes("/")) {
        const name = rest.split("/")[0];
        const cur = folders.get(name);
        folders.set(name, {
          count: (cur?.count ?? 0) + 1,
          modified: cur && cur.modified > modified ? cur.modified : modified,
        });
      } else files.push(key);
    }
    return {
      prefix: base,
      folders: [...folders.entries()]
        .sort((a, b) => a[0].toLowerCase().localeCompare(b[0].toLowerCase()))
        .map(([name, f]) => ({ name, path: pre + name, count: f.count, modified: f.modified })),
      files: files
        .sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
        .map((key) => ({
          name: key.slice(pre.length),
          path: key,
          modified: this.modifiedOf(key),
          tags: this.fileTags.get(key) ?? [],
        })),
      scratch_dir: SCRATCH,
      agent_root: "Pawn",
    };
  }

  private modifiedOf(key: string): string {
    return this.files.get(key)?.modified ?? this.blobModified.get(key) ?? minutesAgo(600);
  }

  async listAllFiles(prefix = ""): Promise<string[]> {
    return this.allKeys().filter((k) => k.startsWith(prefix));
  }

  async readFile(key: string): Promise<FileContent> {
    await sleep(60);
    const f = this.files.get(key);
    if (!f) throw new HttpError(404, `Not found: ${key}`);
    let writable = true;
    try {
      this.assertWritable(key, f.content);
    } catch {
      writable = false;
    }
    return {
      key,
      content: f.content,
      etag: f.etag,
      size: f.content.length,
      last_modified: f.modified,
      writable,
      tags: this.fileTags.get(key) ?? [],
    };
  }

  async readFileBlob(key: string): Promise<Blob> {
    const b = this.blobs.get(key);
    if (b) return b;
    const f = this.files.get(key);
    if (f) return new Blob([f.content], { type: "text/plain" });
    throw new HttpError(404, `Not found: ${key}`);
  }

  async writeFile(key: string, content: string, etag: string): Promise<FileStat> {
    await sleep(80);
    const cur = this.files.get(key);
    this.assertWritable(key, cur?.content);
    if (!etag && cur) throw new HttpError(409, `'${key}' already exists`);
    if (etag && (!cur || cur.etag !== etag)) throw new HttpError(409, `'${key}' changed since it was read`);
    const stat = this.writeKey(key, content);
    this.publish([key], "portal");
    return stat;
  }

  async moveFile(src: string, dst: string, overwrite = false): Promise<FileStat> {
    const f = this.files.get(src);
    const b = this.blobs.get(src);
    if (!f && !b) throw new HttpError(404, `Not found: ${src}`);
    this.assertWritable(src, f?.content);
    this.assertWritable(dst);
    if (!overwrite && (this.files.has(dst) || this.blobs.has(dst))) throw new HttpError(409, `'${dst}' already exists`);
    let stat: FileStat;
    if (f) {
      this.files.delete(src);
      stat = this.writeKey(dst, f.content);
    } else {
      this.blobs.delete(src);
      this.blobs.set(dst, b!);
      this.blobModified.set(dst, nowIso());
      stat = { key: dst, etag: this.nextEtag() };
    }
    const tags = this.fileTags.get(src);
    this.fileTags.delete(src);
    if (tags) this.fileTags.set(dst, tags);
    this.publish([src, dst], "portal");
    return stat;
  }

  async deleteFile(key: string): Promise<void> {
    const f = this.files.get(key);
    if (!f && !this.blobs.has(key)) throw new HttpError(404, `Not found: ${key}`);
    this.assertWritable(key, f?.content);
    this.files.delete(key);
    this.blobs.delete(key);
    this.fileTags.delete(key);
    this.publish([key], "portal");
  }

  async uploadFile(file: File, folder = ""): Promise<FileStat> {
    const base = folder || SCRATCH;
    const key = `${base}/${file.name.replace(/\s+/g, "_")}`;
    this.assertWritable(key);
    let stat: FileStat;
    if (/\.(md|txt|json|ya?ml|csv)$/i.test(file.name)) stat = this.writeKey(key, await file.text());
    else {
      this.blobs.set(key, file);
      this.blobModified.set(key, nowIso());
      stat = { key, etag: this.nextEtag(), size: file.size };
    }
    this.publish([key], "portal");
    return stat;
  }

  async getFileTags(key: string): Promise<FileTags> {
    if (!this.files.has(key) && !this.blobs.has(key)) throw new HttpError(404, `Not found: ${key}`);
    const tags = this.fileTags.get(key) ?? [];
    let source: FileTags["source"] = "db";
    try {
      if (/\.md$/i.test(key)) {
        this.assertWritable(key, this.files.get(key)?.content);
        source = "frontmatter";
      }
    } catch {
      source = "db";
    }
    return { key, tags, frontmatter: source === "frontmatter" ? tags : [], stored: source === "db" ? tags : [], source };
  }

  async setFileTags(key: string, tags: string[]): Promise<FileTags> {
    await sleep(60);
    const clean = cleanTags(tags);
    if (clean.length) this.fileTags.set(key, clean);
    else this.fileTags.delete(key);
    return this.getFileTags(key);
  }

  async reindexFiles(): Promise<ReindexResult> {
    return { files: this.allKeys().length, indexed: 0, removed: 0 };
  }

  async getWorkspace(): Promise<Workspace> {
    return this.workspace;
  }

  async putWorkspace(layout: Record<string, unknown>, name = "default"): Promise<Workspace> {
    this.workspace = { name, layout, updated_at: nowIso() };
    return this.workspace;
  }

  async listModels(): Promise<ModelCatalog> {
    return {
      default: "ollama@gemma4:4b",
      background: "ollama@gemma4:4b",
      models: [
        { id: "ollama@gemma4:4b", provider: "ollama", model: "gemma4:4b", vision: true },
        { id: "openrouter@qwen3-235b", provider: "openrouter", model: "qwen3-235b" },
      ],
    };
  }

  async vaultEvents(since: number, timeout: number, signal?: AbortSignal): Promise<VaultEvents> {
    const snap = (): VaultEvents => ({
      seq: this.eventSeq,
      resync: false,
      events: this.events.filter((e) => e.seq > since),
    });
    const now = snap();
    if (now.events.length || timeout <= 0) return now;
    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, timeout * 1000);
      this.waiters.push(() => {
        clearTimeout(t);
        resolve();
      });
      signal?.addEventListener("abort", () => {
        clearTimeout(t);
        reject(new DOMException("Aborted", "AbortError"));
      });
    });
    return snap();
  }

  // ── inbox ───────────────────────────────────────────────────────────────

  private openIdeas(): Idea[] {
    return this.ideasWith("inbox");
  }

  private ideasWith(status: string): Idea[] {
    const out: Idea[] = [];
    for (const [key, f] of this.files) {
      if (!key.startsWith(`${IDEAS}/`) || !key.endsWith(".md")) continue;
      const meta = frontmatter(f.content);
      if (status !== "all" && meta.status !== status) continue;
      const body = f.content.replace(/^---\n[\s\S]*?\n---\n/, "").trim();
      const lines = body.split("\n");
      const line = lines[0]?.startsWith("#") ? lines.slice(1).join("\n").trim() : body;
      out.push({ path: key, title: headingOf(body, stemOf(key)), line, status: meta.status ?? "", created: stemOf(key).slice(0, 10) });
    }
    return out.sort((a, b) => b.created.localeCompare(a.created) || b.path.localeCompare(a.path));
  }

  private atoms(scope: "inbox" | "all"): CaptureAtom[] {
    const out: CaptureAtom[] = [];
    for (const [key, f] of this.files) {
      if (!key.startsWith(`${CAPTURE_INBOX}/`)) continue;
      const m = frontmatter(f.content);
      if (scope !== "all" && !["inbox", "proposed"].includes(m.status ?? "")) continue;
      out.push({
        path: key,
        title: headingOf(f.content, stemOf(key)),
        status: m.status ?? "",
        collection: m.collection ?? "",
        entity: m.entity ?? "",
        hint: m.hint ?? "",
        type: m.type ?? "",
        snippet_kind: m.kind ?? "",
        source_url: m.source_url ?? "",
        caption: m.caption ?? "",
        created: m.created ?? "",
      });
    }
    return out.sort((a, b) => b.created.localeCompare(a.created));
  }

  async inboxCounts(): Promise<InboxCounts> {
    return {
      items: this.items.filter((i) => OPEN_ITEM.includes(i.status)).length,
      ideas: this.openIdeas().length,
      captures: this.atoms("inbox").length,
    };
  }

  async listItems(p: ItemListParams = {}): Promise<ItemsPage> {
    await sleep(80);
    const statuses = !p.statuses ? OPEN_ITEM : p.statuses === "all" ? null : p.statuses.split(",");
    const q = (p.q ?? "").toLowerCase();
    const rows = this.items
      .filter((i) => !statuses || statuses.includes(i.status))
      .filter((i) => !p.kind || i.kind === p.kind)
      .filter((i) => !p.source_ref || i.source_ref === p.source_ref)
      .filter((i) => !q || i.text.toLowerCase().includes(q))
      .sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
    const start = p.offset ?? 0;
    return { items: rows.slice(start, start + (p.limit ?? 100)), total: rows.length };
  }

  async itemAction(id: string, action: ItemAction, arg?: string): Promise<ItemActionResult> {
    const item = this.items.find((i) => i.id === id);
    if (!item) throw new HttpError(404, `No item ${id}.`);
    const next: Record<ItemAction, string> = {
      file: "filed",
      task: "task",
      todo: "task",
      later: "snoozed",
      ignore: "dismissed",
      approve: "filed",
      reject: "dismissed",
      delete: "dismissed",
    };
    item.status = next[action];
    if (action === "later") item.snooze_until = arg || minutesAgo(-60 * 24);
    item.updated_at = nowIso();
    if (item.note_key) this.publish([item.note_key], "agent");
    const verb = action === "later" ? `Snoozed ${item.short_id}` : `${action[0].toUpperCase()}${action.slice(1)} ${item.short_id}`;
    return { receipt: `${verb}.`, item: { ...item } };
  }

  async deleteItems(body: ItemsDeleteBody): Promise<{ deleted: number; receipts: string[] }> {
    const ids = body.all_open
      ? this.items.filter((i) => OPEN_ITEM.includes(i.status) && (!body.kind || i.kind === body.kind)).map((i) => i.id)
      : (body.ids ?? []);
    for (const id of ids) {
      const it = this.items.find((i) => i.id === id);
      if (it) it.status = "dismissed";
    }
    this.publish(["Pawn/Items/"], "agent");
    return { deleted: ids.length, receipts: ids.map((id) => `Deleted ${id}.`) };
  }

  async listIdeas(status = "inbox"): Promise<Idea[]> {
    return this.ideasWith(status);
  }

  async captureIdea(line: string): Promise<string> {
    const text = line.trim();
    if (!text) throw new HttpError(422, "/idea needs one line.");
    const title = text.replace(/[\\/:*?"<>|]/g, "").slice(0, 60).trim();
    const key = `${IDEAS}/${nowIso().slice(0, 10)} ${title}.md`;
    this.writeKey(key, `---\ntags: [idea]\nstatus: inbox\n---\n\n# ${title}\n\n${text}\n`);
    this.publish([key], "portal");
    return `Captured ${key}`;
  }

  async ideaAction(path: string, action: IdeaActionKind, title?: string): Promise<IdeaActionResult> {
    const f = this.files.get(path);
    if (!f) throw new HttpError(404, `Not found: ${path}`);
    const status = { keep: "later", goal: "goal", task: "task", drop: "dropped" }[action];
    const name = (title || headingOf(f.content, stemOf(path))).trim();
    let written: string | null = null;
    let receipt = action === "keep" ? "Kept for later." : "Dropped.";
    if (action === "goal") {
      written = "Goals.md";
      const prev = this.files.get(written)?.content ?? "---\npawn: goals\n---\n\n## Parked\n";
      this.writeKey(written, `${prev.trimEnd()}\n\n### ${name}\n- note: [[${path.replace(/\.md$/, "")}]]\n`);
      receipt = `Parked '${name}' in Goals.md.`;
    } else if (action === "task") {
      written = `Pawn/TaskNotes/Tasks/${name}.md`;
      this.writeKey(written, `---\ntags: [task]\ntitle: "${name}"\nstatus: open\n---\n\nSource: [[${path.replace(/\.md$/, "")}]]\n`);
      receipt = `Task ${written}.`;
    }
    this.writeKey(path, setStatus(f.content, status));
    this.publish(written ? [path, written] : [path], "portal");
    return { receipt, path, written, status };
  }

  async listCaptureAtoms(scope: "inbox" | "all" = "inbox"): Promise<CaptureAtom[]> {
    return this.atoms(scope);
  }

  async listCaptureCollections(): Promise<CaptureCollection[]> {
    return [
      { id: "books", title: "Books" },
      { id: "programming", title: "Programming" },
    ];
  }

  async fileCapture(body: CaptureFileBody): Promise<Record<string, unknown>> {
    const f = this.files.get(body.path);
    if (!f) throw new HttpError(404, `Not found: ${body.path}`);
    this.writeKey(body.path, setStatus(f.content, body.ignore ? "ignored" : "filed"));
    this.publish([body.path], "portal");
    return { path: body.path, status: body.ignore ? "ignored" : "filed" };
  }

  // ── settings ────────────────────────────────────────────────────────────

  private settingsPayload(): SettingsPayload {
    const fields: SettingField[] = MOCK_FIELDS.map((f) => {
      const over = f.path in this.overrides;
      const value = over ? this.overrides[f.path] : MOCK_YAML[f.path];
      const running = this.running[f.path];
      const row: SettingField = {
        ...f,
        value: f.secret ? null : value,
        running: f.secret ? null : running,
        source: over ? "override" : "yaml",
        yaml_value: f.secret ? null : MOCK_YAML[f.path],
        pending_restart: f.apply === "restart" && JSON.stringify(value) !== JSON.stringify(running),
      };
      if (f.secret) row.set = Boolean(value);
      if (f.type === "cron") {
        row.next = /^\S+ \S+ \S+ \S+ \S+$/.test(String(value))
          ? [1, 2, 3].map((d) => new Date(Date.now() + d * 86_400_000).toISOString())
          : [];
      }
      return row;
    });
    return {
      groups: [
        { id: "models", label: "Models" },
        { id: "agent", label: "Agent" },
        { id: "coworker", label: "Coworker" },
        { id: "capture", label: "Capture & research" },
        { id: "matrix", label: "Matrix bot" },
        { id: "portal", label: "Portal" },
      ],
      fields,
      catalog: mockCatalog(),
      override_file: "/home/me/.sallm/settings.override.yaml",
      config_file: "/home/me/pawn/pawnai.yaml",
      restart_required: fields.filter((f) => f.pending_restart).map((f) => f.path),
    };
  }

  async getSettings(): Promise<SettingsPayload> {
    await sleep(80);
    return this.settingsPayload();
  }

  async patchSettings(values: Record<string, unknown>): Promise<SettingsPayload> {
    const errors: Record<string, string> = {};
    for (const [path, value] of Object.entries(values)) {
      const f = MOCK_FIELDS.find((x) => x.path === path);
      if (!f) errors[path] = "Not an editable setting";
      else if (f.type === "cron" && !/^\S+ \S+ \S+ \S+ \S+$/.test(String(value))) errors[path] = `Not a valid cron expression: '${value}'`;
      else if (f.type === "int" && (!Number.isInteger(value) || (f.min !== null && Number(value) < f.min))) errors[path] = `Must be at least ${f.min ?? 0}`;
      else if (f.type === "enum" && !f.choices?.includes(String(value))) errors[path] = `Choose one of: ${f.choices?.join(", ")}`;
    }
    if (Object.keys(errors).length) throw new HttpError(422, JSON.stringify({ errors }));
    for (const [path, value] of Object.entries(values)) {
      const f = MOCK_FIELDS.find((x) => x.path === path)!;
      if (f.secret && !value) continue;
      this.overrides[path] = value;
      if (f.apply !== "restart") this.running[path] = value;
    }
    return this.settingsPayload();
  }

  async resetSetting(path: string): Promise<SettingsPayload> {
    const f = MOCK_FIELDS.find((x) => x.path === path);
    if (!f) throw new HttpError(422, JSON.stringify({ errors: { [path]: "Not an editable setting" } }));
    delete this.overrides[path];
    if (f.apply !== "restart") this.running[path] = MOCK_YAML[path];
    return this.settingsPayload();
  }

  async settingsStatus(): Promise<SettingsStatus> {
    await sleep(120);
    return {
      version: "1.0.0",
      config_file: "/home/me/pawn/pawnai.yaml",
      override_file: "/home/me/.sallm/settings.override.yaml",
      state_dir: "/home/me/.sallm",
      db: { reachable: true, revision: "0027", head: "0027" },
      queues: [
        { name: "agent", source: "agent_queue", topic: "agent", bucket: "pawn-queue", messages: 0, leases: 0, dead_letters: 0, paused: false },
        { name: "diarize", source: "diarize_queue", topic: "diarize", bucket: "pawn-queue", messages: 2, leases: 1, dead_letters: 0, paused: false },
      ],
    };
  }

  /** Dev helper: simulate an agent write (window.pawnMock.agentWrite()). */
  agentWrite(key = `Pawn/Analyses/Mock ${Date.now()}.md`): void {
    this.writeKey(key, `# Agent output\n\nWritten at ${nowIso()}.\n`);
    this.publish([key], "agent", uid());
  }
}

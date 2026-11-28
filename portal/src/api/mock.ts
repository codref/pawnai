// In-browser mock of the portal API, enabled with VITE_PAWN_MOCK=1.
// State lives in memory only; a reload starts from the seed data again.

import { applyByteRange, silentWav, wavBlob } from "@/lib/wav";
import { MockBrainmap, mockObservationId } from "./mockBrainmap";
import { MockBoards } from "./mockBoards";
import {
  HttpError,
  type Annotation,
  type Board,
  type BoardCreate,
  type BoardPatch,
  type BoardProposal,
  type BrainEdge,
  type BrainGraph,
  type BrainGraphParams,
  type BrainLassoBody,
  type BrainLassoResult,
  type BrainLayout,
  type BrainNode,
  type BrainTieResult,
  type EmergenceStatus,
  type EmergenceSubject,
  type EmergenceSubjectDetail,
  type AssistantCommand,
  type AssistantCommandResult,
  type AssistantKind,
  type AssistantRegistration,
  type AssistantSummary,
  type AssistantsOverview,
  type VoiceThread,
  type WakeModel,
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
  type SchedulerDiagnostics,
  type ConfigDiagnostics,
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
  type BulkReidentifyBody,
  type GallerySpeaker,
  type Job,
  type ListSessionsParams,
  type PlannerBlock,
  type PlannerDay,
  type PlannerEvent,
  type PlannerEventCreate,
  type PlannerEventPatch,
  type PlannerRange,
  type SessionAudioUrls,
  type SessionDetail,
  type SessionEnrollBody,
  type SessionExtractResult,
  type SessionLabel,
  type SessionReprocessBody,
  type SessionSummary,
  type SessionTranscript,
  type SessionsPage,
  type SpeakerDetail,
  type SpeakerEnrollmentRow,
  type SpeakerMergeResult,
  type SpeakerPatch,
  type SpeakerPeopleResult,
  type SpeakerSummary,
} from "./types";

const SCRATCH = "Pawn/Scratch";

let counter = 0;
const uid = () => `m${Date.now().toString(36)}${(counter++).toString(36)}`;
const nowIso = () => new Date().toISOString();
const minutesAgo = (m: number) => new Date(Date.now() - m * 60_000).toISOString();

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function localDay(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

function atLocal(dayOffset: number, hour: number, minute: number): string {
  const d = new Date();
  d.setDate(d.getDate() + dayOffset);
  d.setHours(hour, minute, 0, 0);
  return d.toISOString();
}

function dayStamp(offset = 0): string {
  return localDay(atLocal(offset, 12, 0));
}

function eachDay(from: string, to: string): string[] {
  const out: string[] = [];
  const cur = new Date(`${from}T12:00:00`);
  const end = new Date(`${to}T12:00:00`);
  if (Number.isNaN(cur.getTime()) || Number.isNaN(end.getTime()) || cur > end) return out;
  while (cur <= end && out.length < 15) {
    out.push(localDay(cur.toISOString()));
    cur.setDate(cur.getDate() + 1);
  }
  return out;
}

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

/** Item payload pointing at its mock observation (see mockBrainmap.ts). */
function ob(i: number, startSec: number): Record<string, unknown> {
  return { observation_id: mockObservationId(i), start_sec: startSec };
}

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
  { path: "emergence.enabled", label: "Emergence enabled", group: "emergence", type: "bool", help: "Run the observation layer and the night shift.", apply: "restart", secret: false, choices: null, nullable: false, tz_path: null, min: null },
  { path: "emergence.fast_model", label: "Per-window model", group: "emergence", type: "model", help: "Empty uses the background model.", apply: "live", secret: false, choices: null, nullable: true, tz_path: null, min: null },
  { path: "emergence.triage_threshold", label: "Triage threshold", group: "emergence", type: "float", help: "A type is extracted when triage scores it at least this (0-1).", apply: "live", secret: false, choices: null, nullable: false, tz_path: null, min: 0 },
  { path: "emergence.night_cron", label: "Night shift", group: "emergence", type: "cron", help: "Cron for the night jobs.", apply: "restart", secret: false, choices: null, nullable: false, tz_path: "coworker.timezone", min: null },
  { path: "emergence.night_jobs", label: "Night jobs", group: "emergence", type: "list", help: "Jobs the night shift runs, in priority order.", apply: "live", secret: false, choices: null, nullable: false, tz_path: null, min: null },
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
  "emergence.enabled": true,
  "emergence.fast_model": "",
  "emergence.triage_threshold": 0.35,
  "emergence.night_cron": "30 1 * * *",
  "emergence.night_jobs": ["sync", "clustering", "synthesis", "dreaming", "provocations"],
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
    seedItem(1, "commitment", "Davide drafts the Q4 launch plan", { owner: "Davide", due: "2026-12-05", quote: "I'll have a draft by next Friday.", payload: ob(1, 612) }),
    seedItem(2, "decision", "Q4 launch is confirmed for 1 December", { reason: "Matches goal: Acme launch", interrupt: true, thread: "Acme launch", payload: ob(2, 1340) }),
    seedItem(3, "open_question", "Who signs off the Acme budget?", { owner: "Ana", payload: ob(3, 1802) }),
    seedItem(4, "proposal", "Research competitor pricing for the launch", { payload: ob(4, 2400) }),
    seedItem(5, "block", "Design review is waiting on legal", { status: "snoozed", snooze_until: minutesAgo(-60 * 24), payload: ob(5, 980) }),
    seedItem(6, "commitment", "Send the retro notes", { status: "filed" }),
    seedItem(7, "commitment", "Ana sends the revised budget to finance", { owner: "Ana", due: "2026-12-03", quote: "Finance gets the new numbers Wednesday.", payload: ob(7, 1520) }),
    seedItem(8, "block", "Load testing is blocked until staging is upgraded", { owner: "Luca", quote: "We can't load test on that staging box.", source_ref: "platform-1on1", payload: ob(8, 455) }),
    seedItem(9, "open_question", "Is 1 December realistic given the legal review?", { payload: ob(9, 2105) }),
    seedItem(10, "contradiction", "Marco says Q2 at the earliest; the plan says December", { owner: "Marco", quote: "Honestly, Q2 at the earliest.", source_ref: "platform-1on1", payload: ob(10, 1022) }),
    seedItem(11, "decision", "Freeze the public API on Monday", { owner: "Mira", quote: "We freeze Monday, assuming team B is ready.", source_ref: "platform-1on1", payload: ob(11, 1460) }),
    seedItem(12, "commitment", "Mira shares the migration runbook", { owner: "Mira", due: "2026-12-02", source_ref: "platform-1on1", payload: ob(12, 1610) }),
    seedItem(13, "open_question", "Who owns on-call during launch week?", { payload: ob(13, 2650) }),
    seedItem(14, "commitment", "Davide books the vendor contract review", { owner: "Davide", quote: "I'll get legal and the vendor in a room.", payload: ob(14, 3020) }),
    seedItem(15, "block", "Vendor contract is still unsigned", { quote: "Still no signature from their side.", payload: ob(15, 3105) }),
    seedItem(16, "decision", "Drop the beta waitlist for launch", { owner: "Ana", payload: ob(16, 2290) }),
    seedItem(17, "proposal", "Compare vendor SLA terms before signing", { payload: ob(17, 3200) }),
    seedItem(18, "commitment", "Luca writes the load-test plan", { owner: "Luca", due: "2026-12-08", source_ref: "platform-1on1", payload: ob(18, 720) }),
    seedItem(19, "open_question", "Does finance need the forecast before the 1st?", { owner: "Ana", payload: ob(19, 1940) }),
    seedItem(20, "contradiction", "Two budget figures were quoted: 40k and 55k", { quote: "It was forty — no, fifty-five.", payload: ob(20, 1710) }),
  ];
  private brain = new MockBrainmap(() => this.items);
  private boards = new MockBoards(() => this.brain.graph());
  private overrides: Record<string, unknown> = {};
  private running: Record<string, unknown> = { ...MOCK_YAML };
  private sessionRows: SessionDetail[] = [];
  private transcripts = new Map<string, SessionTranscript>();
  private plannerEvents: PlannerEvent[] = [];
  private gallery: SpeakerDetail[] = [];
  private jobRows = new Map<string, Job>();
  private jobListeners = new Set<(job: Job) => void>();
  private wav: Uint8Array | null = null;
  private assistants: AssistantSummary[] = [];
  private wakeModels: WakeModel[] = [];

  constructor() {
    this.seed();
    this.seedAssistants();
  }

  private seedAssistants(): void {
    const ago = (min: number) => new Date(Date.now() - min * 60_000).toISOString();
    const thread = (device: string, name: string): VoiceThread => ({
      conversation: "voice:4f1c2a9b0d3e",
      device_id: device,
      device_name: name,
      access: "chat",
      lang: "it",
      state: "active",
      turns: 4,
      last_user: "che cosa ho domani mattina?",
      last_answer: "Alle nove hai la riunione con Sean, poi niente fino a pranzo.",
      portal_chat_id: null,
      started_at: ago(12),
      last_turn_at: ago(3),
    });
    this.assistants = [
      {
        device_id: "dev-laptop",
        name: "laptop",
        kind: "desktop",
        platform: "linux",
        enabled_categories: [],
        capabilities: ["speak", "notify", "timer_set"],
        unifiedpush: false,
        created_at: ago(60 * 24 * 3),
        last_seen: ago(1),
        presence: { state: "active", last_wake_at: ago(3) },
        presence_at: ago(1),
        thread: thread("dev-laptop", "laptop"),
        recent_threads: [thread("dev-laptop", "laptop")],
        commands: [
          {
            id: "c1",
            tool: "timer_set",
            args: { seconds: "600", label: "pasta" },
            action: false,
            status: "ok",
            result: {},
            created_at: ago(40),
            expires_at: ago(39),
            completed_at: ago(40),
          },
        ],
        access_requests: [
          {
            id: "ar1",
            requester_device_id: "dev-laptop",
            approver_device_id: "dev-pixel",
            approver_name: "Pixel 9",
            conversation: "voice:4f1c2a9b0d3e",
            minutes: 15,
            status: "approved",
            message: null,
            created_at: ago(10),
            expires_at: ago(9),
            access_until: ago(-5),
          },
        ],
        wake_clips: { confirmed: 42, false_wake: 7, barge: 3 },
      },
      {
        device_id: "dev-pixel",
        name: "Pixel 9",
        kind: "phone",
        platform: "android",
        enabled_categories: ["call", "sms", "status"],
        capabilities: ["location_now", "status_now", "notify", "sms_send", "speak", "timer_set", "access_approve"],
        unifiedpush: true,
        created_at: ago(60 * 24 * 20),
        last_seen: ago(14),
        presence: { state: "idle", headset: false },
        presence_at: ago(14),
        thread: null,
        recent_threads: [],
        commands: [
          {
            id: "c2",
            tool: "access_approve",
            args: { requester: "laptop", minutes: "15" },
            action: true,
            status: "ok",
            result: {},
            created_at: ago(10),
            expires_at: ago(9),
            completed_at: ago(10),
          },
        ],
        access_requests: [],
        wake_clips: { confirmed: 18, false_wake: 11, near_miss: 2 },
      },
    ];
    this.wakeModels = [
      {
        version: "v20261120-2130",
        name: "hey_pawn",
        sha256: "9f2c".padEnd(64, "0"),
        size: 812_000,
        notes: "first model with phone clips",
        thresholds: { wake: 0.55, barge_in: 0.65 },
        active: true,
        created_at: ago(60 * 24 * 7),
        url: "/v1/assistants/wake/models/v20261120-2130",
      },
    ];
  }

  private requireAssistant(id: string): AssistantSummary {
    const row = this.assistants.find((a) => a.device_id === id);
    if (!row) throw new HttpError(404, "Unknown device");
    return row;
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
    this.seedFoundations();
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
      const origin = (chat.origin ?? {}) as Record<string, unknown>;
      const boardName = (origin.board as string) || (origin.kind === "board" ? (origin.ref as string) : "");
      if (boardName) {
        const sel = body.board_selection ?? [];
        const p = this.boards.propose(boardName, "card", {
          title: `What links ${sel.length || "these"} card(s)`,
          body: `Mock conclusion for: ${text}`,
          sources: sel,
          reason: "Mock board chat",
        });
        reply.context = { ...reply.context, board_proposals: [p.id] };
        h.onBoard?.({ board: boardName, proposals: [p] });
      }
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
    const sessions = this.sessionRows
      .filter((row) => !tags.length && (hit(row.title) || hit(row.summary ?? "") || hit(row.session_id)))
      .map((row) => ({
        session_id: row.session_id,
        title: row.title,
        snippet: snippet(row.summary || row.title),
        created_at: row.created_at,
      }));
    return {
      query: needle,
      tags,
      chats: [...this.chats.values()].filter((c) => hit(c.title) && tagged(c.tags)),
      sessions,
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
    const kinds = ["message", "file", "item", "session", "clip", "capture", "event"];
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
        { id: "emergence", label: "Emergence" },
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

  async settingsScheduler(): Promise<SchedulerDiagnostics> {
    await sleep(120);
    const ago = (m: number) => new Date(Date.now() - m * 60_000).toISOString();
    const ahead = (m: number) => new Date(Date.now() + m * 60_000).toISOString();
    const days = (h: number) => [0, 1, 2].map((d) => ahead(h * 60 + d * 1440));
    return {
      now: new Date().toISOString(),
      config: { enabled: true, poll_interval_seconds: 30, max_due_per_tick: 5, default_timezone: "Europe/Rome", stale_fire_after_seconds: 3600, allow_agent_auto_apply: false },
      workers: {
        scheduler: { running: true, started_at: ago(300), last_tick_at: ago(0.3), ticks: 600, error: null, poll_interval_seconds: 30, max_due_per_tick: 5, fired: 4, last_claimed: 0, last_fire_at: ago(42) },
        coworker: { running: true, started_at: ago(300), last_tick_at: ago(0.5), ticks: 290, error: null, timezone: "Europe/Rome", last: { night: { at: ago(600), ok: true, result: { sync: "ok", clustering: "2 threads", _calls: 81 } } } },
      },
      crons: [
        { id: "briefing", label: "Morning briefing", cron: "0 8 * * *", timezone: "Europe/Rome", enabled: true, path: "coworker.briefing_cron", gate: "coworker.enabled", next: days(9), running_next: ahead(540), last: { at: ago(900), ok: true, result: null } },
        { id: "weekly", label: "Weekly review", cron: "0 17 * * 5", timezone: "Europe/Rome", enabled: true, path: "coworker.weekly_cron", gate: "coworker.enabled", next: [ahead(3000), ahead(13080), ahead(23160)], running_next: ahead(3000), last: null },
        { id: "night", label: "Emergence night shift", cron: "30 1 * * *", timezone: "Europe/Rome", enabled: true, path: "emergence.night_cron", gate: "emergence.enabled", next: days(14), running_next: ahead(840), last: { at: ago(600), ok: true, result: { sync: "ok", _calls: 81 } } },
      ],
      schedules: [
        { id: "s1", name: "Daily inbox digest", prompt: "Summarise yesterday's open items.", session_id: "chat:digest", model: null, status: "active", schedule_kind: "cron", timezone: "Europe/Rome", run_at: null, interval_seconds: null, cron_expression: "0 9 * * 1-5", next_run_at: ahead(600), last_run_at: ago(840), created_by: "user", output_note: "Pawn/Digest.md" },
        { id: "s2", name: "Check vendor contract", prompt: "Ask whether the vendor contract is signed.", session_id: "chat:vendor", model: "ollama@gemma4:4b", status: "paused", schedule_kind: "interval", timezone: "UTC", run_at: null, interval_seconds: 86400, cron_expression: null, next_run_at: ahead(120), last_run_at: ago(1320), created_by: "agent", output_note: null },
      ],
      fires: [
        { id: "f3", schedule_id: "s1", schedule_name: "Daily inbox digest", scheduled_for: ago(42), status: "completed", attempt: 1, claimed_at: ago(42), started_at: ago(42), completed_at: ago(41), duration_s: 38.2, error: null, agent_run_id: "r3", run_status: "done" },
        { id: "f2", schedule_id: "s2", schedule_name: "Check vendor contract", scheduled_for: ago(1320), status: "failed", attempt: 1, claimed_at: ago(1320), started_at: ago(1320), completed_at: ago(1319), duration_s: 4.1, error: "Model ollama@gemma4:4b timed out", agent_run_id: "r2", run_status: "failed" },
      ],
      fire_counts_24h: { completed: 3, failed: 1 },
      proposals: [{ id: "p1", action: "create", schedule_id: null, proposed_payload: { name: "Friday wrap-up", cron_expression: "0 16 * * 5" }, rationale: "You asked for a weekly wrap-up.", status: "proposed", created_at: ago(200) }],
      runs: [
        { id: "r3", source: "schedule", command: "run", session_id: "chat:digest", schedule_id: "s1", model: "ollama@gemma4:4b", status: "done", error: null, prompt: "Summarise yesterday's open items.", created_at: ago(42), started_at: ago(42), completed_at: ago(41), duration_s: 38.2 },
        { id: "r4", source: "emergence", command: "night", session_id: "emergence:night", schedule_id: null, model: "emergence", status: "done", error: null, prompt: "sync,clustering,synthesis", created_at: ago(600), started_at: ago(600), completed_at: ago(560), duration_s: 2400 },
        { id: "r2", source: "schedule", command: "run", session_id: "chat:vendor", schedule_id: "s2", model: "ollama@gemma4:4b", status: "failed", error: "Model ollama@gemma4:4b timed out", prompt: "Ask whether the vendor contract is signed.", created_at: ago(1320), started_at: ago(1320), completed_at: ago(1319), duration_s: 4.1 },
      ],
      last_night_run: { id: "r4", status: "done", created_at: ago(600), completed_at: ago(560), error: null },
    };
  }

  async settingsConfig(): Promise<ConfigDiagnostics> {
    await sleep(80);
    const config: Record<string, unknown> = {};
    for (const [path, value] of Object.entries({ ...MOCK_YAML, ...this.overrides })) {
      const parts = path.split(".");
      let cur = config;
      for (const p of parts.slice(0, -1)) cur = (cur[p] ??= {}) as Record<string, unknown>;
      cur[parts[parts.length - 1]] = path.endsWith("token") ? "••••••" : value;
    }
    config.db_dsn = "postgresql://pawn:••••••@localhost:5433/pawn";
    config.agent_scheduler = { enabled: true, poll_interval_seconds: 30, max_due_per_tick: 5, default_timezone: "Europe/Rome" };
    return {
      config,
      overrides: { ...this.overrides },
      editable: Object.fromEntries(MOCK_FIELDS.map((f) => [f.path, f.group])),
      config_file: "/home/me/pawn/pawnai.yaml",
      override_file: "/home/me/.sallm/settings.override.yaml",
    };
  }

  // ── sessions / day / speakers ─────────────────────────────────────────────

  private seedFoundations(): void {
    const today = dayStamp(0);
    const ada: SpeakerDetail = {
      id: "ada",
      display_name: "Ada",
      aliases: ["A. Lovelace"],
      active: true,
      enrollments: 2,
      seconds: 95,
      sessions: 4,
      last_seen_session: "weekly-sync",
      last_seen_at: atLocal(0, 10, 2),
      people_path: "People/ada.md",
      notes: "Leads the Acme account.",
      enrollment_rows: [
        { id: "enr-ada-1", source_session_id: "weekly-sync", start_time: 12, end_time: 48, duration: 36, quality_score: 0.91, approved_at: minutesAgo(60 * 24) },
        { id: "enr-ada-2", source_session_id: "acme-kickoff", start_time: 40, end_time: 99, duration: 59, quality_score: 0.88, approved_at: minutesAgo(60 * 48) },
      ],
      recent_sessions: [
        { session_id: "weekly-sync", title: "Product weekly", created_at: atLocal(0, 10, 2) },
        { session_id: "acme-kickoff", title: "Acme kickoff", created_at: atLocal(-1, 15, 40) },
      ],
    };
    const bob: SpeakerDetail = {
      id: "bob",
      display_name: "Bob",
      aliases: [],
      active: true,
      enrollments: 1,
      seconds: 40,
      sessions: 2,
      last_seen_session: "acme-kickoff",
      last_seen_at: atLocal(-1, 15, 40),
      people_path: "People/bob.md",
      notes: null,
      enrollment_rows: [
        { id: "enr-bob-1", source_session_id: "acme-kickoff", start_time: 100, end_time: 140, duration: 40, quality_score: 0.8, approved_at: minutesAgo(60 * 30) },
      ],
      recent_sessions: [{ session_id: "acme-kickoff", title: "Acme kickoff", created_at: atLocal(-1, 15, 40) }],
    };
    this.gallery = [ada, bob];
    this.writeKey("People/ada.md", "---\npawn: person\nspeaker_id: ada\n---\n\n# Ada\n\n## Summary\n\nLeads the Acme account.\n");
    this.writeKey("People/bob.md", "---\npawn: person\nspeaker_id: bob\n---\n\n# Bob\n");

    const weeklyLabels: SessionLabel[] = [
      {
        local_label: "SPEAKER_00",
        duration_s: 900,
        sample_text: "I'll send the launch checklist after this.",
        mapped_speaker_id: null,
        mapped_name: null,
        ignored: false,
        has_centroid: true,
        talk_seconds: 900,
        share: 0.36,
        suggested_speaker_id: "bob",
        suggested_score: 0.82,
        samples: [
          { start: 48, end: 70, text: "I'll send the launch checklist after this." },
          { start: 400, end: 418, text: "Can we move the review to Thursday?" },
        ],
      },
      {
        local_label: "Ada",
        duration_s: 1080,
        sample_text: "The budget is signed off.",
        mapped_speaker_id: "ada",
        mapped_name: "Ada",
        ignored: false,
        has_centroid: true,
        talk_seconds: 1080,
        share: 0.43,
        samples: [{ start: 12, end: 40, text: "The budget is signed off." }],
      },
    ];
    const weekly: SessionDetail = {
      session_id: "weekly-sync",
      title: "Product weekly",
      created_at: atLocal(0, 10, 2),
      updated_at: atLocal(0, 10, 44),
      segments: 18,
      duration_seconds: 42 * 60,
      summary: "Launch stays on 1 December. Ada owns the budget; one speaker is still unnamed.",
      transcript_path: "Pawn/Transcripts/Product weekly.md",
      speakers: ["Ada", "SPEAKER_00"],
      unmapped_count: 1,
      needs_speakers: true,
      speakers_trained: false,
      tags: ["work/acme"],
      topics: ["launch", "budget"],
      sentiment: "neutral",
      has_analysis: true,
      labels: weeklyLabels,
      gallery: this.gallery.map((s) => ({ id: s.id, display_name: s.display_name, enrollments: s.enrollments })),
      analysis: {
        summary: "Launch stays on 1 December. Ada owns the budget; one speaker is still unnamed.",
        topics: ["launch", "budget"],
        tags: ["work/acme"],
        sentiment: "neutral",
        key_points: ["Launch date is 1 December", "Budget is signed off"],
      },
      analysis_path: "Pawn/Analyses/Product weekly.md",
      processing: false,
    };
    const kickoff: SessionDetail = {
      session_id: "acme-kickoff",
      title: "Acme kickoff",
      created_at: atLocal(-1, 15, 40),
      updated_at: atLocal(-1, 15, 52),
      segments: 9,
      duration_seconds: 12 * 60,
      summary: "Kickoff with Bob. Speakers are trained.",
      transcript_path: "Pawn/Transcripts/Acme kickoff.md",
      speakers: ["Ada", "Bob"],
      unmapped_count: 0,
      needs_speakers: false,
      speakers_trained: true,
      tags: ["work/acme"],
      topics: ["kickoff"],
      sentiment: "positive",
      has_analysis: true,
      labels: [],
      gallery: weekly.gallery,
      analysis: {
        summary: "Kickoff with Bob. Speakers are trained.",
        topics: ["kickoff"],
        tags: ["work/acme"],
        sentiment: "positive",
        key_points: ["Bob joins the account"],
      },
      processing: false,
    };
    const call: SessionDetail = {
      session_id: "call-bob",
      title: "Call with Bob",
      created_at: atLocal(0, 15, 10),
      updated_at: atLocal(0, 15, 30),
      segments: 4,
      duration_seconds: 20 * 60,
      summary: "A short call that is not on the calendar yet.",
      transcript_path: "Pawn/Transcripts/Call with Bob.md",
      speakers: ["Bob"],
      unmapped_count: 0,
      needs_speakers: false,
      speakers_trained: true,
      tags: ["work/acme"],
      topics: ["call"],
      sentiment: "positive",
      has_analysis: false,
      labels: [],
      gallery: weekly.gallery,
      processing: false,
    };
    this.sessionRows = [weekly, kickoff, call];
    this.transcripts.set("weekly-sync", {
      session_id: "weekly-sync",
      duration: 42 * 60,
      files: [{ index: 0, audio_file: "s3://pawn/weekly-sync.webm", offset: 0, duration: 42 * 60, mime: "audio/webm" }],
      speakers: [
        { label: "Ada", name: "Ada", speaker_id: "ada", ignored: false },
        { label: "SPEAKER_00", name: null, speaker_id: null, ignored: false },
      ],
      turns: [
        {
          id: "turn-ada",
          label: "Ada",
          start: 12,
          end: 48,
          segments: [{ id: "seg-1", start: 12, end: 48, text: "The budget is signed off." }],
        },
        {
          id: "turn-anon",
          label: "SPEAKER_00",
          start: 48,
          end: 90,
          segments: [{ id: "seg-2", start: 48, end: 90, text: "I'll send the launch checklist after this." }],
        },
      ],
      captures: [{ kind: "screenshot", t: 301, text: "Roadmap slide", image_key: "Pawn/Captures/assets/roadmap.png" }],
    });
    this.transcripts.set("call-bob", {
      session_id: "call-bob",
      duration: 20 * 60,
      files: [{ index: 0, audio_file: "s3://pawn/call-bob.webm", offset: 0, duration: 20 * 60, mime: "audio/webm" }],
      speakers: [{ label: "Bob", name: "Bob", speaker_id: "bob", ignored: false }],
      turns: [
        {
          id: "turn-call",
          label: "Bob",
          start: 4,
          end: 20,
          segments: [{ id: "seg-call", start: 4, end: 20, text: "Can we lock the launch date?" }],
        },
      ],
      captures: [],
    });
    this.transcripts.set("acme-kickoff", {
      session_id: "acme-kickoff",
      duration: 12 * 60,
      files: [{ index: 0, audio_file: "s3://pawn/acme.wav", offset: 0, duration: 12 * 60, mime: "audio/wav" }],
      speakers: [
        { label: "Ada", name: "Ada", speaker_id: "ada", ignored: false },
        { label: "Bob", name: "Bob", speaker_id: "bob", ignored: false },
      ],
      turns: [
        {
          id: "turn-bob",
          label: "Bob",
          start: 5,
          end: 40,
          segments: [{ id: "seg-b", start: 5, end: 40, text: "Happy to be here." }],
        },
      ],
      captures: [],
    });

    const weeklyEvent: PlannerEvent = {
      id: "ev-weekly",
      title: "Product weekly",
      start: `${today}T09:00:00`,
      end: `${today}T09:45:00`,
      source: "manual",
      attendees: ["Ada", "Bob"],
      topics: ["launch"],
      organizer: "Ada",
      location: "Teams",
      meeting_url: "https://teams.example/weekly",
      session_id: "weekly-sync",
      focus: true,
      focus_before: 15,
      path: `Pawn/Events/${today}-product-weekly-ev-weekly.md`,
      prep: "",
      day: today,
      has_prep: false,
    };
    const lunch: PlannerEvent = {
      id: "ev-lunch",
      title: "Lunch with Chiara",
      start: `${today}T12:30:00`,
      end: `${today}T13:15:00`,
      source: "manual",
      attendees: ["Chiara"],
      day: today,
      path: `Pawn/Events/${today}-lunch-ev-lunch.md`,
      has_prep: false,
    };
    const offsite: PlannerEvent = {
      id: "ev-offsite",
      title: "Offsite",
      start: today,
      source: "manual",
      day: today,
      path: `Pawn/Events/${today}-offsite-ev-offsite.md`,
      has_prep: false,
    };
    this.plannerEvents = [weeklyEvent, lunch, offsite];
    this.writeKey(weeklyEvent.path!, `---\npawn: event\nid: ev-weekly\ntitle: Product weekly\nstart: ${today}T09:00:00\n---\n\n# Product weekly\n`);
    this.writeKey(
      "Pawn/Transcripts/Product weekly.md",
      "---\npawn: transcript\nsession_id: weekly-sync\n---\n\n# Product weekly\n\nLaunch stays on 1 December.\n",
    );
  }

  private openItemsFor(sessionId: string): number {
    return this.items.filter((i) => i.source_ref === sessionId && OPEN_ITEM.includes(i.status)).length;
  }

  private sessionView(row: SessionDetail): SessionSummary {
    const event = this.plannerEvents.find((e) => e.session_id === row.session_id);
    return {
      ...row,
      items_open: this.openItemsFor(row.session_id),
      event: event ? { id: event.id, title: event.title } : null,
      labels: undefined,
      gallery: undefined,
      analysis: undefined,
    } as SessionSummary;
  }

  private requireSession(id: string): SessionDetail {
    const row = this.sessionRows.find((s) => s.session_id === id);
    if (!row) throw new HttpError(404, `Unknown session: ${id}`);
    return row;
  }

  private requireEvent(id: string): PlannerEvent {
    const ev = this.plannerEvents.find((e) => e.id === id);
    if (!ev) throw new HttpError(404, `event not found: ${id}`);
    return ev;
  }

  private requireSpeaker(id: string): SpeakerDetail {
    const sp = this.gallery.find((s) => s.id === id);
    if (!sp) throw new HttpError(404, `Unknown speaker: ${id}`);
    return sp;
  }

  private sideChatsFor(ref: string, kind: string): { id: string; title: string }[] {
    return [...this.chats.values()]
      .filter((c) => c.origin && !c.origin.promoted && c.origin.kind === kind && c.origin.ref === ref)
      .map((c) => ({ id: c.id, title: c.title }));
  }

  private emitJob(job: Job): void {
    const copy = { ...job, updated_at: nowIso() };
    this.jobRows.set(job.id, copy);
    for (const fn of this.jobListeners) fn({ ...copy });
  }

  private makeJob(kind: string, title: string, status: Job["status"] = "queued"): Job {
    const job: Job = {
      id: uid(),
      kind,
      status,
      title,
      instruction: title,
      conversation: "",
      approved: false,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    this.emitJob(job);
    return job;
  }

  private audioBytes(): Uint8Array {
    if (!this.wav) this.wav = silentWav(0.25, 8000);
    return this.wav;
  }

  private buildDay(date: string): PlannerDay {
    const day = date || dayStamp(0);
    const events = this.plannerEvents.filter((e) => (e.day || localDay(e.start)) === day);
    const sessions = this.sessionRows
      .filter((s) => localDay(s.created_at) === day)
      .map((s) => this.sessionView(s));
    const linked = new Set(events.map((e) => e.session_id).filter(Boolean) as string[]);
    const blocks: PlannerBlock[] = [
      ...events.map((event) => ({
        kind: "event" as const,
        sort_key: event.start || day,
        event,
        session: sessions.find((s) => s.session_id === event.session_id) ?? null,
        items: this.items.filter((i) => i.source_ref && i.source_ref === event.session_id),
      })),
      ...sessions
        .filter((s) => !linked.has(s.session_id))
        .map((session) => ({
          kind: "session" as const,
          sort_key: session.created_at || day,
          event: null,
          session,
          items: this.items.filter((i) => i.source_ref === session.session_id),
        })),
    ];
    const meetingRefs = new Set(sessions.map((s) => s.session_id));
    const unlinked = sessions.filter((s) => !linked.has(s.session_id));
    const pick = unlinked[0];
    return {
      date: day,
      events,
      sessions,
      blocks,
      other_items: this.items.filter((i) => OPEN_ITEM.includes(i.status) && !meetingRefs.has(i.source_ref)),
      suggestions: events
        .filter((e) => !e.session_id)
        .slice(0, 3)
        .map((e) => ({
          event_id: e.id,
          event_path: e.path,
          session_id: pick?.session_id ?? "",
          session_title: pick?.title,
          score: pick ? 0.4 : 0,
        }))
        .filter((s) => s.session_id),
      plan: "- Ship the sessions view\n- Prep Product weekly",
      plan_path: `Pawn/Daily/${day}.md`,
    };
  }

  async listSessions(params: ListSessionsParams = {}): Promise<SessionsPage> {
    await sleep(30);
    let rows = this.sessionRows.map((s) => this.sessionView(s));
    const tags = new Map<string, number>();
    let needs = 0;
    const speakers = new Map<string, number>();
    for (const row of rows) {
      if (row.needs_speakers) needs += 1;
      for (const name of row.speakers ?? []) {
        if (/^SPEAKER_\d+$/i.test(name)) continue;
        speakers.set(name, (speakers.get(name) ?? 0) + 1);
      }
      for (const tag of row.tags ?? []) tags.set(tag, (tags.get(tag) ?? 0) + 1);
    }
    const q = (params.q ?? "").trim().toLowerCase();
    rows = rows.filter((row) => {
      if (params.needs_speakers && !row.needs_speakers) return false;
      if (params.has_analysis && !row.has_analysis) return false;
      if (params.speaker && !(row.speakers ?? []).some((s) => s.toLowerCase() === params.speaker!.toLowerCase())) return false;
      const day = localDay(row.created_at);
      if (params.from && day && day < params.from) return false;
      if (params.to && day && day > params.to) return false;
      if (!q) return true;
      const hay = [row.session_id, row.title, row.summary ?? "", ...(row.speakers ?? []), ...(row.tags ?? []), ...(row.topics ?? [])]
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
    const sort = params.sort ?? "recent";
    rows.sort((a, b) => {
      if (sort === "duration") return (b.duration_seconds ?? 0) - (a.duration_seconds ?? 0);
      const av = a.updated_at ?? a.created_at ?? "";
      const bv = b.updated_at ?? b.created_at ?? "";
      return sort === "oldest" ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    const total = rows.length;
    const offset = params.offset ?? 0;
    const limit = params.limit ?? 50;
    const top = (map: Map<string, number>, n: number) =>
      [...map.entries()]
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .slice(0, n)
        .map(([name, count]) => ({ name, count }));
    return {
      data: rows.slice(offset, offset + limit),
      facets: { speakers: top(speakers, 12), needs_speakers: needs, tags: top(tags, 20), total },
    };
  }

  async getSession(id: string): Promise<SessionDetail> {
    await sleep(20);
    const row = this.requireSession(id);
    return { ...row, ...this.sessionView(row), labels: row.labels, gallery: row.gallery, analysis: row.analysis, side_chats: this.sideChatsFor(id, "session") };
  }

  async sessionTranscript(id: string, words = false): Promise<SessionTranscript> {
    await sleep(20);
    this.requireSession(id);
    const tr = this.transcripts.get(id);
    if (!tr) throw new HttpError(404, `No transcript for ${id}`);
    if (words) return tr;
    return {
      ...tr,
      turns: tr.turns.map((t) => ({ ...t, segments: t.segments.map(({ words: _w, ...seg }) => seg) })),
    };
  }

  async sessionAudioUrl(id: string): Promise<SessionAudioUrls> {
    await sleep(10);
    this.requireSession(id);
    const bytes = this.audioBytes();
    let binary = "";
    for (const b of bytes) binary += String.fromCharCode(b);
    return {
      session_id: id,
      files: [
        {
          index: 0,
          url: `data:audio/wav;base64,${btoa(binary)}`,
          exp: Math.floor(Date.now() / 1000) + 6 * 3600,
          mime: "audio/wav",
          offset: 0,
          duration: this.transcripts.get(id)?.duration ?? 0,
        },
      ],
    };
  }

  async sessionAudio(id: string, index: number, range?: string): Promise<Response> {
    this.requireSession(id);
    const tr = this.transcripts.get(id);
    const file = tr?.files.find((f) => f.index === index);
    if (!file) throw new HttpError(404, `No audio file ${index}`);
    if (!["audio/wav", "audio/flac", "audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg"].includes(file.mime)) {
      return new Response(JSON.stringify({ detail: `${file.mime} is not playable in the browser` }), {
        status: 415,
        headers: { "Content-Type": "application/json" },
      });
    }
    const sliced = applyByteRange(this.audioBytes(), range);
    const headers: Record<string, string> = {
      "Content-Type": "audio/wav",
      "Accept-Ranges": "bytes",
    };
    if (sliced.contentRange) headers["Content-Range"] = sliced.contentRange;
    return new Response(wavBlob(sliced.body), { status: sliced.status, headers });
  }

  async createSpeaker(displayName: string): Promise<GallerySpeaker> {
    const name = displayName.trim();
    if (!name) throw new HttpError(422, "display_name is required");
    const existing = this.gallery.find((s) => s.display_name.toLowerCase() === name.toLowerCase() || s.id === name.toLowerCase());
    if (existing) return { id: existing.id, display_name: existing.display_name, enrollments: existing.enrollments, created: false };
    const id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || uid();
    const row: SpeakerDetail = {
      id,
      display_name: name,
      aliases: [],
      active: true,
      enrollments: 0,
      seconds: 0,
      sessions: 0,
      people_path: `People/${id}.md`,
      notes: null,
      enrollment_rows: [],
      recent_sessions: [],
    };
    this.gallery.push(row);
    return { id, display_name: name, enrollments: 0, created: true };
  }

  async enrollSession(id: string, body: SessionEnrollBody): Promise<Record<string, unknown>> {
    const row = this.requireSession(id);
    const label = row.labels.find((l) => l.local_label === body.from_label);
    if (!label) throw new HttpError(422, `Unknown label ${body.from_label}`);
    let speaker = this.gallery.find((s) => s.id === body.speaker || s.display_name.toLowerCase() === body.speaker.toLowerCase());
    if (!speaker) {
      const created = await this.createSpeaker(body.speaker);
      speaker = this.requireSpeaker(created.id);
    }
    const enrollment: SpeakerEnrollmentRow = {
      id: uid(),
      source_session_id: id,
      start_time: label.samples?.[0]?.start ?? 0,
      end_time: label.samples?.[0]?.end ?? label.duration_s,
      duration: label.talk_seconds ?? label.duration_s,
      quality_score: 0.9,
      approved_at: nowIso(),
    };
    speaker.enrollment_rows.push(enrollment);
    speaker.enrollments = speaker.enrollment_rows.length;
    speaker.seconds += enrollment.duration;
    label.mapped_speaker_id = speaker.id;
    label.mapped_name = speaker.display_name;
    return { enrollment_id: enrollment.id, speaker_id: speaker.id, display_name: speaker.display_name, duration: enrollment.duration };
  }

  async reidentifySession(id: string): Promise<Record<string, unknown>> {
    this.requireSession(id);
    return { session_id: id, updated: true };
  }

  async relabelSession(id: string, body: { from: string; to: string }): Promise<Record<string, unknown>> {
    const row = this.requireSession(id);
    const label = row.labels.find((l) => l.local_label === body.from);
    if (!label) throw new HttpError(422, `Unknown label ${body.from}`);
    label.local_label = body.to;
    label.mapped_name = body.to;
    return { session_id: id, from: body.from, to: body.to, segments_updated: 1 };
  }

  async reprocessSession(id: string, body: SessionReprocessBody): Promise<Record<string, unknown>> {
    const row = this.requireSession(id);
    if (!body.confirm) throw new HttpError(422, "confirm must be true");
    row.processing = true;
    const job = this.makeJob("session_reprocess", `Reprocess ${row.title}`, "running");
    setTimeout(() => {
      row.processing = false;
      job.status = "done";
      this.emitJob(job);
      if (row.transcript_path) this.publish([row.transcript_path], "agent");
    }, 40);
    return { job_id: job.id, session_id: id, mode: body.mode ?? "rediarize", queued: true };
  }

  async ignoreSessionSpeaker(id: string, body: { from: string; ignore?: boolean }): Promise<Record<string, unknown>> {
    const row = this.requireSession(id);
    const label = row.labels.find((l) => l.local_label === body.from);
    if (!label) throw new HttpError(422, `Unknown label ${body.from}`);
    label.ignored = body.ignore !== false;
    return { session_id: id, from_label: body.from, ignored: label.ignored };
  }

  async setSessionTrained(id: string, trained = true): Promise<Record<string, unknown>> {
    const row = this.requireSession(id);
    row.speakers_trained = trained;
    row.needs_speakers = trained ? false : row.unmapped_count! > 0 && !trained;
    if (trained) row.needs_speakers = false;
    return { session_id: id, speakers_trained: trained, needs_speakers: row.needs_speakers };
  }

  async extractSession(id: string, body: { items?: boolean; people?: boolean } = {}): Promise<SessionExtractResult> {
    this.requireSession(id);
    return { session_id: id, items: body.items === false ? null : { source_ref: id, items: 1, interrupts: 0 }, people: body.people ? {} : null };
  }

  async reidentifySessions(body: BulkReidentifyBody = {}): Promise<{ job_id: string }> {
    const ids = body.session_ids?.length
      ? body.session_ids
      : this.sessionRows.filter((s) => !body.needs_speakers || s.needs_speakers).slice(0, body.limit ?? 50).map((s) => s.session_id);
    const job = this.makeJob("reidentify", `Reidentify ${ids.length} sessions`, "running");
    setTimeout(() => {
      job.status = "done";
      job.result = ids.join(", ");
      this.emitJob(job);
    }, 30);
    return { job_id: job.id };
  }

  async plannerDay(date = ""): Promise<PlannerDay> {
    await sleep(20);
    return this.buildDay(date || dayStamp(0));
  }

  async plannerRange(from: string, to: string): Promise<PlannerRange> {
    const days = eachDay(from, to);
    if (!days.length) throw new HttpError(422, "from and to must be YYYY-MM-DD");
    if (days.length > 14) throw new HttpError(422, "range is at most 14 days");
    return { days: days.map((d) => this.buildDay(d)) };
  }

  async listPlannerEvents(from = "", to = ""): Promise<PlannerEvent[]> {
    return this.plannerEvents.filter((e) => {
      const day = e.day || localDay(e.start);
      if (from && day < from) return false;
      if (to && day > to) return false;
      return true;
    });
  }

  async upsertPlannerEvent(body: PlannerEventCreate): Promise<PlannerEvent> {
    const id = body.id || uid();
    const existing = this.plannerEvents.find((e) => e.id === id);
    const start = body.start || "";
    const ev: PlannerEvent = {
      ...(existing ?? { id, has_prep: false }),
      id,
      title: body.title,
      start,
      end: body.end,
      source: body.source || existing?.source || "manual",
      source_url: body.source_url,
      attendees: body.attendees ?? existing?.attendees ?? [],
      topics: body.topics ?? existing?.topics ?? [],
      organizer: body.organizer,
      location: body.location,
      meeting_url: body.meeting_url,
      session_id: body.session_id ?? existing?.session_id,
      body: body.body ?? existing?.body,
      day: localDay(start) || existing?.day || dayStamp(0),
      path: existing?.path || `Pawn/Events/${localDay(start) || dayStamp(0)}-${id}.md`,
    };
    if (existing) Object.assign(existing, ev);
    else this.plannerEvents.push(ev);
    this.publish([ev.path!], "portal");
    return ev;
  }

  async patchPlannerEvent(id: string, body: PlannerEventPatch): Promise<PlannerEvent> {
    const ev = this.requireEvent(id);
    const prep = ev.prep;
    if (body.title !== undefined) ev.title = body.title;
    if (body.start !== undefined) {
      ev.start = body.start;
      ev.day = localDay(body.start) || ev.day;
    }
    if (body.end !== undefined) ev.end = body.end;
    if (body.attendees !== undefined) ev.attendees = body.attendees;
    if (body.organizer !== undefined) ev.organizer = body.organizer;
    if (body.location !== undefined) ev.location = body.location;
    if (body.meeting_url !== undefined) ev.meeting_url = body.meeting_url;
    if (body.topics !== undefined) ev.topics = body.topics;
    if (body.focus !== undefined) ev.focus = body.focus;
    if (body.focus_before !== undefined) ev.focus_before = body.focus_before;
    if (body.session_id !== undefined) ev.session_id = body.session_id || "";
    if (body.body !== undefined) ev.body = body.body;
    ev.prep = prep;
    if (ev.path) this.publish([ev.path], "portal");
    return ev;
  }

  async linkPlannerEvent(id: string, sessionId: string | null): Promise<PlannerEvent> {
    const ev = this.requireEvent(id);
    ev.session_id = sessionId || "";
    return ev;
  }

  async deletePlannerEvent(id: string): Promise<{ deleted: boolean }> {
    const ev = this.requireEvent(id);
    this.plannerEvents = this.plannerEvents.filter((e) => e.id !== id);
    if (ev.path) this.publish([ev.path], "portal");
    return { deleted: true };
  }

  async prepPlannerEvent(id: string): Promise<Job> {
    const ev = this.requireEvent(id);
    const job = this.makeJob("meeting_prep", `Prep ${ev.title}`, "running");
    ev.prep_job_id = job.id;
    ev.prep_job_status = "running";
    setTimeout(() => {
      job.status = "done";
      job.result = `Prep for ${ev.title}.`;
      ev.prep = job.result ?? "";
      ev.prep_job_status = "done";
      ev.has_prep = true;
      this.emitJob(job);
    }, 40);
    return { ...job };
  }

  async listSpeakers(params: { q?: string; include_inactive?: boolean } = {}): Promise<SpeakerSummary[]> {
    const q = (params.q ?? "").trim().toLowerCase();
    return this.gallery
      .filter((s) => (params.include_inactive || s.active) && (!q || `${s.display_name} ${s.aliases.join(" ")} ${s.id}`.toLowerCase().includes(q)))
      .map(({ enrollment_rows: _e, recent_sessions: _r, ...summary }) => summary);
  }

  async getSpeaker(id: string): Promise<SpeakerDetail> {
    const sp = this.requireSpeaker(id);
    return { ...sp, enrollment_rows: [...sp.enrollment_rows], recent_sessions: [...sp.recent_sessions] };
  }

  async patchSpeaker(id: string, body: SpeakerPatch): Promise<SpeakerDetail & { referenced_sessions?: number; job_id?: string | null }> {
    const sp = this.requireSpeaker(id);
    const previous = sp.display_name;
    if (body.display_name !== undefined) sp.display_name = body.display_name.trim() || sp.display_name;
    if (body.aliases !== undefined) sp.aliases = body.aliases.map((a) => a.trim()).filter(Boolean);
    if (body.notes !== undefined) sp.notes = body.notes;
    const referenced = this.sessionRows.filter((s) => (s.speakers ?? []).includes(previous)).length;
    let job_id: string | null = null;
    if (body.reidentify) job_id = (await this.reidentifySessions({ limit: 50 })).job_id;
    return { ...sp, referenced_sessions: referenced, job_id };
  }

  async deactivateSpeaker(id: string): Promise<SpeakerSummary> {
    const sp = this.requireSpeaker(id);
    sp.active = false;
    const { enrollment_rows: _e, recent_sessions: _r, ...summary } = sp;
    return summary;
  }

  async deleteEnrollment(speakerId: string, enrollmentId: string): Promise<void> {
    const sp = this.requireSpeaker(speakerId);
    const before = sp.enrollment_rows.length;
    sp.enrollment_rows = sp.enrollment_rows.filter((e) => e.id !== enrollmentId);
    if (sp.enrollment_rows.length === before) throw new HttpError(404, `Unknown enrollment ${enrollmentId}`);
    sp.enrollments = sp.enrollment_rows.length;
    sp.seconds = sp.enrollment_rows.reduce((n, e) => n + e.duration, 0);
  }

  async mergeSpeakers(id: string, into: string, reidentify = false): Promise<SpeakerMergeResult> {
    const src = this.requireSpeaker(id);
    const dst = this.requireSpeaker(into);
    if (src.id === dst.id) throw new HttpError(422, "Cannot merge a speaker into itself");
    dst.enrollment_rows.push(...src.enrollment_rows);
    dst.enrollments = dst.enrollment_rows.length;
    dst.seconds += src.seconds;
    dst.sessions += src.sessions;
    const aliases = new Set([...dst.aliases, ...src.aliases, src.display_name]);
    dst.aliases = [...aliases];
    src.active = false;
    src.enrollment_rows = [];
    src.enrollments = 0;
    const affected = this.sessionRows.filter((s) => (s.speakers ?? []).some((n) => n === src.display_name || n === dst.display_name)).map((s) => s.session_id);
    let job_id: string | null = null;
    if (reidentify) job_id = (await this.reidentifySessions({ session_ids: affected })).job_id;
    return { into: dst.id, speaker: { ...dst }, affected_sessions: affected, job_id };
  }

  async ensureSpeakerNote(id: string): Promise<SpeakerPeopleResult> {
    const sp = this.requireSpeaker(id);
    const path = sp.people_path || `People/${id}.md`;
    const created = !this.files.has(path);
    if (created) {
      this.writeKey(
        path,
        `---\npawn: person\nspeaker_id: ${id}\n---\n\n# ${sp.display_name}\n`,
      );
    }
    sp.people_path = path;
    return { path, created };
  }

  async assistantsOverview(): Promise<AssistantsOverview> {
    const clips: Record<string, number> = {};
    for (const a of this.assistants)
      for (const [k, n] of Object.entries(a.wake_clips)) clips[k] = (clips[k] ?? 0) + n;
    const tools: AssistantsOverview["tools"] = {
      location_now: { action: false, summary: "current location fix", args: [] },
      status_now: { action: false, summary: "battery / network / ringer now", args: [] },
      sms_send: { action: true, summary: "send an SMS (confirmed on the phone)", args: ["to", "body"] },
      notify: { action: false, summary: "show a notification on the device", args: ["text", "title"] },
      speak: { action: false, summary: "say a sentence aloud on the assistant", args: ["text"] },
      timer_set: { action: false, summary: "start a countdown timer on the assistant", args: ["seconds", "label"] },
      access_approve: { action: true, summary: "approve full voice access for another assistant", args: ["requester", "minutes"] },
    };
    return {
      assistants: this.assistants.map((a) => structuredClone(a)),
      wake: {
        model: this.wakeModels.find((m) => m.active) ?? null,
        models: this.wakeModels.map((m) => ({ ...m })),
        clips,
      },
      handoff_window_s: 600,
      presence_fresh_s: 600,
      tools,
    };
  }

  async registerAssistant(name: string, kind: AssistantKind): Promise<AssistantRegistration> {
    const id = `dev-${Math.random().toString(36).slice(2, 10)}`;
    const now = new Date().toISOString();
    this.assistants.push({
      device_id: id,
      name: name || kind,
      kind,
      platform: null,
      enabled_categories: [],
      capabilities: [],
      unifiedpush: false,
      created_at: now,
      last_seen: now,
      presence: {},
      presence_at: null,
      thread: null,
      recent_threads: [],
      commands: [],
      access_requests: [],
      wake_clips: {},
    });
    return { device_id: id, device_token: `mock-token-${id}`, name: name || kind, kind };
  }

  async renameAssistant(id: string, name: string): Promise<AssistantSummary> {
    const row = this.requireAssistant(id);
    row.name = name.trim() || row.name;
    return structuredClone(row);
  }

  async revokeAssistant(id: string): Promise<{ deleted: string; events_removed: number }> {
    this.requireAssistant(id);
    this.assistants = this.assistants.filter((a) => a.device_id !== id);
    return { deleted: id, events_removed: 0 };
  }

  async runAssistantCommand(id: string, tool: string, args: Record<string, string>): Promise<AssistantCommandResult> {
    const row = this.requireAssistant(id);
    if (!row.capabilities.includes(tool)) throw new HttpError(422, `'${row.name}' does not offer ${tool}`);
    await sleep(600);
    const now = new Date().toISOString();
    const cmd: AssistantCommand = {
      id: `c-${Date.now()}`,
      tool,
      args,
      action: tool === "sms_send" || tool === "access_approve",
      status: "ok",
      result: {},
      created_at: now,
      expires_at: now,
      completed_at: now,
    };
    row.commands.unshift(cmd);
    return { ...cmd, text: `${tool} ran on ${row.name}.` };
  }

  async assistantCommands(id: string, limit = 50): Promise<AssistantCommand[]> {
    return this.requireAssistant(id).commands.slice(0, limit).map((c) => ({ ...c }));
  }

  async publishWakeModel(file: File, opts: { version?: string; notes?: string } = {}): Promise<WakeModel> {
    const version = opts.version || `v${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")}`;
    if (this.wakeModels.some((m) => m.version === version)) throw new HttpError(409, `Version ${version} already exists`);
    for (const m of this.wakeModels) m.active = false;
    const model: WakeModel = {
      version,
      name: "hey_pawn",
      sha256: "a1".padEnd(64, "0"),
      size: file.size,
      notes: opts.notes ?? null,
      thresholds: {},
      active: true,
      created_at: new Date().toISOString(),
      url: `/v1/assistants/wake/models/${version}`,
    };
    this.wakeModels.unshift(model);
    return { ...model };
  }

  async activateWakeModel(version: string): Promise<WakeModel> {
    const model = this.wakeModels.find((m) => m.version === version);
    if (!model) throw new HttpError(404, "Unknown model version");
    for (const m of this.wakeModels) m.active = m === model;
    return { ...model };
  }

  // ── brainmaps ───────────────────────────────────────────────────────────

  async brainmap(board: string, _params: BrainGraphParams = {}): Promise<BrainGraph> {
    await sleep(120);
    return board === "items" ? this.brain.graph() : this.boards.graph(board);
  }

  async brainmapSpill(_board: string, subjectId: string): Promise<{ nodes: BrainNode[]; edges: BrainEdge[] }> {
    await sleep(160);
    return this.brain.spill(subjectId);
  }

  async brainmapTie(_board: string, from: string, to: string, role: string): Promise<BrainTieResult> {
    await sleep(140);
    return this.brain.tie(from, to, role);
  }

  async brainmapCut(_board: string, edgeId: string): Promise<{ receipt: string }> {
    await sleep(100);
    return this.brain.cut(edgeId);
  }

  async brainmapSuggestion(
    board: string,
    id: string,
    verdict: "confirm" | "snip",
  ): Promise<{ edge?: BrainEdge | null; receipt: string; reload?: boolean }> {
    await sleep(100);
    if (id.startsWith("prop:")) {
      const res = this.boards.decide(board, id.slice(5), verdict === "confirm" ? "accept" : "reject");
      return { edge: null, receipt: res.receipt, reload: true };
    }
    return this.brain.suggestion(id, verdict);
  }

  async listBoards(archived = false): Promise<Board[]> {
    await sleep(40);
    return this.boards.list(archived);
  }

  async createBoard(body: BoardCreate): Promise<Board> {
    await sleep(60);
    const b = this.boards.create(body);
    if (body.chat_id) {
      const c = this.chat(body.chat_id);
      c.origin = { ...(c.origin ?? {}), board: b.name } as Chat["origin"];
    }
    return b;
  }

  async getBoard(name: string): Promise<Board> {
    return this.boards.read(name);
  }

  async patchBoard(name: string, patch: BoardPatch): Promise<Board> {
    await sleep(40);
    return this.boards.patch(name, patch);
  }

  async deleteBoard(name: string): Promise<{ deleted: string }> {
    await sleep(40);
    return this.boards.remove(name);
  }

  async pinToBoard(name: string, nodeIds: string[]): Promise<Board> {
    await sleep(40);
    return this.boards.pin(name, nodeIds);
  }

  async removeFromBoard(name: string, nodeIds: string[]): Promise<Board> {
    await sleep(40);
    return this.boards.unpin(name, nodeIds);
  }

  async boardSeen(name: string): Promise<Board> {
    return this.boards.seen(name);
  }

  async boardChat(name: string): Promise<Chat> {
    await sleep(40);
    const existing = this.boards.chatId(name);
    if (existing && this.chats.has(existing)) return { ...this.chat(existing) };
    const board = this.boards.read(name);
    const c = this.addChat(`Board: ${board.title}`, null, [], 0);
    c.origin = { kind: "board", ref: name, quote: board.title } as Chat["origin"];
    c.last_message_at = null;
    this.boards.setChat(name, c.id);
    return { ...c, message_count: 0 };
  }

  async decideBoardProposal(
    name: string,
    id: string,
    verdict: "accept" | "reject",
  ): Promise<{ receipt: string; proposal: BoardProposal }> {
    await sleep(60);
    return this.boards.decide(name, id, verdict);
  }

  async brainmapLasso(_board: string, body: BrainLassoBody): Promise<BrainLassoResult> {
    await sleep(160);
    return this.brain.lasso(body);
  }

  async getBrainmapLayout(_board: string): Promise<BrainLayout> {
    return this.brain.getLayout();
  }

  async putBrainmapLayout(_board: string, layout: BrainLayout): Promise<BrainLayout> {
    return this.brain.putLayout(layout);
  }

  async emergenceSubject(id: string): Promise<EmergenceSubjectDetail> {
    await sleep(120);
    return this.brain.detail(id);
  }

  async emergenceStatus(id: string, status: EmergenceStatus): Promise<{ subject: EmergenceSubject }> {
    await sleep(120);
    return this.brain.status(id, status);
  }

  async emergenceCouncil(id: string): Promise<{ started: string }> {
    await sleep(80);
    return { started: id };
  }

  async getJob(id: string): Promise<Job> {
    const job = this.jobRows.get(id);
    if (!job) throw new HttpError(404, `Unknown job ${id}`);
    return { ...job };
  }

  async jobEvents(onJob: (job: Job) => void, signal?: AbortSignal): Promise<void> {
    this.jobListeners.add(onJob);
    for (const job of this.jobRows.values()) onJob({ ...job });
    try {
      await sleep(86_400_000, signal);
    } finally {
      this.jobListeners.delete(onJob);
    }
  }

  /** Dev helper: simulate an agent write (window.pawnMock.agentWrite()). */
  agentWrite(key = `Pawn/Analyses/Mock ${Date.now()}.md`): void {
    this.writeKey(key, `# Agent output\n\nWritten at ${nowIso()}.\n`);
    this.publish([key], "agent", uid());
  }
}

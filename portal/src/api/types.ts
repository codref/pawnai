// Response shapes of /v1/portal/* (pawn_server/core/portal_api.py) and the
// existing /v1/pawn/models and /v1/vault/events routes.

import type { TextAnchor } from "@/lib/anchoring";
export type { TextAnchor } from "@/lib/anchoring";

export type ChatSort = "updated" | "created" | "name" | "tag";
export type SortOrder = "asc" | "desc";

export interface ChatOrigin {
  kind: string;
  ref: string;
  quote?: string;
  anchor?: unknown;
  /** Promoted side chats keep the origin as a link but lose badge + nesting. */
  promoted?: boolean;
  /** Source chat when the origin is a message. */
  chat_id?: string;
  /** The annotation this side chat was asked from. */
  annotation_id?: string;
}

export interface Chat {
  id: string;
  conversation_key: string;
  title: string;
  folder_id: string | null;
  position: number;
  tags: string[];
  pinned: boolean;
  archived: boolean;
  parent_chat_id: string | null;
  fork_from_message_id: string | null;
  origin: ChatOrigin | null;
  /** The background model may still rename / retag it (until the user edits). */
  title_auto?: boolean;
  tags_auto?: boolean;
  last_message_at: string | null;
  created_at: string | null;
  updated_at: string | null;
  message_count?: number;
}

export interface MessageContext {
  written_files?: string[];
  notes?: string[];
  selection?: string;
  active_note?: string;
  images?: { filename: string; media_type: string; role: string }[];
  origin?: ChatOrigin;
  [key: string]: unknown;
}

export interface Message {
  id: string;
  chat_id: string;
  seq: number;
  role: "user" | "assistant" | string;
  content: string;
  context: MessageContext;
  run_id: string | null;
  model: string | null;
  created_at: string | null;
}

export interface ChatDetail extends Chat {
  messages: Message[];
  side_chats: Chat[];
}

export interface ChatListParams {
  q?: string;
  tag?: string;
  folder_id?: string;
  unfiled?: boolean;
  pinned?: boolean;
  archived?: boolean;
  /** Side chats spawned from this ref (message id, file key, …). */
  origin_ref?: string;
  sort?: ChatSort;
  order?: SortOrder;
}

export interface ChatCreate {
  title?: string;
  folder_id?: string | null;
  tags?: string[];
  parent_chat_id?: string | null;
  origin?: ChatOrigin | null;
}

export interface ChatPatch {
  title?: string;
  folder_id?: string;
  position?: number;
  tags?: string[];
  pinned?: boolean;
  archived?: boolean;
  /** Move to Unfiled (a null folder_id means "unchanged"). */
  unfile?: boolean;
  /** Side chats only: promote to a normal chat (or undo). */
  promoted?: boolean;
}

export interface Fork extends Chat {
  sallm_seeded?: number;
}

export interface ClipCreate {
  chat_id?: string | null;
  message_ids?: string[];
  title?: string;
  tags?: string[];
  /** Explicit body (a selection); otherwise built from the messages. */
  content_md?: string;
}

export interface ClipPatch {
  title?: string;
  tags?: string[];
  content_md?: string;
}

export interface Folder {
  id: string;
  parent_id: string | null;
  name: string;
  position: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface FolderNode extends Folder {
  chat_count: number;
  children: FolderNode[];
}

export interface FolderTree {
  folders: FolderNode[];
  unfiled_count: number;
}

export interface FolderCreate {
  name: string;
  parent_id?: string | null;
  position?: number;
}

export interface FolderPatch {
  name?: string;
  parent_id?: string;
  position?: number;
  to_root?: boolean;
}

export interface TagNode {
  name: string;
  path: string;
  count: number;
  children: TagNode[];
}

export interface TagTrees {
  /** Tags across chats, clips and annotations. */
  tree: TagNode[];
  /** Chat tags only. */
  chats: TagNode[];
  /** Vault file tags (frontmatter + stored). */
  files?: TagNode[];
}

export interface MessageHit {
  id: string;
  chat_id: string;
  chat_title: string;
  role: string;
  seq: number;
  snippet: string;
  created_at?: string | null;
}

export type FileKind = "markdown" | "text" | "pdf" | "docx" | "image" | "audio" | "other" | string;

export interface FileHit {
  key: string;
  title: string;
  kind: FileKind;
  snippet: string;
  tags: string[];
  modified: string | null;
}

export interface Clip {
  id: string;
  chat_id: string | null;
  message_ids: string[];
  title: string;
  tags: string[];
  content_md: string;
  vault_key: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export type AnnotationKind = "message" | "file" | "item" | "session" | "clip" | "capture";

/** Where a search hit's annotation lives (search results only). */
export interface AnnotationContext {
  chat_id?: string;
  chat_title?: string;
  label?: string;
}

export interface Annotation {
  id: string;
  target_kind: AnnotationKind;
  target_ref: string;
  anchor: TextAnchor | null;
  body_md: string;
  tags: string[];
  color: string | null;
  side_chat_id: string | null;
  created_at: string | null;
  updated_at: string | null;
  context?: AnnotationContext;
}

export interface AnnotationCreate {
  target_kind: AnnotationKind;
  target_ref: string;
  anchor?: TextAnchor | null;
  body_md?: string;
  tags?: string[];
  color?: string | null;
  side_chat_id?: string | null;
}

/** Omitted fields stay; null clears color / side_chat_id. */
export interface AnnotationPatch {
  anchor?: TextAnchor;
  body_md?: string;
  tags?: string[];
  color?: string | null;
  side_chat_id?: string | null;
}

export interface AnnotationListParams {
  target_kind?: AnnotationKind;
  target_ref?: string | string[];
  tag?: string;
}

export interface SearchResults {
  query?: string;
  tags?: string[];
  chats: Chat[];
  files: FileHit[];
  messages: MessageHit[];
  clips: Clip[];
  annotations: Annotation[];
}

export interface ContextNote {
  path: string;
  content?: string;
  inline?: boolean;
}

export interface ChatImage {
  filename: string;
  media_type: string;
  data_base64: string;
  role?: string;
}

export interface MessageCreate {
  message: string;
  active_note?: ContextNote | null;
  selection?: string | null;
  context?: ContextNote[];
  images?: ChatImage[];
  model?: string | null;
  reasoning?: string | null;
  route?: string | null;
}

export interface FileEntry {
  name: string;
  path: string;
  modified?: string | null;
  size?: number | null;
  tags?: string[];
}

export interface FolderEntry {
  name: string;
  path: string;
  count: number;
  modified?: string | null;
}

export interface FileListing {
  prefix: string;
  folders: FolderEntry[];
  files: FileEntry[];
  scratch_dir: string;
  agent_root: string;
}

export interface FileContent {
  key: string;
  content: string;
  etag: string;
  size: number;
  last_modified: string | null;
  writable: boolean;
  tags?: string[];
}

export interface FileTags {
  key: string;
  tags: string[];
  frontmatter: string[];
  stored: string[];
  /** Where an edit goes: the note's frontmatter, or the portal DB. */
  source: "frontmatter" | "db";
}

export interface ReindexResult {
  files: number;
  indexed: number;
  removed: number;
}

export interface FileStat {
  key: string;
  etag: string;
  size?: number;
}

export interface Workspace {
  name: string;
  layout: Record<string, unknown> | null;
  updated_at: string | null;
}

export interface ModelChoice {
  id: string;
  provider: string;
  model: string;
  vision?: boolean;
  reasoning?: string[];
  reasoning_default?: string;
  routes?: string[];
  route_default?: string;
}

export interface ModelCatalog {
  default: string;
  background: string;
  models: ModelChoice[];
}

export interface VaultEvent {
  seq: number;
  paths: string[];
  source: string;
  run_id: string | null;
}

export interface VaultEvents {
  seq: number;
  resync: boolean;
  events: VaultEvent[];
}

/** Callbacks for one streamed turn of POST /v1/portal/chats/{id}/messages. */
export interface ChatStreamHandlers {
  onProgress?: (text: string, kind?: string) => void;
  onAnswer?: (content: string) => void;
  onMessage?: (msg: { id: string; seq: number; role: string }) => void;
  onFiles?: (paths: string[], runId: string | null) => void;
  onJob?: (job: Record<string, unknown>) => void;
  /** Auto-naming changed the chat's title / tags after the answer. */
  onChat?: (chat: { id: string; title: string; tags: string[] }) => void;
  onError?: (message: string) => void;
  onDone?: (data: Record<string, unknown>) => void;
}

// ── inbox ─────────────────────────────────────────────────────────────────

/** One coworker item (`/v1/items`). */
export interface InboxItem {
  id: string;
  short_id: string;
  source_kind: string;
  source_ref: string;
  kind: string;
  text: string;
  owner?: string | null;
  due?: string | null;
  quote?: string | null;
  context?: string | null;
  thread?: string | null;
  interrupt: boolean;
  movement: boolean;
  reason?: string | null;
  recurrence: number;
  status: string;
  snooze_until?: string | null;
  note_key?: string | null;
  payload: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
}

export interface ItemListParams {
  /** Comma list of statuses, or "all". */
  statuses?: string;
  kind?: string;
  q?: string;
  source_kind?: string;
  source_ref?: string;
  limit?: number;
  offset?: number;
}

export interface ItemsPage {
  items: InboxItem[];
  total: number;
}

export type ItemAction = "file" | "task" | "todo" | "later" | "ignore" | "approve" | "reject" | "delete";

export interface ItemActionResult {
  receipt: string;
  item: InboxItem;
}

export interface ItemsDeleteBody {
  ids?: string[];
  all_open?: boolean;
  statuses?: string;
  kind?: string;
  q?: string;
}

export interface Idea {
  path: string;
  title: string;
  line: string;
  status: string;
  created: string;
}

export type IdeaActionKind = "keep" | "goal" | "task" | "drop";

export interface IdeaActionResult {
  receipt: string;
  path: string;
  written: string | null;
  status: string;
}

export interface CaptureAtom {
  path: string;
  title: string;
  status: string;
  collection: string;
  entity: string;
  hint: string;
  type: string;
  snippet_kind: string;
  source_url: string;
  caption: string;
  created: string;
}

export interface CaptureCollection {
  id: string;
  title: string;
}

export interface CaptureFileBody {
  path: string;
  collection?: string;
  entity?: string;
  tags?: string[];
  ignore?: boolean;
}

export interface InboxCounts {
  items: number;
  ideas: number;
  captures: number;
}

// ── settings ──────────────────────────────────────────────────────────────

export type SettingType = "bool" | "int" | "float" | "str" | "text" | "enum" | "list" | "cron" | "tz" | "model";
export type SettingSource = "default" | "env" | "yaml" | "override";
export type SettingApply = "live" | "new_chats" | "restart";

export interface SettingField {
  path: string;
  label: string;
  group: string;
  type: SettingType;
  help: string;
  apply: SettingApply;
  secret: boolean;
  choices: string[] | null;
  nullable: boolean;
  tz_path: string | null;
  min: number | null;
  value: unknown;
  running: unknown;
  source: SettingSource;
  yaml_value: unknown;
  pending_restart: boolean;
  /** Secrets: whether a value is configured. */
  set?: boolean;
  /** Cron fields: the next fire times (ISO, in the field's timezone). */
  next?: string[];
}

export interface SettingsPayload {
  groups: { id: string; label: string }[];
  fields: SettingField[];
  catalog: ModelCatalog;
  override_file: string;
  config_file: string | null;
  restart_required: string[];
}

export interface QueueStat {
  name: string;
  source: string;
  topic: string;
  bucket: string;
  messages: number;
  leases: number;
  dead_letters: number;
  paused: boolean;
  paused_at?: string | null;
}

export interface SettingsStatus {
  version: string | null;
  config_file: string | null;
  override_file: string;
  state_dir: string;
  db: { reachable: boolean; revision: string | null; head: string | null; error?: string };
  queues: QueueStat[];
  queues_error?: string;
}

export class HttpError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "HttpError";
  }
}

/** Everything the portal UI needs from the server (HTTP client or the mock). */
export interface PortalApi {
  readonly mock: boolean;
  health(): Promise<boolean>;
  /** Cheap authenticated probe used to validate a token. */
  checkToken(): Promise<void>;

  listChats(params?: ChatListParams): Promise<Chat[]>;
  createChat(body?: ChatCreate): Promise<Chat>;
  getChat(id: string): Promise<ChatDetail>;
  patchChat(id: string, body: ChatPatch): Promise<Chat>;
  deleteChat(id: string): Promise<void>;
  moveChats(ids: string[], folderId: string | null): Promise<void>;
  sendMessage(
    chatId: string,
    body: MessageCreate,
    handlers: ChatStreamHandlers,
    signal?: AbortSignal,
  ): Promise<void>;

  listFolders(): Promise<FolderTree>;
  createFolder(body: FolderCreate): Promise<Folder>;
  patchFolder(id: string, body: FolderPatch): Promise<Folder>;
  deleteFolder(id: string): Promise<void>;

  listTags(): Promise<TagTrees>;
  renameTag(from: string, to: string): Promise<number>;
  search(q: string, limit?: number): Promise<SearchResults>;

  forkChat(chatId: string, messageId: string, title?: string): Promise<Fork>;

  listClips(params?: { tag?: string; chat_id?: string }): Promise<Clip[]>;
  getClip(id: string): Promise<Clip>;
  createClip(body: ClipCreate): Promise<Clip>;
  patchClip(id: string, body: ClipPatch): Promise<Clip>;
  deleteClip(id: string): Promise<void>;
  /** Write the clip to the scratchpad (or *path*); returns the vault key. */
  saveClip(id: string, path?: string): Promise<FileStat>;

  listAnnotations(params?: AnnotationListParams): Promise<Annotation[]>;
  createAnnotation(body: AnnotationCreate): Promise<Annotation>;
  patchAnnotation(id: string, body: AnnotationPatch): Promise<Annotation>;
  deleteAnnotation(id: string): Promise<void>;

  listFiles(prefix?: string): Promise<FileListing>;
  listAllFiles(prefix?: string): Promise<string[]>;
  readFile(key: string): Promise<FileContent>;
  readFileBlob(key: string): Promise<Blob>;
  writeFile(key: string, content: string, etag: string): Promise<FileStat>;
  moveFile(src: string, dst: string, overwrite?: boolean): Promise<FileStat>;
  deleteFile(key: string): Promise<void>;
  uploadFile(file: File, folder?: string): Promise<FileStat>;
  getFileTags(key: string): Promise<FileTags>;
  setFileTags(key: string, tags: string[]): Promise<FileTags>;
  reindexFiles(): Promise<ReindexResult>;

  getWorkspace(name?: string): Promise<Workspace>;
  putWorkspace(layout: Record<string, unknown>, name?: string): Promise<Workspace>;

  listModels(): Promise<ModelCatalog>;
  vaultEvents(since: number, timeout: number, signal?: AbortSignal): Promise<VaultEvents>;

  inboxCounts(): Promise<InboxCounts>;
  listItems(params?: ItemListParams): Promise<ItemsPage>;
  itemAction(id: string, action: ItemAction, arg?: string): Promise<ItemActionResult>;
  deleteItems(body: ItemsDeleteBody): Promise<{ deleted: number; receipts: string[] }>;
  listIdeas(status?: string): Promise<Idea[]>;
  captureIdea(line: string): Promise<string>;
  ideaAction(path: string, action: IdeaActionKind, title?: string): Promise<IdeaActionResult>;
  listCaptureAtoms(scope?: "inbox" | "all"): Promise<CaptureAtom[]>;
  listCaptureCollections(): Promise<CaptureCollection[]>;
  fileCapture(body: CaptureFileBody): Promise<Record<string, unknown>>;

  getSettings(): Promise<SettingsPayload>;
  patchSettings(values: Record<string, unknown>): Promise<SettingsPayload>;
  resetSetting(path: string): Promise<SettingsPayload>;
  settingsStatus(): Promise<SettingsStatus>;
}

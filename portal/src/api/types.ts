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
  /** Transcript offset in seconds, when the ask started from a turn. */
  t?: number;
  /** A board made from this chat: the chat is that board's chat. */
  board?: string;
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

export type AnnotationKind = "message" | "file" | "item" | "session" | "clip" | "capture" | "event";

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

export interface SessionHit {
  session_id: string;
  title: string;
  snippet: string;
  created_at?: string | null;
}

export interface SearchResults {
  query?: string;
  tags?: string[];
  chats: Chat[];
  files: FileHit[];
  messages: MessageHit[];
  clips: Clip[];
  annotations: Annotation[];
  sessions: SessionHit[];
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
  /** Board chats: the cards selected on the board. */
  board_selection?: string[];
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
  /** Board chats: ghosts the turn proposed on the board. */
  onBoard?: (data: { board: string; proposals: BoardProposal[] }) => void;
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

export interface WorkerBeat {
  running: boolean;
  started_at: string | null;
  stopped_at?: string;
  last_tick_at?: string;
  ticks: number;
  error?: string | null;
  [key: string]: unknown;
}

export interface CronLast {
  at: string;
  ok: boolean;
  result: unknown;
}

export interface SchedulerCron {
  id: string;
  label: string;
  cron: string;
  timezone: string;
  enabled: boolean;
  /** Settings path of the cron expression. */
  path: string;
  /** Settings path of the flag that turns it on. */
  gate: string;
  next: string[];
  /** Next fire the running coworker loop holds (it only rereads crons on restart). */
  running_next: string | null;
  last: CronLast | null;
}

export interface ScheduleRow {
  id: string;
  name: string;
  prompt: string;
  session_id: string;
  model: string | null;
  status: string;
  schedule_kind: string;
  timezone: string;
  run_at: string | null;
  interval_seconds: number | null;
  cron_expression: string | null;
  next_run_at: string | null;
  last_run_at: string | null;
  created_by: string;
  output_note: string | null;
}

export interface ScheduleFireRow {
  id: string;
  schedule_id: string;
  schedule_name: string | null;
  scheduled_for: string | null;
  status: string;
  attempt: number;
  claimed_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_s: number | null;
  error: string | null;
  agent_run_id: string | null;
  run_status: string | null;
}

export interface BackgroundRun {
  id: string;
  source: string;
  command: string;
  session_id: string | null;
  schedule_id: string | null;
  model: string;
  status: string;
  error: string | null;
  prompt: string | null;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_s: number | null;
}

export interface ScheduleProposalRow {
  id: string;
  action: string;
  schedule_id: string | null;
  proposed_payload: Record<string, unknown>;
  rationale: string | null;
  status: string;
  created_at: string | null;
}

export interface SchedulerDiagnostics {
  now: string;
  config: Record<string, unknown>;
  workers: Record<string, WorkerBeat>;
  crons: SchedulerCron[];
  schedules: ScheduleRow[];
  fires: ScheduleFireRow[];
  fire_counts_24h: Record<string, number>;
  proposals: ScheduleProposalRow[];
  runs: BackgroundRun[];
  last_night_run: { id: string; status: string; created_at: string | null; completed_at: string | null; error: string | null } | null;
  db_error?: string;
}

export interface ConfigDiagnostics {
  /** The running config, credentials masked. */
  config: Record<string, unknown>;
  overrides: Record<string, unknown>;
  /** Settings path → group id, for paths the Settings tab can edit. */
  editable: Record<string, string>;
  config_file: string | null;
  override_file: string;
}

// ── Sessions, Day, Speakers ────────────────────────────────────────────────
// Shapes for the endpoints in docs/plans/PORTAL_SESSIONS_DAY.md. Later tracks
// append to this section rather than editing the portal chat types above.

export type JobKind = "ask" | "push_note" | "upload" | "capture_enrich" | "meeting_prep" | "session_reprocess" | string;
export type JobStatus = "todo" | "queued" | "claimed" | "running" | "review" | "done" | "blocked" | "deleted" | string;

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  title: string;
  instruction: string;
  conversation: string;
  note_path?: string | null;
  task_key?: string | null;
  result?: string | null;
  error_code?: string | null;
  approved: boolean;
  via?: string;
  payload?: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface SessionEventLink {
  id: string;
  title: string;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  updated_at?: string | null;
  created_at?: string | null;
  /** Wall-clock recording span. created_at is the segment insert time. */
  start?: string | null;
  end?: string | null;
  segments?: number | null;
  duration_seconds?: number | null;
  summary?: string | null;
  transcript_path?: string | null;
  speakers?: string[];
  unmapped_count?: number;
  needs_speakers?: boolean;
  speakers_trained?: boolean;
  tags?: string[];
  topics?: string[];
  sentiment?: string | null;
  has_analysis?: boolean;
  items_open?: number;
  event?: SessionEventLink | null;
}

export interface SessionFacet {
  name: string;
  count: number;
}

export interface SessionFacets {
  speakers: SessionFacet[];
  needs_speakers: number;
  tags?: SessionFacet[];
  total?: number;
}

export interface SessionsPage {
  data: SessionSummary[];
  facets: SessionFacets;
}

export interface ListSessionsParams {
  q?: string;
  limit?: number;
  offset?: number;
  needs_speakers?: boolean;
  speaker?: string;
  from?: string;
  to?: string;
  sort?: "recent" | "oldest" | "duration";
  has_analysis?: boolean;
}

export interface SessionSample {
  start: number;
  end: number;
  text: string;
}

export interface SessionLabel {
  local_label: string;
  duration_s: number;
  sample_text: string;
  mapped_speaker_id?: string | null;
  mapped_name?: string | null;
  ignored?: boolean;
  has_centroid: boolean;
  samples?: SessionSample[];
  talk_seconds?: number;
  share?: number;
  suggested_speaker_id?: string | null;
  suggested_score?: number | null;
}

export interface GallerySpeaker {
  id: string;
  display_name: string;
  enrollments: number;
  created?: boolean;
}

export interface SessionAnalysisView {
  summary: string | null;
  topics: string[];
  tags: string[];
  sentiment: string | null;
  key_points: string[];
}

export interface SessionSideChat {
  id: string;
  title: string;
}

export interface SessionDetail extends SessionSummary {
  labels: SessionLabel[];
  gallery: GallerySpeaker[];
  analysis?: SessionAnalysisView | null;
  analysis_path?: string | null;
  side_chats?: SessionSideChat[];
  processing?: boolean;
}

export interface TranscriptWord {
  text: string;
  start: number;
  end: number;
}

export interface TranscriptSegment {
  id: string;
  start: number;
  end: number;
  text: string;
  words?: TranscriptWord[];
}

export interface TranscriptTurn {
  id: string;
  label: string;
  start: number;
  end: number;
  segments: TranscriptSegment[];
}

export interface TranscriptFile {
  index: number;
  audio_file: string;
  offset: number;
  duration: number;
  mime: string;
  estimated?: boolean;
}

export interface TranscriptSpeaker {
  label: string;
  name: string | null;
  speaker_id: string | null;
  ignored: boolean;
}

export interface TranscriptCapture {
  kind: "screenshot" | "annotation" | string;
  t: number;
  text: string;
  image_key?: string | null;
}

export interface SessionTranscript {
  session_id: string;
  duration: number;
  files: TranscriptFile[];
  speakers: TranscriptSpeaker[];
  turns: TranscriptTurn[];
  captures: TranscriptCapture[];
}

export interface SessionAudioFile {
  index: number;
  url: string;
  exp: number;
  mime: string;
  /** Global start of the file in session seconds. */
  offset: number;
  duration: number;
  /** Offset estimated from segment times (header unreadable). */
  estimated?: boolean;
}

export interface SessionAudioUrls {
  session_id: string;
  files: SessionAudioFile[];
}

export interface SessionEnrollBody {
  speaker: string;
  from_label: string;
  force?: boolean;
}

export interface SessionReprocessBody {
  confirm: boolean;
  mode?: "rediarize" | "retranscribe" | string;
  backend?: string;
  language?: string;
  threshold?: number;
  cross_file_threshold?: number;
  margin?: number;
  id?: string;
}

export interface SessionExtractResult {
  session_id: string;
  items: { source_ref?: string; items?: number; interrupts?: number } | null;
  people: Record<string, unknown> | null;
}

export interface BulkReidentifyBody {
  session_ids?: string[];
  needs_speakers?: boolean;
  limit?: number;
}

export interface PlannerEvent {
  id: string;
  title: string;
  start?: string;
  end?: string;
  source?: string;
  source_url?: string;
  attendees?: string[];
  topics?: string[];
  organizer?: string;
  location?: string;
  meeting_url?: string;
  session_id?: string;
  fingerprint?: string;
  prep_job_id?: string;
  prep_at?: string;
  prep_job_status?: string | null;
  focus?: boolean;
  focus_before?: number;
  path?: string;
  body?: string;
  prep?: string;
  day?: string;
  clip_count?: number;
  image_count?: number;
  has_prep?: boolean;
}

export interface PlannerSuggestion {
  event_id: string;
  event_path?: string;
  session_id: string;
  session_title?: string;
  score?: number;
}

export interface PlannerBlock {
  kind: "event" | "session";
  sort_key?: string;
  event?: PlannerEvent | null;
  session?: SessionSummary | null;
  items?: InboxItem[];
}

export interface PlannerDay {
  date: string;
  events: PlannerEvent[];
  sessions: SessionSummary[];
  blocks: PlannerBlock[];
  other_items: InboxItem[];
  suggestions: PlannerSuggestion[];
  plan?: string;
  plan_path?: string | null;
}

export interface PlannerRange {
  days: PlannerDay[];
}

export interface PlannerEventCreate {
  title: string;
  start?: string;
  end?: string;
  source?: string;
  source_url?: string;
  attendees?: string[];
  topics?: string[];
  organizer?: string;
  location?: string;
  meeting_url?: string;
  session_id?: string;
  body?: string;
  id?: string;
}

export interface PlannerEventPatch {
  title?: string;
  start?: string;
  end?: string;
  attendees?: string[];
  organizer?: string;
  location?: string;
  meeting_url?: string;
  topics?: string[];
  focus?: boolean;
  focus_before?: number;
  session_id?: string | null;
  body?: string;
}

export interface SpeakerEnrollmentRow {
  id: string;
  source_session_id: string | null;
  start_time: number | null;
  end_time: number | null;
  duration: number;
  quality_score: number | null;
  approved_at: string | null;
}

export interface SpeakerSessionRef {
  session_id: string;
  title: string;
  created_at?: string | null;
}

export interface SpeakerSummary {
  id: string;
  display_name: string;
  aliases: string[];
  active: boolean;
  enrollments: number;
  seconds: number;
  sessions: number;
  last_seen_session?: string | null;
  last_seen_at?: string | null;
  people_path?: string | null;
  notes?: string | null;
}

export interface SpeakerDetail extends SpeakerSummary {
  enrollment_rows: SpeakerEnrollmentRow[];
  recent_sessions: SpeakerSessionRef[];
}

export interface SpeakerPatch {
  display_name?: string;
  aliases?: string[];
  notes?: string;
  reidentify?: boolean;
}

export interface SpeakerMergeResult {
  into: string;
  speaker: SpeakerDetail;
  affected_sessions: string[];
  job_id?: string | null;
}

export interface SpeakerPeopleResult {
  path: string;
  created: boolean;
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
// ── Assistants (voice assistants: phone + laptop) ─────────────────────────────

export type AssistantKind = "phone" | "desktop";
export type PresenceState = "active" | "idle" | "away";

export interface AssistantPresence {
  state?: PresenceState;
  headset?: boolean;
  on_call?: boolean;
  in_session?: boolean;
  last_wake_at?: string;
}

export interface AssistantCommand {
  id: string;
  tool: string;
  args: Record<string, string>;
  action: boolean;
  status: "pending" | "delivered" | "ok" | "denied" | "error" | "expired";
  result: Record<string, unknown> | null;
  created_at: string | null;
  expires_at: string | null;
  completed_at?: string | null;
}

export interface VoiceThread {
  conversation: string;
  device_id: string | null;
  device_name: string | null;
  access: "chat" | "full";
  lang: string;
  state: "active" | "handed_off" | "ended";
  turns: number;
  last_user: string | null;
  last_answer: string | null;
  portal_chat_id: string | null;
  started_at: string | null;
  last_turn_at: string | null;
}

export interface AccessRequest {
  id: string;
  requester_device_id: string;
  approver_device_id: string | null;
  approver_name: string | null;
  conversation: string | null;
  minutes: number;
  status: "pending" | "approved" | "denied" | "expired" | "error";
  message: string | null;
  created_at: string | null;
  expires_at: string | null;
  access_until: string | null;
}

export interface AssistantSummary {
  device_id: string;
  name: string;
  kind: AssistantKind;
  platform: string | null;
  enabled_categories: string[];
  capabilities: string[];
  unifiedpush: boolean;
  created_at: string | null;
  last_seen: string | null;
  presence: AssistantPresence;
  presence_at: string | null;
  thread: VoiceThread | null;
  recent_threads: VoiceThread[];
  commands: AssistantCommand[];
  access_requests: AccessRequest[];
  wake_clips: Record<string, number>;
}

export interface WakeModel {
  version: string;
  name: string;
  sha256: string;
  size: number;
  notes: string | null;
  thresholds: Record<string, number>;
  active: boolean;
  created_at: string | null;
  url: string;
}

export interface AssistantToolSpec {
  action: boolean;
  summary: string;
  args: string[];
}

export interface AssistantsOverview {
  assistants: AssistantSummary[];
  wake: { model: WakeModel | null; models: WakeModel[]; clips: Record<string, number> };
  handoff_window_s: number;
  presence_fresh_s: number;
  tools: Record<string, AssistantToolSpec>;
}

export interface AssistantRegistration {
  device_id: string;
  device_token: string;
  name: string;
  kind: AssistantKind;
}

export interface AssistantCommandResult extends AssistantCommand {
  text: string;
}

// ── brainmaps ─────────────────────────────────────────────────────────────

/** Card kinds a brainmap board can pin up. */
export type BrainKind =
  | "item"
  | "thread"
  | "question"
  | "decision"
  | "commitment"
  | "prediction"
  | "hypothesis"
  | "person"
  | "session"
  | "scrap"
  | "note";

export type BrainRole =
  | "mentions"
  | "evidence_for"
  | "evidence_against"
  | "updates"
  | "resolves"
  | "contradicts"
  | "origin"
  | "parent"
  | "owner"
  | "speaker"
  | "source";

/** Who put the string there. */
export type BrainBy = "user" | "pawn" | "goal" | "derived";

export interface BrainAssumption {
  text: string;
  health: "ok" | "weak" | "broken" | string;
}

export interface BrainNodeMeta {
  item_id?: string;
  item_kind?: string;
  observation_id?: string;
  subject_id?: string;
  source_ref?: string;
  start_sec?: number | null;
  momentum?: number | null;
  due?: string | null;
  direction?: string | null;
  owner?: string | null;
  speaker?: string | null;
  assumptions?: BrainAssumption[];
  evidence_count?: number;
  note_key?: string | null;
  by_when?: string | null;
  [key: string]: unknown;
}

export interface BrainNode {
  id: string;
  kind: BrainKind;
  title: string;
  body?: string | null;
  quote?: string | null;
  tone?: string | null;
  status: string;
  origin?: string | null;
  proposed: boolean;
  meta: BrainNodeMeta;
  /** Verbs the server allows on this card (item verbs, accept, dismiss, spill, council). */
  actions: string[];
}

export interface BrainEdge {
  id: string;
  from: string;
  to: string;
  role: BrainRole | string;
  by: BrainBy | string;
  cuttable: boolean;
  score?: number | null;
  /** Tied since the user last looked at the board ("while you slept"). */
  new?: boolean;
}

export interface BrainSuggestion {
  id: string;
  from: string;
  to: string;
  role: BrainRole | string;
  reason: string;
  score: number | null;
  /** Board proposals (`prop:` ids): who proposed it, chat or night. */
  by?: string;
}

export interface BrainGraphMeta {
  name: string;
  title: string;
  kind: BoardKind;
  chat_id?: string | null;
  subject_id?: string | null;
  seen_at?: string | null;
  spec?: BoardSpec;
}

export interface BrainGraph {
  board: string;
  meta?: BrainGraphMeta;
  nodes: BrainNode[];
  edges: BrainEdge[];
  suggestions: BrainSuggestion[];
  proposals?: BoardProposal[];
  emergence_enabled: boolean;
  truncated: boolean;
}

// ── user boards ───────────────────────────────────────────────────────────

/** items = the built-in board; lens = a saved query; manual = pins only; question = worked overnight. */
export type BoardKind = "items" | "lens" | "manual" | "question";

export interface BoardSpec {
  q?: string;
  subject_ids?: string[];
  subject_kinds?: string[];
  subject_statuses?: string[];
  item_kinds?: string[];
  item_statuses?: string[];
  people?: string[];
  sources?: string[];
  threads?: string[];
  days?: number;
  expand?: boolean;
}

export interface BoardNote {
  id: string;
  title: string;
  body: string;
  sources: string[];
  created_at?: string | null;
}

export interface Board {
  name: string;
  title: string;
  kind: BoardKind;
  spec: BoardSpec;
  pins: string[];
  hidden: string[];
  notes: BoardNote[];
  folder_id: string | null;
  tags: string[];
  pinned: boolean;
  archived: boolean;
  chat_id: string | null;
  subject_id: string | null;
  seen_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface BoardCreate {
  title?: string;
  kind?: Exclude<BoardKind, "items">;
  spec?: BoardSpec;
  nodes?: string[];
  folder_id?: string | null;
  tags?: string[];
  chat_id?: string | null;
  question?: string;
  subject_id?: string | null;
}

export interface BoardPatch {
  title?: string;
  folder_id?: string;
  unfile?: boolean;
  tags?: string[];
  pinned?: boolean;
  archived?: boolean;
  spec?: BoardSpec;
}

export interface BoardProposal {
  id: string;
  board: string;
  chat_id: string | null;
  run_id: string | null;
  origin: "chat" | "night" | string;
  kind: "string" | "card" | "group" | "pin";
  payload: Record<string, unknown>;
  status: "proposed" | "accepted" | "rejected";
  created_at: string | null;
  decided_at: string | null;
}

export interface BrainGraphParams {
  statuses?: string;
  kinds?: string;
  focus?: string;
  limit?: number;
}

export interface BrainPos {
  x: number;
  y: number;
  pinned?: boolean;
}

export interface BrainViewport {
  x: number;
  y: number;
  zoom: number;
}

export interface BrainLayout {
  positions: Record<string, BrainPos>;
  viewport: BrainViewport | null;
}

export interface BrainTieResult {
  edge: BrainEdge;
  receipt: string;
}

export interface BrainLassoBody {
  title?: string;
  into?: string;
  node_ids: string[];
}

export interface BrainLassoResult {
  subject: BrainNode;
  edges: BrainEdge[];
}

/** Emergence subject as `/v1/emergence/subjects/{id}` returns it. */
export interface EmergenceSubject {
  id: string;
  kind: string;
  title: string;
  slug: string;
  origin: string;
  status: string;
  state_md?: string | null;
  fields: Record<string, unknown>;
  momentum?: number | null;
  source_ref?: string | null;
  parent_id?: string | null;
  note_key?: string | null;
  last_evidence_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface EmergenceObservation {
  id: string;
  type: string;
  text: string;
  context: string;
  quote: string;
  speaker_name: string;
  start_sec: number | null;
  fields: Record<string, unknown>;
  status: string;
  created_at: string | null;
  source_ref: string;
  source_kind: string;
  role: string;
  linked_at: string | null;
}

export interface EmergenceSubjectEvent {
  id?: string;
  subject_id: string;
  kind: string;
  at: string;
  diff: Record<string, unknown>;
}

export interface EmergenceSubjectDetail {
  subject: EmergenceSubject;
  observations: EmergenceObservation[];
  events: EmergenceSubjectEvent[];
}

export type EmergenceStatus = "active" | "dismissed" | "done" | "parked" | "resolved" | "open";

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
  settingsScheduler(): Promise<SchedulerDiagnostics>;
  settingsConfig(): Promise<ConfigDiagnostics>;

  // Sessions, Day, Speakers. Append further methods below this block.
  listSessions(params?: ListSessionsParams): Promise<SessionsPage>;
  getSession(id: string): Promise<SessionDetail>;
  sessionTranscript(id: string, words?: boolean): Promise<SessionTranscript>;
  sessionAudioUrl(id: string): Promise<SessionAudioUrls>;
  /** Byte range uses the ``Range`` header (``bytes=start-end``). 206, 200, or 416. */
  sessionAudio(id: string, index: number, range?: string): Promise<Response>;
  createSpeaker(displayName: string): Promise<GallerySpeaker>;
  enrollSession(id: string, body: SessionEnrollBody): Promise<Record<string, unknown>>;
  reidentifySession(id: string): Promise<Record<string, unknown>>;
  relabelSession(id: string, body: { from: string; to: string }): Promise<Record<string, unknown>>;
  reprocessSession(id: string, body: SessionReprocessBody): Promise<Record<string, unknown>>;
  ignoreSessionSpeaker(id: string, body: { from: string; ignore?: boolean }): Promise<Record<string, unknown>>;
  setSessionTrained(id: string, trained?: boolean): Promise<Record<string, unknown>>;
  extractSession(id: string, body?: { items?: boolean; people?: boolean }): Promise<SessionExtractResult>;
  reidentifySessions(body?: BulkReidentifyBody): Promise<{ job_id: string }>;

  plannerDay(date?: string): Promise<PlannerDay>;
  plannerRange(from: string, to: string): Promise<PlannerRange>;
  listPlannerEvents(from?: string, to?: string): Promise<PlannerEvent[]>;
  upsertPlannerEvent(body: PlannerEventCreate): Promise<PlannerEvent>;
  patchPlannerEvent(id: string, body: PlannerEventPatch): Promise<PlannerEvent>;
  linkPlannerEvent(id: string, sessionId: string | null): Promise<PlannerEvent>;
  deletePlannerEvent(id: string): Promise<{ deleted: boolean }>;
  prepPlannerEvent(id: string): Promise<Job>;

  listSpeakers(params?: { q?: string; include_inactive?: boolean }): Promise<SpeakerSummary[]>;
  getSpeaker(id: string): Promise<SpeakerDetail>;
  patchSpeaker(id: string, body: SpeakerPatch): Promise<SpeakerDetail & { referenced_sessions?: number; job_id?: string | null }>;
  deactivateSpeaker(id: string): Promise<SpeakerSummary>;
  deleteEnrollment(speakerId: string, enrollmentId: string): Promise<void>;
  mergeSpeakers(id: string, into: string, reidentify?: boolean): Promise<SpeakerMergeResult>;
  ensureSpeakerNote(id: string): Promise<SpeakerPeopleResult>;

  assistantsOverview(): Promise<AssistantsOverview>;
  registerAssistant(name: string, kind: AssistantKind): Promise<AssistantRegistration>;
  renameAssistant(id: string, name: string): Promise<AssistantSummary>;
  revokeAssistant(id: string): Promise<{ deleted: string; events_removed: number }>;
  runAssistantCommand(id: string, tool: string, args: Record<string, string>): Promise<AssistantCommandResult>;
  assistantCommands(id: string, limit?: number): Promise<AssistantCommand[]>;
  publishWakeModel(file: File, opts?: { version?: string; notes?: string }): Promise<WakeModel>;
  activateWakeModel(version: string): Promise<WakeModel>;

  brainmap(board: string, params?: BrainGraphParams): Promise<BrainGraph>;
  brainmapSpill(board: string, subjectId: string): Promise<{ nodes: BrainNode[]; edges: BrainEdge[] }>;
  brainmapTie(board: string, from: string, to: string, role: string): Promise<BrainTieResult>;
  brainmapCut(board: string, edgeId: string): Promise<{ receipt: string }>;
  brainmapSuggestion(
    board: string,
    id: string,
    verdict: "confirm" | "snip",
  ): Promise<{ edge?: BrainEdge | null; receipt: string; reload?: boolean }>;
  brainmapLasso(board: string, body: BrainLassoBody): Promise<BrainLassoResult>;
  getBrainmapLayout(board: string): Promise<BrainLayout>;
  putBrainmapLayout(board: string, layout: BrainLayout): Promise<BrainLayout>;
  listBoards(archived?: boolean): Promise<Board[]>;
  createBoard(body: BoardCreate): Promise<Board>;
  getBoard(name: string): Promise<Board>;
  patchBoard(name: string, patch: BoardPatch): Promise<Board>;
  deleteBoard(name: string): Promise<{ deleted: string }>;
  pinToBoard(name: string, nodeIds: string[]): Promise<Board>;
  removeFromBoard(name: string, nodeIds: string[]): Promise<Board>;
  boardSeen(name: string): Promise<Board>;
  boardChat(name: string): Promise<Chat>;
  decideBoardProposal(
    name: string,
    id: string,
    verdict: "accept" | "reject",
  ): Promise<{ receipt: string; proposal: BoardProposal }>;
  emergenceSubject(id: string): Promise<EmergenceSubjectDetail>;
  emergenceStatus(id: string, status: EmergenceStatus): Promise<{ subject: EmergenceSubject }>;
  emergenceCouncil(id: string): Promise<{ started: string }>;

  getJob(id: string): Promise<Job>;
  jobEvents(onJob: (job: Job) => void, signal?: AbortSignal): Promise<void>;
}

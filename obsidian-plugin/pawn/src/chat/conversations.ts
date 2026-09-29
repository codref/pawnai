import type { EditorPosition } from "obsidian";

export interface SelectionRef {
  path: string;
  from: EditorPosition;
  to: EditorPosition;
  text: string;
}

export type MessageRole = "user" | "assistant" | "error" | "job";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  /** Job id for role=job cards. */
  jobId?: string;
  /** Tool steps shown while/after the agent worked. */
  progress?: string[];
  /** Note paths attached as context (user messages). */
  context?: string[];
  /** Selection the user message was about; enables precise "Replace selection". */
  selection?: SelectionRef;
  /** Note the conversation was about when the message was sent. */
  notePath?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
  /** Catalog id selected in the composer. Empty uses the server background default. */
  model?: string;
  /** OpenRouter reasoning effort chosen in the composer. */
  reasoning?: string;
  /** OpenRouter route chosen in the composer. */
  route?: string;
}

const MAX_MESSAGES = 60;
const MAX_CONVERSATIONS = 40;

export function newId(prefix = ""): string {
  const raw =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return prefix ? `${prefix}${raw}` : raw;
}

export function normalizeVaultKey(path: string): string {
  let p = (path || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
  if (p && !p.toLowerCase().endsWith(".md")) p = `${p}.md`;
  return p;
}

export function noteConversationId(notePath: string): string {
  return `note:${normalizeVaultKey(notePath)}`;
}

export function conversationLabel(id: string, title?: string): string {
  if (id.startsWith("note:")) {
    const base = id.slice(5).replace(/\.md$/i, "");
    return base.split("/").pop() || base;
  }
  return title || "Chat";
}

/** Local transcript cache; the server's sallm memory remains the source of truth. */
export class ConversationStore {
  private items: Record<string, Conversation>;

  constructor(
    initial: Record<string, Conversation> | undefined,
    private persist: () => void,
  ) {
    this.items = initial ?? {};
  }

  toJSON(): Record<string, Conversation> {
    return this.items;
  }

  get(id: string): Conversation {
    let conv = this.items[id];
    if (!conv) {
      conv = { id, title: conversationLabel(id), messages: [], updatedAt: Date.now() };
      this.items[id] = conv;
    }
    return conv;
  }

  has(id: string): boolean {
    return !!this.items[id];
  }

  recent(limit = 15): Conversation[] {
    return Object.values(this.items)
      .filter((c) => c.messages.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, limit);
  }

  add(id: string, msg: ChatMessage): ChatMessage {
    const conv = this.get(id);
    conv.messages.push(msg);
    if (conv.messages.length > MAX_MESSAGES) {
      conv.messages.splice(0, conv.messages.length - MAX_MESSAGES);
    }
    if (!id.startsWith("note:") && conv.messages.filter((m) => m.role === "user").length === 1) {
      if (msg.role === "user") conv.title = msg.content.split(/\r?\n/)[0].slice(0, 48) || "Chat";
    }
    conv.updatedAt = Date.now();
    this.prune();
    this.persist();
    return msg;
  }

  touch(): void {
    this.persist();
  }

  setModel(id: string, modelId: string): void {
    const conv = this.get(id);
    conv.model = modelId;
    conv.updatedAt = Date.now();
    this.persist();
  }

  setTuning(id: string, patch: { reasoning?: string; route?: string }): void {
    const conv = this.get(id);
    if (patch.reasoning !== undefined) conv.reasoning = patch.reasoning;
    if (patch.route !== undefined) conv.route = patch.route;
    conv.updatedAt = Date.now();
    this.persist();
  }

  clear(id: string): void {
    const conv = this.items[id];
    if (!conv) return;
    conv.messages = [];
    conv.updatedAt = Date.now();
    this.persist();
  }

  remove(id: string): void {
    delete this.items[id];
    this.persist();
  }

  private prune(): void {
    const all = Object.values(this.items).sort((a, b) => b.updatedAt - a.updatedAt);
    for (const conv of all.slice(MAX_CONVERSATIONS)) delete this.items[conv.id];
  }
}

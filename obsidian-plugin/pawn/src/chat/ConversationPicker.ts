import { Modal, setIcon } from "obsidian";
import type { Conversation, ConversationStore } from "./conversations";
import { conversationLabel, noteConversationId } from "./conversations";

export interface PickerOption {
  id: string;
  label: string;
  /** False for ephemeral entries (active note / empty current) that are not yet stored. */
  deletable: boolean;
  /** Last local activity; omitted when the entry has never been stored. */
  updatedAt?: number;
}

/** Compact relative stamp for the picker (Today 14:32, Yesterday, 16 Oct, …). */
export function formatUpdatedAt(ms?: number): string {
  if (!ms) return "";
  const m = window.moment(ms);
  if (!m.isValid()) return "";
  const today = window.moment().startOf("day");
  const day = m.clone().startOf("day");
  const diff = today.diff(day, "days");
  if (diff === 0) return `Today ${m.format("HH:mm")}`;
  if (diff === 1) return `Yesterday ${m.format("HH:mm")}`;
  if (diff < 7) return m.format("ddd D MMM HH:mm");
  if (m.year() === today.year()) return m.format("D MMM");
  return m.format("D MMM YYYY");
}

/** Build the option list for the conversation picker (active note + stored + current). */
export function conversationPickerOptions(
  store: ConversationStore,
  currentId: string,
  active: { path: string; basename: string } | null,
  query: string,
): PickerOption[] {
  const options: PickerOption[] = [];
  const seen = new Set<string>();
  const push = (id: string, label: string, deletable: boolean, updatedAt?: number) => {
    if (seen.has(id)) return;
    seen.add(id);
    options.push({ id, label, deletable, updatedAt });
  };

  const canDelete = (id: string): boolean => {
    if (!store.has(id)) return false;
    return store.get(id).messages.length > 0;
  };

  const storedAt = (id: string): number | undefined => {
    if (!store.has(id)) return undefined;
    return store.get(id).updatedAt;
  };

  if (active) {
    const noteId = noteConversationId(active.path);
    push(noteId, `Note: ${active.basename}`, canDelete(noteId), storedAt(noteId));
  }

  for (const conv of store.search(query)) {
    const prefix = conv.id.startsWith("note:") ? "Note: " : "";
    push(conv.id, prefix + conversationLabel(conv.id, conv.title), true, conv.updatedAt);
  }

  if (!seen.has(currentId)) {
    const conv = store.get(currentId);
    push(currentId, displayLabel(currentId, conv), canDelete(currentId), storedAt(currentId));
  }

  const q = query.trim().toLowerCase();
  if (!q) return options;
  return options.filter(
    (o) => o.label.toLowerCase().includes(q) || o.id.toLowerCase().includes(q),
  );
}

export function displayLabel(id: string, conv?: Conversation): string {
  if (id.startsWith("note:")) return `Note: ${conversationLabel(id, conv?.title)}`;
  return conversationLabel(id, conv?.title);
}

/** Confirm before wiping a conversation (local + server memory). */
export class DeleteConversationModal extends Modal {
  constructor(
    app: ConstructorParameters<typeof Modal>[0],
    private label: string,
    private onConfirm: () => void,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: "Delete conversation?" });
    contentEl.createEl("p", {
      text: `Remove “${this.label}” from the picker and clear Pawn's memory of it.`,
    });
    const row = contentEl.createDiv({ cls: "pawn-job-actions" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const ok = row.createEl("button", { text: "Delete", cls: "mod-warning" });
    ok.onclick = () => {
      this.onConfirm();
      this.close();
    };
  }
}

export interface ConversationPickerHandlers {
  currentId: string;
  getOptions: (query: string) => PickerOption[];
  onSelect: (id: string) => void;
  onDelete: (id: string, label: string) => void;
}

/**
 * Mount a trigger button that opens a searchable conversation menu.
 * Returns a closer so the host can dismiss the popover on re-render.
 */
export function mountConversationPicker(
  parent: HTMLElement,
  handlers: ConversationPickerHandlers,
): { close: () => void } {
  let pop: HTMLElement | null = null;
  let closer: (() => void) | null = null;
  let query = "";

  const trigger = parent.createEl("button", {
    cls: "dropdown pawn-conv-trigger",
    attr: { type: "button", "aria-haspopup": "listbox", "aria-label": "Choose conversation" },
  });
  const currentOpts = handlers.getOptions("");
  const current =
    currentOpts.find((o) => o.id === handlers.currentId) ??
    ({ id: handlers.currentId, label: displayLabel(handlers.currentId), deletable: false } as PickerOption);
  trigger.createSpan({ cls: "pawn-conv-trigger-label", text: current.label });
  const chevron = trigger.createSpan({ cls: "pawn-conv-trigger-chevron" });
  setIcon(chevron, "chevron-down");

  const close = () => {
    closer?.();
    closer = null;
    pop?.remove();
    pop = null;
  };

  const paintList = (list: HTMLElement) => {
    list.empty();
    const options = handlers.getOptions(query);
    if (!options.length) {
      list.createDiv({ cls: "pawn-conv-empty", text: "No conversations match." });
      return;
    }
    for (const opt of options) {
      const row = list.createDiv({
        cls: "pawn-conv-option" + (opt.id === handlers.currentId ? " is-active" : ""),
      });
      const pick = row.createEl("button", {
        cls: "pawn-conv-option-pick",
        attr: { type: "button", title: opt.label },
      });
      pick.createSpan({ cls: "pawn-conv-option-title", text: opt.label });
      const when = formatUpdatedAt(opt.updatedAt);
      if (when) {
        pick.createSpan({
          cls: "pawn-conv-option-date",
          text: when,
          attr: { title: window.moment(opt.updatedAt).format("YYYY-MM-DD HH:mm") },
        });
      }
      pick.onclick = (ev) => {
        ev.preventDefault();
        close();
        handlers.onSelect(opt.id);
      };
      if (opt.deletable) {
        const del = row.createEl("button", {
          cls: "clickable-icon pawn-conv-option-delete",
          attr: { type: "button", "aria-label": `Delete ${opt.label}` },
        });
        setIcon(del, "trash");
        del.onclick = (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          handlers.onDelete(opt.id, opt.label);
        };
      }
    }
  };

  const open = () => {
    if (pop) {
      close();
      return;
    }
    pop = document.body.createDiv({ cls: "pawn-conv-pop" });
    const search = pop.createEl("input", {
      type: "search",
      cls: "pawn-conv-search",
      attr: { placeholder: "Search conversations…", "aria-label": "Search conversations" },
    });
    search.value = query;
    const list = pop.createDiv({ cls: "pawn-conv-list", attr: { role: "listbox" } });
    paintList(list);
    search.oninput = () => {
      query = search.value;
      paintList(list);
    };

    const rect = trigger.getBoundingClientRect();
    const width = Math.min(
      Math.max(rect.width, 320),
      Math.max(280, window.innerWidth - 16),
    );
    pop.style.width = `${width}px`;
    pop.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))}px`;
    pop.style.top = `${rect.bottom + 4}px`;
    window.requestAnimationFrame(() => {
      if (!pop) return;
      const box = pop.getBoundingClientRect();
      if (box.bottom > window.innerHeight - 8) {
        pop.style.top = `${Math.max(8, rect.top - box.height - 4)}px`;
      }
      search.focus();
      search.select();
    });

    const onDown = (ev: MouseEvent) => {
      const target = ev.target;
      if (!(target instanceof Node)) return;
      if (pop?.contains(target) || trigger.contains(target)) return;
      close();
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") close();
    };
    window.setTimeout(() => {
      document.addEventListener("mousedown", onDown, true);
      document.addEventListener("keydown", onKey);
    }, 0);
    closer = () => {
      document.removeEventListener("mousedown", onDown, true);
      document.removeEventListener("keydown", onKey);
    };
  };

  trigger.onclick = (ev) => {
    ev.preventDefault();
    open();
  };

  return { close };
}

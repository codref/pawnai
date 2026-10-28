import { Modal, Notice, setIcon } from "obsidian";
import type { InboxItem, PlannerBlock, PlannerDay, PlannerEvent, PlannerSuggestion, SessionSummary } from "../api";
import { focusInboxOnSession } from "../inbox/InboxView";
import { dayLabel } from "../inbox/items";
import type PawnPlugin from "../main";
import { noteConversationId } from "../chat/conversations";
import { ExtractModal } from "../sessions/ExtractModal";
import { iconButton } from "../sessions/TrainModal";
import { AddEventModal } from "./AddEventModal";
import type { DayStore } from "./DayStore";

class ConfirmModal extends Modal {
  constructor(
    app: PawnPlugin["app"],
    private titleText: string,
    private body: string,
    private confirmLabel: string,
    private onConfirm: () => void,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: this.titleText });
    contentEl.createEl("p", { text: this.body });
    const row = contentEl.createDiv({ cls: "pawn-job-actions" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const ok = row.createEl("button", { text: this.confirmLabel, cls: "mod-warning" });
    ok.onclick = () => {
      this.onConfirm();
      this.close();
    };
  }
}

function fmtTime(local: string | undefined): string {
  if (!local) return "";
  const m = window.moment(local, ["YYYY-MM-DDTHH:mm", "YYYY-MM-DD HH:mm", "YYYY-MM-DD"], true);
  if (!m.isValid()) return local;
  if (local.length <= 10) return "";
  return m.format("HH:mm");
}

/** Schedule line for an event card (time range, all-day, or unknown). */
function fmtSchedule(event: PlannerEvent | null | undefined): string {
  if (!event) return "";
  const start = (event.start || "").trim();
  const end = (event.end || "").trim();
  if (!start) return "Time unknown";
  const startM = window.moment(start, ["YYYY-MM-DDTHH:mm", "YYYY-MM-DD HH:mm", "YYYY-MM-DD"], true);
  if (!startM.isValid()) return start;
  if (start.length <= 10 && !end) return "All day";
  const startClock = start.length > 10 ? startM.format("HH:mm") : "";
  if (!end) return startClock || "All day";
  const endM = window.moment(end, ["YYYY-MM-DDTHH:mm", "YYYY-MM-DD HH:mm", "YYYY-MM-DD"], true);
  if (!endM.isValid()) return startClock || start;
  const endClock = end.length > 10 ? endM.format("HH:mm") : "";
  if (startClock && endClock) return `${startClock} – ${endClock}`;
  if (startClock) return startClock;
  return "All day";
}

async function openPath(plugin: PawnPlugin, path: string | null | undefined): Promise<void> {
  if (!path) {
    new Notice("No note path.", 3000);
    return;
  }
  const file = plugin.app.vault.getAbstractFileByPath(path);
  if (!file) {
    new Notice(`Missing note: ${path}`, 4000);
    return;
  }
  await plugin.app.workspace.getLeaf(false).openFile(file as import("obsidian").TFile);
}

/** Day (planner) tab — timeline of events + sessions + meeting-first items. */
export function renderDayView(
  parent: HTMLElement,
  plugin: PawnPlugin,
  store: DayStore,
  opts: { onOpenChat: (conversation: string) => void },
): void {
  const refresh = () => void store.refresh();

  const toolbar = parent.createDiv({ cls: "pawn-jobs-toolbar pawn-day-toolbar" });
  const nav = toolbar.createDiv({ cls: "pawn-day-nav" });
  const prev = nav.createEl("button", { cls: "clickable-icon", attr: { "aria-label": "Previous day" } });
  setIcon(prev, "chevron-left");
  prev.onclick = () => store.shiftDays(-1);

  const todayBtn = nav.createEl("button", { cls: "pawn-chip", text: "Today" });
  todayBtn.onclick = () => store.goToday();

  const next = nav.createEl("button", { cls: "clickable-icon", attr: { "aria-label": "Next day" } });
  setIcon(next, "chevron-right");
  next.onclick = () => store.shiftDays(1);

  const dateLabel = toolbar.createDiv({
    cls: "pawn-day-date",
    text: dayLabel(store.getDate()),
  });
  dateLabel.title = store.getDate();

  const dateInput = toolbar.createEl("input", {
    type: "date",
    cls: "pawn-day-date-input",
    value: store.getDate(),
  });
  dateInput.onchange = () => {
    if (dateInput.value) store.setDate(dateInput.value);
  };

  const addBtn = toolbar.createEl("button", { cls: "pawn-chip", text: "Add event" });
  addBtn.onclick = () => {
    if (store.isOffline()) {
      new Notice("Add event requires the Pawn server.", 4000);
      return;
    }
    new AddEventModal(plugin, store.getDate(), refresh).open();
  };

  const reloadBtn = toolbar.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": "Refresh day" },
  });
  setIcon(reloadBtn, "refresh-cw");
  reloadBtn.onclick = () => refresh();

  if (store.isOffline()) {
    parent.createDiv({
      cls: "pawn-session-offline",
      text: "Server offline — browsing vault events/transcripts only. Prep and link need the server.",
    });
  }
  const err = store.lastError();
  if (err) {
    parent.createDiv({
      cls: "pawn-session-offline",
      text: `Day load error: ${err}`,
    });
  }

  const day = store.snapshot();
  if (store.isLoading() && !day) {
    parent.createDiv({ cls: "pawn-empty", text: "Loading…" });
    return;
  }
  if (!day) {
    parent.createDiv({ cls: "pawn-empty", text: "No day data." });
    return;
  }
  if (store.isLoading()) {
    parent.createDiv({ cls: "pawn-hint", text: "Refreshing…" });
  }

  renderTimeline(parent, plugin, day, store, opts, refresh);
}

function suggestionFor(
  day: PlannerDay,
  eventId: string | undefined,
): PlannerSuggestion | undefined {
  if (!eventId) return undefined;
  return day.suggestions.find((s) => s.event_id === eventId);
}

function renderTimeline(
  parent: HTMLElement,
  plugin: PawnPlugin,
  day: PlannerDay,
  store: DayStore,
  opts: { onOpenChat: (conversation: string) => void },
  refresh: () => void,
): void {
  const list = parent.createDiv({ cls: "pawn-jobs-list" });
  if (!day.blocks.length) {
    list.createDiv({
      cls: "pawn-empty",
      text: "Nothing on this day. Add an event, or record a session.",
    });
  }

  for (const block of day.blocks) {
    renderBlock(list, plugin, day, block, store, opts, refresh);
  }

  if (day.other_items.length) {
    list.createDiv({ cls: "pawn-inbox-day", text: "Other TODOs" });
    for (const item of day.other_items) {
      renderItemRow(list, item);
    }
  }
}

function renderBlock(
  parent: HTMLElement,
  plugin: PawnPlugin,
  day: PlannerDay,
  block: PlannerBlock,
  store: DayStore,
  opts: { onOpenChat: (conversation: string) => void },
  refresh: () => void,
): void {
  const event = block.event ?? null;
  const session = block.session ?? null;
  const title =
    event?.title ||
    session?.title ||
    session?.session_id ||
    "Untitled";

  const card = parent.createDiv({
    cls: `pawn-job-card pawn-day-block is-${block.kind}`,
  });
  const head = card.createDiv({ cls: "pawn-job-head" });
  head.createSpan({ cls: "pawn-job-title", text: title });
  head.createSpan({
    cls: "pawn-session-badge",
    text: block.kind === "event" ? event?.source || "event" : "session",
  });
  if (event?.has_prep || (event?.prep || "").trim()) {
    head.createSpan({ cls: "pawn-session-badge", text: "prep" });
  }
  if (event?.session_id) {
    head.createSpan({ cls: "pawn-session-badge", text: "linked" });
  }
  if ((event?.image_count || 0) > 0) {
    head.createSpan({ cls: "pawn-session-badge", text: "shot" });
  } else if ((event?.clip_count || 0) > 0) {
    head.createSpan({
      cls: "pawn-session-badge",
      text: event!.clip_count === 1 ? "1 clip" : `${event!.clip_count} clips`,
    });
  }

  const meta = card.createDiv({ cls: "pawn-job-meta" });
  const bits = [
    block.kind === "event" ? fmtSchedule(event) : fmtTime(session?.created_at || undefined),
    session?.session_id && session.session_id !== title ? session.session_id : "",
    event?.location ? String(event.location).slice(0, 48) : "",
    event?.organizer ? `org ${event.organizer}` : "",
    event?.attendees?.length ? event.attendees.slice(0, 4).join(", ") : "",
  ].filter(Boolean);
  meta.setText(bits.join(" · ") || (block.kind === "event" ? "No schedule yet" : ""));

  const decor = [...(event?.topics ?? [])].slice(0, 6);
  if (decor.length) {
    const tags = card.createDiv({ cls: "pawn-session-tags" });
    for (const tag of decor) {
      tags.createSpan({ cls: "pawn-session-tag", text: tag });
    }
  }

  if (event?.prep) {
    card.createDiv({
      cls: "pawn-session-summary pawn-day-prep",
      text: event.prep.replace(/\s+/g, " ").trim().slice(0, 280),
    });
  } else if (session?.summary) {
    card.createDiv({
      cls: "pawn-session-summary",
      text: session.summary.replace(/\s+/g, " ").trim().slice(0, 160),
    });
  } else if (event?.body) {
    // Strip snippet chrome; show a short plain preview when there is no prep.
    const preview = event.body
      .replace(/<!--\s*\/?pawn-snippet:[^>]+-->/g, "")
      .replace(/^###\s*Snippet[^\n]*$/gm, "")
      .replace(/^Source:\s*\S+$/gm, "")
      .replace(/!\[\[[^\]]+\]\]/g, "[image]")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 160);
    if (preview) {
      card.createDiv({ cls: "pawn-session-summary", text: preview });
    }
  }

  const items = block.items ?? [];
  if (items.length) {
    const box = card.createDiv({ cls: "pawn-day-items" });
    for (const item of items.slice(0, 8)) {
      renderItemRow(box, item);
    }
  }

  const sug = suggestionFor(day, event?.id);
  if (sug && !event?.session_id) {
    const row = card.createDiv({ cls: "pawn-day-suggest" });
    row.createSpan({
      text: `Link to session “${sug.session_title || sug.session_id}”?`,
    });
    const linkBtn = row.createEl("button", { cls: "pawn-chip", text: "Link" });
    linkBtn.onclick = () => void confirmLink(plugin, sug, refresh);
  }

  const actions = card.createDiv({ cls: "pawn-msg-actions" });
  if (event?.path) {
    iconButton(actions, "calendar", "Open event note", () => {
      void openPath(plugin, event.path);
    });
  }
  if (session) {
    iconButton(actions, "file-text", "Open transcript", () => {
      void openPath(plugin, session.transcript_path);
    });
    iconButton(actions, "messages-square", "Chat about this session", () => {
      void openSessionChat(plugin, session, opts.onOpenChat);
    });
    iconButton(actions, "inbox", "Show items", () => {
      focusInboxOnSession(plugin, session.session_id);
    });
    iconButton(actions, "list-checks", "Extract", () => {
      if (store.isOffline()) {
        new Notice("Extract requires the Pawn server.", 4000);
        return;
      }
      new ExtractModal(
        plugin,
        session.session_id,
        session.transcript_path ?? null,
        refresh,
      ).open();
    });
  }
  if (event?.id) {
    iconButton(actions, "sparkles", "Prep (LLM)", () => {
      if (store.isOffline()) {
        new Notice("Prep requires the Pawn server.", 4000);
        return;
      }
      void runPrep(plugin, event.id, refresh);
    });
    if (event.session_id) {
      iconButton(actions, "unlink", "Unlink session", () => {
        if (store.isOffline()) {
          new Notice("Unlink requires the Pawn server.", 4000);
          return;
        }
        void unlinkSession(plugin, event.id, refresh);
      });
    }
    iconButton(actions, "trash-2", "Delete event", () => {
      if (store.isOffline()) {
        new Notice("Delete requires the Pawn server.", 4000);
        return;
      }
      confirmDelete(plugin, event, refresh);
    });
  }
}

function renderItemRow(parent: HTMLElement, item: InboxItem): void {
  const row = parent.createDiv({ cls: "pawn-day-item" });
  const kind = item.kind || "item";
  const text = (item.text || "").replace(/\s+/g, " ").trim().slice(0, 120);
  row.setText(`${kind} — ${text}`);
}

async function confirmLink(
  plugin: PawnPlugin,
  sug: PlannerSuggestion,
  refresh: () => void,
): Promise<void> {
  try {
    await plugin.client.linkPlannerEvent(sug.event_id, sug.session_id);
    new Notice("Session linked.");
    refresh();
  } catch (err) {
    new Notice(`Link failed: ${err instanceof Error ? err.message : err}`);
  }
}

async function unlinkSession(
  plugin: PawnPlugin,
  eventId: string,
  refresh: () => void,
): Promise<void> {
  try {
    await plugin.client.linkPlannerEvent(eventId, null);
    new Notice("Session unlinked.");
    refresh();
  } catch (err) {
    new Notice(`Unlink failed: ${err instanceof Error ? err.message : err}`);
  }
}

function confirmDelete(
  plugin: PawnPlugin,
  event: PlannerEvent,
  refresh: () => void,
): void {
  const label = event.title || event.id;
  new ConfirmModal(
    plugin.app,
    "Delete event?",
    `Delete “${label}” from the planner. The vault note is removed; linked sessions stay.`,
    "Delete",
    () => {
      void (async () => {
        try {
          await plugin.client.deletePlannerEvent(event.id);
          new Notice("Event deleted.");
          refresh();
        } catch (err) {
          new Notice(`Delete failed: ${err instanceof Error ? err.message : err}`);
        }
      })();
    },
  ).open();
}

async function runPrep(
  plugin: PawnPlugin,
  eventId: string,
  refresh: () => void,
): Promise<void> {
  try {
    const job = await plugin.client.prepPlannerEvent(eventId);
    new Notice(`Prep job ${job.id.slice(0, 8)}…`);
    void plugin.openChat({ tab: "jobs", focusJob: job.id });
    // Refresh after a short delay so ## Prep may land.
    window.setTimeout(refresh, 4000);
  } catch (err) {
    new Notice(`Prep failed: ${err instanceof Error ? err.message : err}`);
  }
}

async function openSessionChat(
  plugin: PawnPlugin,
  session: SessionSummary,
  onOpenChat: (conversation: string) => void,
): Promise<void> {
  const path = session.transcript_path;
  if (path) {
    onOpenChat(noteConversationId(path));
    return;
  }
  onOpenChat(session.session_id);
}

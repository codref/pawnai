import { Modal, Notice, TFile, setIcon } from "obsidian";
import type {
  InboxItem,
  PlannerBlock,
  PlannerDay,
  PlannerEvent,
  PlannerSuggestion,
  SessionSummary,
} from "../api";
import { focusInboxOnSession } from "../inbox/InboxView";
import { dayLabel } from "../inbox/items";
import type PawnPlugin from "../main";
import { noteConversationId } from "../chat/conversations";
import { ExtractModal } from "../sessions/ExtractModal";
import { iconButton } from "../sessions/TrainModal";
import { AddEventModal } from "./AddEventModal";
import type { DayStore } from "./DayStore";

const SLOT_MINUTES = 30;
const SLOTS_PER_DAY = (24 * 60) / SLOT_MINUTES;
/** Tall enough for title + badges + actions in one 30m slot. */
const SLOT_PX = 72;
const PX_PER_MIN = SLOT_PX / SLOT_MINUTES;
const DEFAULT_FOCUS_BEFORE = 15;
/** Sessions have a start clock but no end — one-slot chip on the grid. */
const SESSION_CHIP_MINUTES = SLOT_MINUTES;
/** Gap between side-by-side overlapping cards (px). */
const OVERLAP_GAP_PX = 3;

type GridSpan = {
  start: number;
  end: number;
  kind: "session" | "event";
  block: PlannerBlock;
};

type GridPlacement = GridSpan & {
  col: number;
  cols: number;
};

/** Greedy columns + per-cluster width (Outlook-style side-by-side). */
function layoutOverlaps(spans: GridSpan[]): GridPlacement[] {
  if (!spans.length) return [];
  const sorted = [...spans].sort(
    (a, b) => a.start - b.start || b.end - a.end || a.kind.localeCompare(b.kind),
  );
  const colEnds: number[] = [];
  const withCol: { span: GridSpan; col: number }[] = [];
  for (const span of sorted) {
    let col = colEnds.findIndex((end) => end <= span.start);
    if (col < 0) {
      col = colEnds.length;
      colEnds.push(span.end);
    } else {
      colEnds[col] = span.end;
    }
    withCol.push({ span, col });
  }

  // Connected components by time overlap → shared column count.
  const n = withCol.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => {
    if (parent[i] !== i) parent[i] = find(parent[i]);
    return parent[i];
  };
  const unite = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = withCol[i].span;
      const b = withCol[j].span;
      if (a.start < b.end && b.start < a.end) unite(i, j);
    }
  }
  const clusterMax = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    clusterMax.set(root, Math.max(clusterMax.get(root) ?? 0, withCol[i].col + 1));
  }
  return withCol.map(({ span, col }, i) => ({
    ...span,
    col,
    cols: Math.max(1, clusterMax.get(find(i)) ?? col + 1),
  }));
}

function applyOverlapLayout(card: HTMLElement, place: GridPlacement): void {
  const cols = Math.max(1, place.cols);
  const col = Math.min(place.col, cols - 1);
  const widthPct = 100 / cols;
  card.style.left = `calc(${col * widthPct}% + ${OVERLAP_GAP_PX}px)`;
  card.style.width = `calc(${widthPct}% - ${OVERLAP_GAP_PX * 2}px)`;
  card.style.right = "auto";
  if (cols > 1) {
    card.classList.add("is-overlap");
    card.style.zIndex = String(3 + col);
  }
}

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

/** Minutes from local midnight; null when date-only / unparseable. */
function minutesFromMidnight(local: string | undefined): number | null {
  const raw = (local || "").trim();
  if (!raw || raw.length <= 10) return null;
  const m = window.moment(
    raw,
    [
      "YYYY-MM-DDTHH:mm:ss.SSSSSS",
      "YYYY-MM-DDTHH:mm:ss.SSS",
      "YYYY-MM-DDTHH:mm:ss",
      "YYYY-MM-DDTHH:mm",
      "YYYY-MM-DD HH:mm:ss",
      "YYYY-MM-DD HH:mm",
      window.moment.ISO_8601,
    ],
    true,
  );
  if (!m.isValid()) {
    const loose = window.moment(raw);
    if (!loose.isValid() || raw.length <= 10) return null;
    return loose.hours() * 60 + loose.minutes();
  }
  return m.hours() * 60 + m.minutes();
}

function isTimedActivity(event: PlannerEvent | null | undefined): boolean {
  return minutesFromMidnight(event?.start) != null;
}

function sessionClock(session: SessionSummary | null | undefined): string {
  return (session?.created_at || "").trim();
}

function isTimedSession(session: SessionSummary | null | undefined): boolean {
  return minutesFromMidnight(sessionClock(session)) != null;
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function slotLabel(slotIndex: number): string {
  const mins = slotIndex * SLOT_MINUTES;
  return `${pad2(Math.floor(mins / 60))}:${pad2(mins % 60)}`;
}

function slotStartIso(date: string, slotIndex: number): string {
  const mins = slotIndex * SLOT_MINUTES;
  return `${date}T${pad2(Math.floor(mins / 60))}:${pad2(mins % 60)}`;
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
  await plugin.app.workspace.getLeaf(false).openFile(file as TFile);
}

/** Day (planner) tab — hourly activity grid + sessions strip. */
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

  const addBtn = toolbar.createEl("button", { cls: "pawn-chip", text: "Add activity" });
  addBtn.onclick = () => {
    if (store.isOffline()) {
      new Notice("Add activity requires the Pawn server.", 4000);
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

  renderSchedule(parent, plugin, day, store, opts, refresh);
}

function suggestionFor(
  day: PlannerDay,
  eventId: string | undefined,
): PlannerSuggestion | undefined {
  if (!eventId) return undefined;
  return day.suggestions.find((s) => s.event_id === eventId);
}

function renderSchedule(
  parent: HTMLElement,
  plugin: PawnPlugin,
  day: PlannerDay,
  store: DayStore,
  opts: { onOpenChat: (conversation: string) => void },
  refresh: () => void,
): void {
  const timedActivities: PlannerBlock[] = [];
  const timedSessions: PlannerBlock[] = [];
  const untimed: PlannerBlock[] = [];
  for (const block of day.blocks) {
    if (block.kind === "event" && isTimedActivity(block.event)) {
      timedActivities.push(block);
      continue;
    }
    // Session (or event-linked session without its own timed event): place by created_at.
    if (block.kind === "session" && isTimedSession(block.session)) {
      timedSessions.push(block);
      continue;
    }
    // Linked session on a timed event already shows via the activity — skip duplicate.
    if (block.kind === "event" && block.session && isTimedActivity(block.event)) {
      continue;
    }
    if (block.kind === "event" && block.session && isTimedSession(block.session) && !isTimedActivity(block.event)) {
      timedSessions.push(block);
      continue;
    }
    untimed.push(block);
  }

  if (untimed.length) {
    const strip = parent.createDiv({ cls: "pawn-day-untimed" });
    strip.createDiv({ cls: "pawn-day-untimed-label", text: "All-day & untimed" });
    for (const block of untimed) {
      renderUntimedCard(strip, plugin, day, block, store, opts, refresh);
    }
  }

  if (day.other_items.length) {
    const other = parent.createDiv({ cls: "pawn-day-untimed" });
    other.createDiv({ cls: "pawn-day-untimed-label", text: "Other TODOs" });
    for (const item of day.other_items) {
      renderItemRow(other, item);
    }
  }

  const scroll = parent.createDiv({ cls: "pawn-day-grid-scroll" });
  const grid = scroll.createDiv({ cls: "pawn-day-grid" });
  grid.style.height = `${SLOTS_PER_DAY * SLOT_PX}px`;

  for (let i = 0; i < SLOTS_PER_DAY; i++) {
    const slot = grid.createDiv({
      cls: `pawn-day-slot${i % 2 === 0 ? " is-hour" : ""}`,
      attr: { "data-slot": String(i) },
    });
    slot.style.top = `${i * SLOT_PX}px`;
    slot.style.height = `${SLOT_PX}px`;
    if (i % 2 === 0) {
      slot.createSpan({ cls: "pawn-day-slot-label", text: slotLabel(i) });
    }
    slot.onclick = () => {
      if (store.isOffline()) {
        new Notice("Add activity requires the Pawn server.", 4000);
        return;
      }
      const start = slotStartIso(day.date, i);
      const endIso =
        i + 1 >= SLOTS_PER_DAY
          ? `${day.date}T23:59`
          : slotStartIso(day.date, i + 1);
      new AddEventModal(plugin, { date: day.date, start, end: endIso }, refresh).open();
    };
  }

  const layer = grid.createDiv({ cls: "pawn-day-blocks" });
  const spans: GridSpan[] = [];
  for (const block of timedSessions) {
    const start = minutesFromMidnight(sessionClock(block.session)) ?? 0;
    spans.push({
      start,
      end: start + SESSION_CHIP_MINUTES,
      kind: "session",
      block,
    });
  }
  for (const block of timedActivities) {
    const start = minutesFromMidnight(block.event?.start) ?? 0;
    let end = minutesFromMidnight(block.event?.end);
    if (end == null || end <= start) end = start + SLOT_MINUTES;
    spans.push({ start, end, kind: "event", block });
  }
  const placed = layoutOverlaps(spans);
  for (const place of placed) {
    if (place.kind === "session") {
      renderTimedSession(layer, plugin, place.block, store, opts, refresh, place);
    } else {
      renderTimedActivity(layer, plugin, day, place.block, store, refresh, place);
    }
  }

  const todayKey = window.moment().format("YYYY-MM-DD");
  const gridStarts = [
    ...timedActivities.map((b) => minutesFromMidnight(b.event?.start)),
    ...timedSessions.map((b) => minutesFromMidnight(sessionClock(b.session))),
  ].filter((n): n is number => n != null);

  if (day.date === todayKey) {
    const nowMins = window.moment().hours() * 60 + window.moment().minutes();
    const nowLine = grid.createDiv({ cls: "pawn-day-now" });
    nowLine.style.top = `${nowMins * PX_PER_MIN}px`;
    window.setTimeout(() => {
      scroll.scrollTop = Math.max(0, nowMins * PX_PER_MIN - scroll.clientHeight / 3);
    }, 0);
  } else if (gridStarts.length) {
    const first = Math.min(...gridStarts);
    window.setTimeout(() => {
      scroll.scrollTop = Math.max(0, first * PX_PER_MIN - SLOT_PX * 2);
    }, 0);
  } else if (!untimed.length && !day.other_items.length) {
    parent.createDiv({
      cls: "pawn-hint pawn-day-empty-hint",
      text: "Nothing timed yet. Click a slot or Add activity.",
    });
  }
}

function renderTimedSession(
  parent: HTMLElement,
  plugin: PawnPlugin,
  block: PlannerBlock,
  store: DayStore,
  opts: { onOpenChat: (conversation: string) => void },
  refresh: () => void,
  place: GridPlacement,
): void {
  const session = block.session!;
  const startMin = place.start;
  const duration = Math.max(SESSION_CHIP_MINUTES, place.end - place.start);
  const title = session.title || session.session_id || "Session";

  const card = parent.createDiv({ cls: "pawn-day-activity is-session" });
  card.style.top = `${startMin * PX_PER_MIN}px`;
  card.style.height = `${Math.max(SLOT_PX - 2, duration * PX_PER_MIN - 2)}px`;
  applyOverlapLayout(card, place);

  const titleEl = card.createSpan({
    cls: "pawn-day-activity-title",
    text: title,
  });
  const clock = fmtTime(sessionClock(session));
  titleEl.setAttr("role", "button");
  titleEl.tabIndex = 0;
  titleEl.title = clock ? `${title} · ${clock} · no duration` : "Open transcript";
  const openTranscript = (ev: Event) => {
    ev.stopPropagation();
    void openPath(plugin, session.transcript_path);
  };
  titleEl.onclick = openTranscript;
  titleEl.onkeydown = (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      openTranscript(ev);
    }
  };

  const badges = card.createDiv({ cls: "pawn-day-activity-badges" });
  badges.createSpan({ cls: "pawn-session-badge", text: "session" });
  badges.createSpan({ cls: "pawn-session-badge", text: "untimed" });

  const actions = card.createDiv({ cls: "pawn-msg-actions pawn-day-activity-actions" });
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
  card.onclick = (ev) => ev.stopPropagation();
}

function renderTimedActivity(
  parent: HTMLElement,
  plugin: PawnPlugin,
  day: PlannerDay,
  block: PlannerBlock,
  store: DayStore,
  refresh: () => void,
  place: GridPlacement,
): void {
  const event = block.event!;
  const startMin = place.start;
  const endMin = place.end;
  const duration = Math.max(SLOT_MINUTES, endMin - startMin);
  const focusOn = Boolean(event.focus);
  const focusBefore = Math.max(
    0,
    Number(event.focus_before) || (focusOn ? DEFAULT_FOCUS_BEFORE : 0),
  );

  if (focusOn && focusBefore > 0) {
    const bandStart = Math.max(0, startMin - focusBefore);
    const band = parent.createDiv({ cls: "pawn-day-focus-band" });
    band.style.top = `${bandStart * PX_PER_MIN}px`;
    band.style.height = `${(startMin - bandStart) * PX_PER_MIN}px`;
    band.createSpan({ cls: "pawn-day-focus-band-label", text: "focus" });
    band.title = `${focusBefore}m prep before`;
  }

  const card = parent.createDiv({
    cls: `pawn-day-activity${focusOn ? " is-focus" : ""}`,
  });
  card.style.top = `${startMin * PX_PER_MIN}px`;
  card.style.height = `${Math.max(SLOT_PX - 2, duration * PX_PER_MIN - 2)}px`;
  applyOverlapLayout(card, place);

  // Row 1 — title (span, not button — Obsidian button theme blows up the card)
  const titleEl = card.createSpan({
    cls: "pawn-day-activity-title",
    text: event.title || "Untitled",
  });
  const metaBits = [
    fmtSchedule(event),
    event.organizer ? `org ${event.organizer}` : "",
    event.attendees?.length ? event.attendees.slice(0, 3).join(", ") : "",
  ].filter(Boolean);
  titleEl.setAttr("role", "button");
  titleEl.tabIndex = 0;
  titleEl.title = metaBits.length
    ? `${event.title || "Untitled"} · ${metaBits.join(" · ")}`
    : "Open activity note";
  const openNote = (ev: Event) => {
    ev.stopPropagation();
    void openPath(plugin, event.path);
  };
  titleEl.onclick = openNote;
  titleEl.onkeydown = (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      openNote(ev);
    }
  };

  // Row 2 — labels
  const badges = card.createDiv({ cls: "pawn-day-activity-badges" });
  if (event.source) {
    badges.createSpan({ cls: "pawn-session-badge", text: event.source });
  }
  if (focusOn) {
    badges.createSpan({ cls: "pawn-session-badge", text: "focus" });
  }
  if (event.has_prep || (event.prep || "").trim()) {
    badges.createSpan({ cls: "pawn-session-badge", text: "prep" });
  }
  if ((event.image_count || 0) > 0) {
    badges.createSpan({ cls: "pawn-session-badge", text: "shot" });
  }
  if (!badges.childElementCount) {
    badges.remove();
  }

  const sug = suggestionFor(day, event.id);
  if (sug && !event.session_id && duration >= 60) {
    const row = card.createDiv({ cls: "pawn-day-suggest" });
    row.createSpan({ text: `Link “${sug.session_title || sug.session_id}”?` });
    const linkBtn = row.createEl("button", { cls: "pawn-chip", text: "Link" });
    linkBtn.onclick = (ev) => {
      ev.stopPropagation();
      void confirmLink(plugin, sug, refresh);
    };
  }

  // Row 3 — actions
  const actions = card.createDiv({ cls: "pawn-msg-actions pawn-day-activity-actions" });
  const star = iconButton(actions, "star", focusOn ? "Clear focus" : "Focus (prep band)", () => {
    void toggleFocus(plugin, event, refresh);
  });
  star.classList.toggle("is-active", focusOn);
  star.classList.toggle("is-muted", !focusOn);

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
  iconButton(actions, "trash-2", "Delete activity", () => {
    if (store.isOffline()) {
      new Notice("Delete requires the Pawn server.", 4000);
      return;
    }
    confirmDelete(plugin, event, refresh);
  });

  // Don't open Add modal when clicking the card.
  card.onclick = (ev) => ev.stopPropagation();
}

function renderUntimedCard(
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
    event?.title || session?.title || session?.session_id || "Untitled";

  const card = parent.createDiv({
    cls: `pawn-job-card pawn-day-block is-${block.kind}`,
  });
  const head = card.createDiv({ cls: "pawn-job-head" });
  if (event?.path) {
    const titleEl = head.createSpan({
      cls: "pawn-day-activity-title",
      text: title,
    });
    titleEl.setAttr("role", "button");
    titleEl.tabIndex = 0;
    titleEl.onclick = () => void openPath(plugin, event.path);
  } else {
    head.createSpan({ cls: "pawn-job-title", text: title });
  }
  head.createSpan({
    cls: "pawn-session-badge",
    text: block.kind === "event" ? event?.source || "event" : "session",
  });

  const meta = card.createDiv({ cls: "pawn-job-meta" });
  const bits = [
    block.kind === "event" ? fmtSchedule(event) : fmtTime(session?.created_at || undefined),
    event?.organizer ? `org ${event.organizer}` : "",
    event?.attendees?.length ? event.attendees.slice(0, 4).join(", ") : "",
  ].filter(Boolean);
  meta.setText(bits.join(" · ") || (block.kind === "event" ? "No schedule yet" : ""));

  const actions = card.createDiv({ cls: "pawn-msg-actions" });
  if (event?.path) {
    iconButton(actions, "calendar", "Open activity note", () => {
      void openPath(plugin, event.path);
    });
  }
  if (event?.id) {
    const focusOn = Boolean(event.focus);
    const star = iconButton(actions, "star", focusOn ? "Clear focus" : "Focus", () =>
      void toggleFocus(plugin, event, refresh),
    );
    star.classList.toggle("is-active", focusOn);
    star.classList.toggle("is-muted", !focusOn);
    iconButton(actions, "sparkles", "Prep (LLM)", () => {
      if (store.isOffline()) {
        new Notice("Prep requires the Pawn server.", 4000);
        return;
      }
      void runPrep(plugin, event.id, refresh);
    });
    iconButton(actions, "trash-2", "Delete activity", () => {
      if (store.isOffline()) {
        new Notice("Delete requires the Pawn server.", 4000);
        return;
      }
      confirmDelete(plugin, event, refresh);
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

  const sug = suggestionFor(day, event?.id);
  if (sug && event && !event.session_id) {
    const row = card.createDiv({ cls: "pawn-day-suggest" });
    row.createSpan({
      text: `Link to session “${sug.session_title || sug.session_id}”?`,
    });
    const linkBtn = row.createEl("button", { cls: "pawn-chip", text: "Link" });
    linkBtn.onclick = () => void confirmLink(plugin, sug, refresh);
  }
}

function renderItemRow(parent: HTMLElement, item: InboxItem): void {
  const row = parent.createDiv({ cls: "pawn-day-item" });
  const kind = item.kind || "item";
  const text = (item.text || "").replace(/\s+/g, " ").trim().slice(0, 120);
  row.setText(`${kind} — ${text}`);
}

async function toggleFocus(
  plugin: PawnPlugin,
  event: PlannerEvent,
  refresh: () => void,
): Promise<void> {
  const path = event.path;
  if (!path) {
    new Notice("No note path for this activity.", 3000);
    return;
  }
  const file = plugin.app.vault.getAbstractFileByPath(path);
  if (!(file instanceof TFile)) {
    new Notice(`Missing note: ${path}`, 4000);
    return;
  }
  const next = !event.focus;
  try {
    await plugin.app.fileManager.processFrontMatter(file, (fm) => {
      if (next) {
        fm.focus = true;
        if (!fm.focus_before) fm.focus_before = DEFAULT_FOCUS_BEFORE;
      } else {
        delete fm.focus;
      }
    });
    new Notice(next ? "Focus on" : "Focus cleared");
    refresh();
  } catch (err) {
    new Notice(`Focus failed: ${err instanceof Error ? err.message : err}`);
  }
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
    "Delete activity?",
    `Delete “${label}” from the planner. The vault note is removed; linked sessions stay.`,
    "Delete",
    () => {
      void (async () => {
        try {
          await plugin.client.deletePlannerEvent(event.id);
          new Notice("Activity deleted.");
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

import { Notice, setIcon } from "obsidian";
import type { SessionSummary } from "../api";
import { dayKey, dayLabel } from "../inbox/items";
import type PawnPlugin from "../main";
import { noteConversationId } from "../chat/conversations";
import { ReprocessModal } from "./ReprocessModal";
import { TrainModal, iconButton } from "./TrainModal";
import type { SessionsScope } from "./SessionsStore";

function relTime(iso?: string | null): string {
  if (!iso) return "";
  return window.moment(iso).fromNow();
}

function fmtDur(seconds?: number | null): string {
  if (seconds == null || seconds < 0) return "";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s ? `${m}m ${s}s` : `${m}m`;
}

function groupSessionsByDay(
  sessions: SessionSummary[],
): { key: string; label: string; sessions: SessionSummary[] }[] {
  const map = new Map<string, SessionSummary[]>();
  for (const s of sessions) {
    const key = dayKey(s.updated_at ?? s.created_at);
    const list = map.get(key) ?? [];
    list.push(s);
    map.set(key, list);
  }
  return Array.from(map.entries())
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([key, group]) => ({ key, label: dayLabel(key), sessions: group }));
}

/** Sessions tab content — navigate recordings; Train is secondary. */
export function renderSessionsList(
  parent: HTMLElement,
  plugin: PawnPlugin,
  opts: {
    onOpenChat: (conversation: string) => void;
  },
): void {
  const store = plugin.sessions;
  const refresh = () => void store.refresh();

  const toolbar = parent.createDiv({ cls: "pawn-jobs-toolbar" });
  const scopes = toolbar.createDiv({ cls: "pawn-inbox-scopes" });
  for (const [id, label] of [
    ["all", "All"],
    ["needs", "Needs speakers"],
  ] as const) {
    const chip = scopes.createEl("button", { text: label, cls: "pawn-chip" });
    chip.toggleClass("is-active", store.scope === id);
    chip.onclick = () => {
      store.setQuery({ scope: id as SessionsScope });
      refresh();
    };
  }
  const reloadBtn = toolbar.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": "Refresh sessions" },
  });
  setIcon(reloadBtn, "refresh-cw");
  reloadBtn.onclick = () => refresh();

  const facets = store.facets();
  if (facets.length) {
    const speakersRow = parent.createDiv({ cls: "pawn-session-speaker-filters" });
    for (const facet of facets) {
      const chip = speakersRow.createEl("button", {
        text: `${facet.name} (${facet.count})`,
        cls: "pawn-chip",
      });
      chip.toggleClass("is-active", store.speaker === facet.name);
      chip.onclick = () => {
        store.setQuery({
          speaker: store.speaker === facet.name ? "" : facet.name,
        });
        refresh();
      };
    }
  }

  const search = parent.createEl("input", {
    type: "search",
    placeholder: "Search title, id, speakers, tags…",
    cls: "pawn-inbox-search",
  });
  search.value = store.q;
  const applySearch = () => {
    store.setQuery({ q: search.value });
    refresh();
  };
  search.onchange = applySearch;
  search.onkeydown = (ev) => {
    if (ev.key === "Enter") applySearch();
  };

  if (store.isOffline()) {
    parent.createDiv({
      cls: "pawn-session-offline",
      text: "Server offline — browsing vault transcripts only. Train is unavailable.",
    });
  }

  const sessions = store.all();
  if (!sessions.length) {
    parent.createDiv({
      cls: "pawn-empty",
      text:
        store.scope === "needs"
          ? "No sessions need speaker training."
          : store.speaker
            ? `No sessions with speaker “${store.speaker}”.`
            : "No sessions found.",
    });
    return;
  }

  const list = parent.createDiv({ cls: "pawn-jobs-list" });
  for (const group of groupSessionsByDay(sessions)) {
    list.createDiv({ cls: "pawn-inbox-day", text: group.label });
    for (const session of group.sessions) {
      renderSessionCard(list, plugin, session, {
        offline: store.isOffline(),
        onOpenChat: opts.onOpenChat,
        onChanged: refresh,
        onSpeaker: (name) => {
          store.setQuery({ speaker: store.speaker === name ? "" : name });
          refresh();
        },
      });
    }
  }
}

function renderSessionCard(
  parent: HTMLElement,
  plugin: PawnPlugin,
  session: SessionSummary,
  opts: {
    offline: boolean;
    onOpenChat: (conversation: string) => void;
    onChanged: () => void;
    onSpeaker: (name: string) => void;
  },
): void {
  const card = parent.createDiv({
    cls: `pawn-job-card pawn-session-card${session.needs_speakers ? " needs-speakers" : ""}`,
  });

  const head = card.createDiv({ cls: "pawn-job-head" });
  const title = head.createSpan({
    cls: "pawn-job-title",
    text: session.title || session.session_id,
  });
  title.addClass("is-clickable");
  title.onclick = () => void openTranscript(plugin, session);
  if (session.needs_speakers) {
    head.createSpan({ cls: "pawn-session-badge is-warn", text: "needs training" });
  } else if (session.has_analysis) {
    head.createSpan({ cls: "pawn-session-badge", text: "analyzed" });
  }

  const meta = card.createDiv({ cls: "pawn-job-meta" });
  const bits = [
    relTime(session.updated_at ?? session.created_at),
    fmtDur(session.duration_seconds),
    session.session_id !== session.title ? session.session_id : "",
  ].filter(Boolean);
  meta.setText(bits.join(" · "));

  if (session.summary) {
    card.createDiv({
      cls: "pawn-session-summary",
      text: session.summary.replace(/\s+/g, " ").trim().slice(0, 160),
    });
  }

  if (session.speakers?.length) {
    const pills = card.createDiv({ cls: "pawn-session-pills" });
    for (const name of session.speakers.slice(0, 8)) {
      const anon = /^SPEAKER_\d+$/.test(name);
      const pill = pills.createEl("button", {
        cls: `pawn-chip${anon ? " is-anon is-active" : ""}`,
        text: name,
      });
      if (!anon) {
        pill.onclick = (ev) => {
          ev.stopPropagation();
          opts.onSpeaker(name);
        };
      }
    }
  }

  const decor = [...(session.tags ?? []), ...(session.topics ?? [])].slice(0, 6);
  if (decor.length || session.sentiment) {
    const tags = card.createDiv({ cls: "pawn-session-tags" });
    if (session.sentiment) {
      tags.createSpan({
        cls: "pawn-session-tag is-sentiment",
        text: session.sentiment.split(/[\n.]/)[0].trim().slice(0, 40),
      });
    }
    for (const tag of decor) {
      tags.createSpan({ cls: "pawn-session-tag", text: tag });
    }
  }

  const actions = card.createDiv({ cls: "pawn-msg-actions" });
  iconButton(actions, "file-text", "Open transcript", () => {
    void openTranscript(plugin, session);
  });
  iconButton(actions, "messages-square", "Chat about this session", () => {
    void openSessionChat(plugin, session, opts.onOpenChat);
  });
  const reprocess = iconButton(actions, "refresh-ccw", "Reprocess (rediarize / retranscribe)", () => {
    if (opts.offline) {
      new Notice("Reprocess requires the Pawn server.", 4000);
      return;
    }
    new ReprocessModal(plugin, session.session_id, opts.onChanged).open();
  });
  const train = iconButton(actions, "users", "Train speakers", () => {
    if (opts.offline) {
      new Notice("Train requires the Pawn server.", 4000);
      return;
    }
    new TrainModal(plugin, session.session_id, opts.onChanged).open();
  });
  if (opts.offline) {
    reprocess.setAttr("disabled", "true");
    train.setAttr("disabled", "true");
  }
}

async function openTranscript(plugin: PawnPlugin, session: SessionSummary): Promise<void> {
  const file = plugin.sessions.resolveTranscript(session);
  if (!file) {
    new Notice(
      session.transcript_path
        ? "Transcript note not in vault yet (sync may be pending)."
        : "No transcript vault path for this session.",
      5000,
    );
    return;
  }
  await plugin.app.workspace.getLeaf(false).openFile(file);
}

async function openSessionChat(
  plugin: PawnPlugin,
  session: SessionSummary,
  onOpenChat: (conversation: string) => void,
): Promise<void> {
  const file = plugin.sessions.resolveTranscript(session);
  if (!file) {
    new Notice(
      session.transcript_path
        ? "Transcript note not in vault yet — cannot soft-bind chat."
        : "No transcript note mapped for this session.",
      5000,
    );
    return;
  }
  await plugin.app.workspace.getLeaf(false).openFile(file);
  onOpenChat(noteConversationId(file.path));
}

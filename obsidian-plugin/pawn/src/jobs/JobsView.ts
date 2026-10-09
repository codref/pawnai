import { Component, MarkdownRenderer, Modal, Notice, TFile, setIcon } from "obsidian";
import type { Job } from "../api";
import { ExtraAction, renderMessageActions } from "../chat/MessageActions";
import { conversationLabel } from "../chat/conversations";
import { dayKey, dayLabel } from "../inbox/items";
import type PawnPlugin from "../main";
import { isActive, isDeletable, JobsStatusScope } from "./JobStore";

const KIND_ICON: Record<string, string> = {
  ask: "bot",
  push_note: "file-up",
  upload: "upload",
  capture_enrich: "tags",
};

const KIND_LABEL: Record<string, string> = {
  ask: "Ask",
  push_note: "Push note",
  upload: "Upload",
  capture_enrich: "Enrich",
};

const KIND_FILTERS = ["", "ask", "push_note", "upload", "capture_enrich"] as const;

/** Drop the ``[tool] …`` trail that enrich jobs used to persist (mirrors server strip_tool_trail). */
function stripToolTrail(raw: string): string {
  const text = (raw || "").trim();
  if (!text.startsWith("[tool]")) return text;
  const parts = text.split("\n\n");
  if (parts.length >= 2) {
    const rest = parts.slice(1).join("\n\n").trim();
    if (rest) return rest;
  }
  const kept = text.split("\n").filter((ln) => !ln.startsWith("[tool]"));
  return kept.join("\n").trim() || text;
}

const STATUS_LABEL: Record<string, string> = {
  todo: "waiting for sync",
  queued: "queued",
  claimed: "starting",
  running: "running",
  review: "ready for review",
  done: "done",
  blocked: "failed",
};

function relTime(iso?: string | null): string {
  if (!iso) return "";
  return window.moment(iso).fromNow();
}

function jobWhen(job: Job): string {
  return job.updated_at ?? job.created_at ?? "";
}

function groupJobsByDay(jobs: Job[]): { key: string; label: string; jobs: Job[] }[] {
  const map = new Map<string, Job[]>();
  for (const job of jobs) {
    const key = dayKey(jobWhen(job));
    const list = map.get(key) ?? [];
    list.push(job);
    map.set(key, list);
  }
  return Array.from(map.entries())
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([key, group]) => ({ key, label: dayLabel(key), jobs: group }));
}

class ConfirmModal extends Modal {
  constructor(
    app: PawnPlugin["app"],
    private titleText: string,
    private body: string,
    private onConfirm: () => void,
    private confirmLabel = "Delete",
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

export interface JobCardOptions {
  compact?: boolean;
  selectMode?: boolean;
  selected?: boolean;
  onToggleSelect?: () => void;
  onOpenConversation?: (conversation: string) => void;
  onDeleted?: () => void;
}

/** One job: status, title, result and actions. Shared by the chat thread and the jobs tab. */
export function renderJobCard(
  parent: HTMLElement,
  job: Job,
  plugin: PawnPlugin,
  component: Component,
  opts: JobCardOptions = {},
): HTMLElement {
  const card = parent.createDiv({ cls: `pawn-job-card is-${job.status}` });
  card.dataset.jobId = job.id;
  const head = card.createDiv({ cls: "pawn-job-head" });
  if (opts.selectMode) {
    const check = head.createEl("input", {
      type: "checkbox",
      cls: "pawn-item-check",
    });
    check.checked = Boolean(opts.selected);
    check.onclick = (ev) => ev.stopPropagation();
    check.onchange = () => opts.onToggleSelect?.();
  }
  const icon = head.createSpan({ cls: "pawn-job-icon" });
  setIcon(icon, KIND_ICON[job.kind] ?? "bot");
  head.createSpan({ cls: "pawn-job-title", text: job.title || job.id });
  const status = head.createSpan({
    cls: `pawn-job-status is-${job.status}`,
    text: job.approved ? "approved" : STATUS_LABEL[job.status] ?? job.status,
  });
  if (isActive(job)) status.addClass("is-active");

  const meta = card.createDiv({ cls: "pawn-job-meta" });
  const bits = [relTime(jobWhen(job))];
  if (!opts.compact && job.conversation) bits.push(conversationLabel(job.conversation));
  if (job.kind && job.kind !== "ask") bits.push(KIND_LABEL[job.kind] ?? job.kind);
  if (job.offline) bits.push("offline task note");
  meta.setText(bits.filter(Boolean).join(" · "));

  const result = stripToolTrail(job.result ?? "");
  if (result && !isActive(job)) {
    const body = card.createDiv({ cls: "pawn-job-result markdown-rendered" });
    if (job.status === "blocked") body.addClass("is-error");
    void MarkdownRenderer.render(plugin.app, result, body, job.note_path ?? "", component);
  }

  const extra: ExtraAction[] = [];
  if (job.kind === "ask" && job.status === "review" && !job.approved) {
    extra.push({
      icon: "check-circle",
      label: "Approve (index into Pawn memory)",
      onClick: () => void plugin.jobs.approve(job),
    });
    extra.push({
      icon: "circle-x",
      label: "Dismiss (close without indexing)",
      onClick: () => void plugin.jobs.dismiss(job),
    });
  }
  if (isActive(job) && !job.offline) {
    extra.push({ icon: "square", label: "Cancel job", onClick: () => void plugin.jobs.cancel(job) });
  }
  if (isDeletable(job)) {
    extra.push({
      icon: "trash-2",
      label: "Delete job",
      onClick: () => {
        void plugin.jobs.deleteIds([job.id]).then((n) => {
          if (n) new Notice("Deleted.", 3000);
          opts.onDeleted?.();
        });
      },
    });
  }
  if (job.task_key && plugin.app.vault.getAbstractFileByPath(job.task_key) instanceof TFile) {
    extra.push({
      icon: "file-search",
      label: "Open task note",
      onClick: () => void plugin.app.workspace.openLinkText(job.task_key!, "", true),
    });
  }
  if (!opts.compact && job.conversation && opts.onOpenConversation) {
    extra.push({
      icon: "messages-square",
      label: "Open conversation",
      onClick: () => opts.onOpenConversation?.(job.conversation),
    });
  }

  if (result && job.kind === "ask" && job.status !== "blocked") {
    renderMessageActions(card, {
      app: plugin.app,
      text: result,
      notePath: job.note_path ?? undefined,
      saveFolder: `${plugin.settings.agentRoot}/Notes`,
      extra,
    });
  } else if (extra.length) {
    const bar = card.createDiv({ cls: "pawn-msg-actions" });
    for (const a of extra) {
      const b = bar.createEl("button", { cls: "clickable-icon", attr: { "aria-label": a.label } });
      setIcon(b, a.icon);
      b.onclick = a.onClick;
    }
  }
  return card;
}

interface JobsUiState {
  selectMode: boolean;
  selected: Set<string>;
}

const uiState: JobsUiState = {
  selectMode: false,
  selected: new Set(),
};

/** Jobs tab content. */
export function renderJobsList(
  parent: HTMLElement,
  plugin: PawnPlugin,
  component: Component,
  opts: {
    conversation: string | null;
    focusJob?: string | null;
    /** Ask the host to re-render (filters live on JobStore). */
    onFilter?: () => void;
    onOpenConversation: (conversation: string) => void;
  },
): void {
  const state = uiState;
  const store = plugin.jobs;
  const refresh = () => {
    if (opts.onFilter) opts.onFilter();
    else void store.refresh();
  };

  const filters = parent.createDiv({ cls: "pawn-inbox-filters" });
  const scopes = filters.createDiv({ cls: "pawn-inbox-scopes" });
  for (const [scope, label] of [
    ["all", "All"],
    ["active", "Running"],
    ["review", "Review"],
    ["done", "Done"],
    ["blocked", "Failed"],
  ] as const) {
    const chip = scopes.createEl("button", { text: label, cls: "pawn-chip" });
    chip.toggleClass("is-active", store.scope === scope);
    chip.onclick = () => {
      store.setQuery({ scope: scope as JobsStatusScope });
      refresh();
    };
  }
  const convChip = scopes.createEl("button", {
    text: "This conversation",
    cls: "pawn-chip",
  });
  convChip.toggleClass("is-active", store.conversationOnly);
  convChip.disabled = !opts.conversation;
  convChip.onclick = () => {
    store.setQuery({ conversationOnly: !store.conversationOnly });
    refresh();
  };

  const search = filters.createEl("input", {
    type: "search",
    placeholder: "Search jobs…",
    cls: "pawn-inbox-search",
  });
  search.value = store.q;
  search.oninput = () => store.setQuery({ q: search.value });
  const applySearch = () => {
    store.setQuery({ q: search.value });
    refresh();
  };
  search.onchange = applySearch;
  search.onkeydown = (ev) => {
    if (ev.key === "Enter") applySearch();
  };

  const kinds = filters.createDiv({ cls: "pawn-inbox-kinds" });
  for (const kind of KIND_FILTERS) {
    const label = kind ? KIND_LABEL[kind] ?? kind : "All kinds";
    const chip = kinds.createEl("button", {
      text: label,
      cls: `pawn-chip${kind ? ` is-job-kind-${kind.replace(/_/g, "-")}` : ""}`,
    });
    chip.toggleClass("is-active", store.kind === kind);
    chip.onclick = () => {
      store.setQuery({ kind });
      refresh();
    };
  }

  const flushable = store.flushableCount();
  const flushing = store.isFlushing();
  if (flushable > 0 || flushing) {
    const flushRow = parent.createDiv({ cls: "pawn-inbox-flush" });
    flushRow.createSpan({
      cls: "pawn-inbox-hint",
      text: flushing
        ? "Flushing finished jobs…"
        : `${flushable} done/failed job(s) can be flushed.`,
    });
    const flushBtn = flushRow.createEl("button", {
      cls: "clickable-icon pawn-item-action",
      attr: {
        "aria-label": flushing ? "Flushing…" : `Flush ${flushable} finished jobs`,
        title: flushing ? "Flushing…" : `Flush ${flushable} finished jobs`,
      },
    });
    setIcon(flushBtn, flushing ? "loader" : "archive");
    flushBtn.toggleClass("is-busy", flushing);
    flushBtn.disabled = flushing;
    if (!flushing) {
      flushBtn.onclick = () => {
        new ConfirmModal(
          plugin.app,
          "Flush finished jobs?",
          `Permanently delete ${flushable} done/failed job(s) and their task notes. Jobs waiting for review stay.`,
          () => {
            void store
              .flushTerminal()
              .then((n) => {
                state.selected.clear();
                state.selectMode = false;
                new Notice(`Flushed ${n}.`, 4000);
                refresh();
              })
              .catch((e) => new Notice(e instanceof Error ? e.message : String(e)));
          },
          "Flush",
        ).open();
      };
    }
  }

  const listed = store.filtered(opts.conversation);
  const listedIds = listed.map((j) => j.id);
  const allSelected =
    listedIds.length > 0 && listedIds.every((id) => state.selected.has(id));
  const bulk = parent.createDiv({ cls: "pawn-inbox-bulk" });
  const selectBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: {
      "aria-label": state.selectMode ? "Cancel select" : "Select",
      title: state.selectMode ? "Cancel select" : "Select",
    },
  });
  setIcon(selectBtn, state.selectMode ? "x" : "check-square");
  selectBtn.onclick = () => {
    state.selectMode = !state.selectMode;
    state.selected.clear();
    refresh();
  };
  const selectAllLabel = allSelected ? "Clear selection" : "Select all";
  const selectAllBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": selectAllLabel, title: selectAllLabel },
  });
  setIcon(selectAllBtn, "list-checks");
  selectAllBtn.disabled = listedIds.length === 0 || flushing;
  selectAllBtn.onclick = () => {
    if (allSelected) {
      state.selected.clear();
    } else {
      state.selectMode = true;
      state.selected = new Set(listedIds);
    }
    refresh();
  };
  const delSel = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: {
      "aria-label":
        state.selected.size > 0
          ? `Delete selected (${state.selected.size})`
          : "Delete selected",
      title:
        state.selected.size > 0
          ? `Delete selected (${state.selected.size})`
          : "Delete selected",
    },
  });
  setIcon(delSel, "trash-2");
  delSel.disabled = state.selected.size === 0 || flushing;
  delSel.onclick = () => {
    const ids = Array.from(state.selected);
    new ConfirmModal(
      plugin.app,
      "Delete selected jobs?",
      `Permanently delete ${ids.length} job(s). Running jobs are skipped — cancel them first.`,
      () => {
        void store
          .deleteIds(ids)
          .then((n) => {
            state.selected.clear();
            state.selectMode = false;
            new Notice(`Deleted ${n}.`, 4000);
            refresh();
          })
          .catch((e) => new Notice(e instanceof Error ? e.message : String(e)));
      },
    ).open();
  };
  const refreshBtn = bulk.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": "Refresh", title: "Refresh" },
  });
  setIcon(refreshBtn, "refresh-cw");
  refreshBtn.onclick = () => void store.refresh();

  if (!store.online) {
    parent.createDiv({
      cls: "pawn-inbox-offline",
      text: "Server offline — showing task notes from the vault when available.",
    });
  }

  const list = parent.createDiv({ cls: "pawn-jobs-list" });
  if (!listed.length) {
    list.createDiv({
      cls: "pawn-empty",
      text: store.all().length
        ? "No jobs match this filter."
        : "No jobs yet. Toggle “Background” in the composer, or use “Send to Pawn (background)”.",
    });
    return;
  }

  for (const group of groupJobsByDay(listed.slice(0, 100))) {
    list.createDiv({ cls: "pawn-inbox-day", text: group.label });
    for (const job of group.jobs) {
      const card = renderJobCard(list, job, plugin, component, {
        selectMode: state.selectMode,
        selected: state.selected.has(job.id),
        onToggleSelect: () => {
          if (state.selected.has(job.id)) state.selected.delete(job.id);
          else state.selected.add(job.id);
          refresh();
        },
        onOpenConversation: opts.onOpenConversation,
        onDeleted: refresh,
      });
      if (opts.focusJob === job.id) {
        card.addClass("is-focused");
        window.setTimeout(() => card.scrollIntoView({ block: "center" }), 50);
      }
    }
  }
}

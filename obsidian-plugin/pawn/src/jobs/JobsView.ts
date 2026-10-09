import { Component, MarkdownRenderer, TFile, setIcon } from "obsidian";
import type { Job } from "../api";
import { ExtraAction, renderMessageActions } from "../chat/MessageActions";
import { conversationLabel } from "../chat/conversations";
import type PawnPlugin from "../main";
import { isActive } from "./JobStore";

const KIND_ICON: Record<string, string> = {
  ask: "bot",
  push_note: "file-up",
  upload: "upload",
  capture_enrich: "tags",
};

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

export interface JobCardOptions {
  compact?: boolean;
  onOpenConversation?: (conversation: string) => void;
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
  const icon = head.createSpan({ cls: "pawn-job-icon" });
  setIcon(icon, KIND_ICON[job.kind] ?? "bot");
  head.createSpan({ cls: "pawn-job-title", text: job.title || job.id });
  const status = head.createSpan({
    cls: `pawn-job-status is-${job.status}`,
    text: job.approved ? "approved" : STATUS_LABEL[job.status] ?? job.status,
  });
  if (isActive(job)) status.addClass("is-active");

  const meta = card.createDiv({ cls: "pawn-job-meta" });
  const bits = [relTime(job.updated_at ?? job.created_at)];
  if (!opts.compact && job.conversation) bits.push(conversationLabel(job.conversation));
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

export type JobFilter = "all" | "conversation" | "active";

/** Jobs tab content. */
export function renderJobsList(
  parent: HTMLElement,
  plugin: PawnPlugin,
  component: Component,
  opts: {
    filter: JobFilter;
    conversation: string | null;
    focusJob?: string | null;
    onFilter: (f: JobFilter) => void;
    onOpenConversation: (conversation: string) => void;
  },
): void {
  const bar = parent.createDiv({ cls: "pawn-jobs-toolbar" });
  const filters: Array<[JobFilter, string]> = [
    ["all", "All"],
    ["active", "Running"],
    ["conversation", "This conversation"],
  ];
  for (const [id, label] of filters) {
    const b = bar.createEl("button", { text: label });
    b.toggleClass("is-active", opts.filter === id);
    b.onclick = () => opts.onFilter(id);
  }
  const refresh = bar.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": "Refresh" },
  });
  setIcon(refresh, "refresh-cw");
  refresh.onclick = () => void plugin.jobs.refresh();

  let jobs = plugin.jobs.all();
  if (opts.filter === "active") jobs = jobs.filter(isActive);
  if (opts.filter === "conversation") jobs = jobs.filter((j) => j.conversation === opts.conversation);

  const list = parent.createDiv({ cls: "pawn-jobs-list" });
  if (!jobs.length) {
    list.createDiv({
      cls: "pawn-empty",
      text: "No jobs yet. Toggle “Background” in the composer, or use “Send to Pawn (background)”.",
    });
    return;
  }
  for (const job of jobs.slice(0, 100)) {
    const card = renderJobCard(list, job, plugin, component, {
      onOpenConversation: opts.onOpenConversation,
    });
    if (opts.focusJob === job.id) {
      card.addClass("is-focused");
      window.setTimeout(() => card.scrollIntoView({ block: "center" }), 50);
    }
  }
}

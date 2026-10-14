import { Modal, Notice, Setting, setIcon } from "obsidian";
import type { GallerySpeaker, SessionDetail, SessionLabel } from "../api";
import { HttpError } from "../api";
import type PawnPlugin from "../main";

function fmtDur(seconds: number): string {
  if (!seconds || seconds < 0) return "0s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s ? `${m}m ${s}s` : `${m}m`;
}

class ConfirmEnrollModal extends Modal {
  constructor(
    app: PawnPlugin["app"],
    private label: SessionLabel,
    private speakerName: string,
    private onConfirm: () => void,
  ) {
    super(app);
  }

  onOpen(): void {
    const { contentEl } = this;
    contentEl.createEl("h3", { text: "Approve voiceprint?" });
    contentEl.createEl("p", {
      text:
        `Enroll ${this.label.local_label} → ${this.speakerName} ` +
        `(${fmtDur(this.label.duration_s)}). No audio preview — confirm from the transcript sample.`,
    });
    if (this.label.sample_text) {
      contentEl.createEl("blockquote", { text: this.label.sample_text });
    }
    const row = contentEl.createDiv({ cls: "pawn-job-actions" });
    const cancel = row.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const ok = row.createEl("button", { text: "Enroll", cls: "mod-cta" });
    ok.onclick = () => {
      this.onConfirm();
      this.close();
    };
  }
}

/** Map session labels to gallery speakers, enroll, reidentify, optional relabel. */
export class TrainModal extends Modal {
  private detail: SessionDetail | null = null;
  private busy = false;
  private picks = new Map<string, string>();

  constructor(
    private plugin: PawnPlugin,
    private sessionId: string,
    private onDone?: () => void,
  ) {
    super(plugin.app);
  }

  onOpen(): void {
    this.titleEl.setText(`Train speakers · ${this.sessionId}`);
    this.contentEl.createEl("p", { text: "Loading…" });
    void this.reload();
  }

  private async reload(): Promise<void> {
    try {
      this.detail = await this.plugin.client.getSession(this.sessionId);
      for (const label of this.detail.labels) {
        if (label.mapped_speaker_id && !this.picks.has(label.local_label)) {
          this.picks.set(label.local_label, label.mapped_speaker_id);
        }
      }
    } catch (e) {
      this.contentEl.empty();
      this.contentEl.createEl("p", {
        text: e instanceof Error ? e.message : String(e),
        cls: "mod-warning",
      });
      return;
    }
    this.renderBody();
  }

  private renderBody(): void {
    const detail = this.detail;
    if (!detail) return;
    const { contentEl } = this;
    contentEl.empty();

    contentEl.createEl("p", {
      text: detail.title || detail.session_id,
      cls: "pawn-session-train-title",
    });

    const list = contentEl.createDiv({ cls: "pawn-session-train-labels" });
    for (const label of detail.labels) {
      this.renderLabelRow(list, label, detail.gallery);
    }

    const actions = contentEl.createDiv({ cls: "pawn-job-actions" });
    const reidentify = actions.createEl("button", {
      text: "Reidentify session",
      cls: "mod-cta",
    });
    reidentify.disabled = this.busy;
    reidentify.onclick = () => void this.runReidentify();

    const trained = actions.createEl("button", {
      text: detail.speakers_trained ? "Clear trained" : "Mark session trained",
    });
    trained.disabled = this.busy;
    trained.onclick = () => void this.runMarkTrained(!(detail.speakers_trained ?? false));

    const refresh = actions.createEl("button", { text: "Refresh" });
    refresh.disabled = this.busy;
    refresh.onclick = () => void this.reload();

    if (detail.speakers_trained) {
      contentEl.createEl("p", {
        cls: "pawn-session-meta",
        text: "Speakers marked trained — needs-training badge is suppressed.",
      });
    }
  }

  private renderLabelRow(
    parent: HTMLElement,
    label: SessionLabel,
    gallery: GallerySpeaker[],
  ): void {
    const row = parent.createDiv({ cls: "pawn-session-train-row" });
    const head = row.createDiv({ cls: "pawn-session-train-head" });
    head.createSpan({ cls: "pawn-chip is-active", text: label.local_label });
    head.createSpan({
      cls: "pawn-session-meta",
      text: fmtDur(label.duration_s),
    });
    if (label.ignored) {
      head.createSpan({ cls: "pawn-session-badge", text: "ignored" });
    } else if (!label.has_centroid) {
      head.createSpan({ cls: "pawn-session-badge is-warn", text: "no centroid" });
    } else if (label.mapped_name) {
      head.createSpan({ cls: "pawn-session-badge", text: label.mapped_name });
    }

    if (label.sample_text) {
      row.createDiv({ cls: "pawn-session-sample", text: label.sample_text });
    }

    new Setting(row)
      .setName("Map to")
      .addDropdown((dd) => {
        dd.addOption("", "— select —");
        for (const sp of gallery) {
          dd.addOption(sp.id, `${sp.display_name} (${sp.enrollments})`);
        }
        dd.addOption("__create__", "Create speaker…");
        const current = this.picks.get(label.local_label) || label.mapped_speaker_id || "";
        if (current && gallery.some((g) => g.id === current)) dd.setValue(current);
        dd.onChange(async (v) => {
          if (v === "__create__") {
            const name = await this.promptName();
            if (!name) {
              dd.setValue(this.picks.get(label.local_label) || "");
              return;
            }
            try {
              const created = await this.plugin.client.createSpeaker(name);
              this.picks.set(label.local_label, created.id);
              await this.reload();
            } catch (e) {
              new Notice(e instanceof Error ? e.message : String(e), 5000);
              dd.setValue(this.picks.get(label.local_label) || "");
            }
            return;
          }
          this.picks.set(label.local_label, v);
        });
      })
      .addButton((btn) => {
        btn.setButtonText("Enroll");
        btn.setDisabled(!label.has_centroid || this.busy);
        btn.onClick(() => void this.runEnroll(label));
      })
      .addExtraButton((btn) => {
        btn.setIcon("pencil");
        btn.setTooltip("Rename label only (not a voiceprint)");
        btn.onClick(() => void this.runRelabel(label));
      })
      .addExtraButton((btn) => {
        btn.setIcon(label.ignored ? "eye" : "eye-off");
        btn.setTooltip(
          label.ignored
            ? "Stop ignoring this label"
            : "Ignore as noise (skip needs-training)",
        );
        btn.onClick(() => void this.runIgnore(label, !label.ignored));
      });
  }

  private async promptName(): Promise<string | null> {
    return new Promise((resolve) => {
      const modal = new Modal(this.app);
      modal.titleEl.setText("Create speaker");
      let value = "";
      new Setting(modal.contentEl).setName("Display name").addText((t) => {
        t.setPlaceholder("Davide");
        t.onChange((v) => {
          value = v;
        });
        t.inputEl.focus();
      });
      const row = modal.contentEl.createDiv({ cls: "pawn-job-actions" });
      row.createEl("button", { text: "Cancel" }).onclick = () => {
        modal.close();
        resolve(null);
      };
      const ok = row.createEl("button", { text: "Create", cls: "mod-cta" });
      ok.onclick = () => {
        modal.close();
        resolve(value.trim() || null);
      };
      modal.open();
    });
  }

  private async runEnroll(label: SessionLabel): Promise<void> {
    const speaker =
      this.picks.get(label.local_label) ||
      label.mapped_speaker_id ||
      label.mapped_name ||
      "";
    if (!speaker) {
      new Notice("Select or create a speaker first.", 4000);
      return;
    }
    if (!label.has_centroid) {
      new Notice("No session centroid for this label — reprocess the session first.", 5000);
      return;
    }
    const galleryName =
      this.detail?.gallery.find((g) => g.id === speaker)?.display_name || speaker;
    new ConfirmEnrollModal(this.app, label, galleryName, () => {
      void this.doEnroll(label, speaker);
    }).open();
  }

  private async doEnroll(label: SessionLabel, speaker: string): Promise<void> {
    this.busy = true;
    try {
      const result = await this.plugin.client.enrollSessionSpeaker(this.sessionId, {
        speaker,
        from_label: label.local_label,
      });
      new Notice(
        `Enrolled ${result.display_name} (${fmtDur(result.duration)}).`,
        4000,
      );
      await this.reload();
    } catch (e) {
      if (e instanceof HttpError && /too short|pairwise|max enrollments/i.test(e.message)) {
        const force = window.confirm(`${e.message}\n\nForce enroll anyway?`);
        if (force) {
          try {
            await this.plugin.client.enrollSessionSpeaker(this.sessionId, {
              speaker,
              from_label: label.local_label,
              force: true,
            });
            new Notice("Enrolled with force.", 4000);
            await this.reload();
            return;
          } catch (e2) {
            new Notice(e2 instanceof Error ? e2.message : String(e2), 6000);
          }
        }
      } else {
        new Notice(e instanceof Error ? e.message : String(e), 6000);
      }
    } finally {
      this.busy = false;
      this.renderBody();
    }
  }

  private async runReidentify(): Promise<void> {
    this.busy = true;
    this.renderBody();
    try {
      const result = await this.plugin.client.reidentifySession(this.sessionId);
      new Notice(result.summary || "Reidentified.", 5000);
      await this.reload();
      this.onDone?.();
    } catch (e) {
      new Notice(e instanceof Error ? e.message : String(e), 6000);
    } finally {
      this.busy = false;
      this.renderBody();
    }
  }

  private async runRelabel(label: SessionLabel): Promise<void> {
    const to = window.prompt(
      `Rename label only (not a voiceprint).\nFrom: ${label.local_label}\nTo:`,
      label.mapped_name || "",
    );
    if (!to?.trim()) return;
    this.busy = true;
    try {
      const result = await this.plugin.client.relabelSession(this.sessionId, {
        from: label.local_label,
        to: to.trim(),
      });
      new Notice(result.summary || "Relabeled.", 4000);
      await this.reload();
      this.onDone?.();
    } catch (e) {
      new Notice(e instanceof Error ? e.message : String(e), 6000);
    } finally {
      this.busy = false;
      this.renderBody();
    }
  }

  private async runIgnore(label: SessionLabel, ignore: boolean): Promise<void> {
    this.busy = true;
    try {
      await this.plugin.client.ignoreSessionSpeaker(this.sessionId, {
        from: label.local_label,
        ignore,
      });
      new Notice(
        ignore
          ? `Ignored ${label.local_label} as noise.`
          : `Stopped ignoring ${label.local_label}.`,
        4000,
      );
      await this.reload();
      this.onDone?.();
    } catch (e) {
      new Notice(e instanceof Error ? e.message : String(e), 6000);
    } finally {
      this.busy = false;
      this.renderBody();
    }
  }

  private async runMarkTrained(trained: boolean): Promise<void> {
    this.busy = true;
    this.renderBody();
    try {
      await this.plugin.client.setSessionSpeakersTrained(this.sessionId, { trained });
      new Notice(
        trained
          ? "Session marked trained — needs-training cleared."
          : "Trained flag cleared.",
        4000,
      );
      await this.reload();
      this.onDone?.();
    } catch (e) {
      new Notice(e instanceof Error ? e.message : String(e), 6000);
    } finally {
      this.busy = false;
      this.renderBody();
    }
  }
}

/** Tiny icon button helper used by SessionsView. */
export function iconButton(
  parent: HTMLElement,
  icon: string,
  label: string,
  onClick: () => void,
): HTMLElement {
  const b = parent.createEl("button", {
    cls: "clickable-icon",
    attr: { "aria-label": label },
  });
  setIcon(b, icon);
  b.onclick = (ev) => {
    ev.stopPropagation();
    onClick();
  };
  return b;
}

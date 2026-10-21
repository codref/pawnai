import { Modal, Notice, Setting } from "obsidian";
import { HttpError } from "../api";
import type PawnPlugin from "../main";

type ReprocessMode = "rediarize" | "retranscribe";
type AsrBackend = "nemo" | "whisper";

/** Queue rediarize (default) or full retranscribe for a session. */
export class ReprocessModal extends Modal {
  private mode: ReprocessMode = "rediarize";
  private backend: AsrBackend = "nemo";
  private language = "";
  private threshold = 0.7;
  private crossFileThreshold = 0.55;
  private margin = "";
  private busy = false;

  constructor(
    private plugin: PawnPlugin,
    private sessionId: string,
    private onDone?: () => void,
  ) {
    super(plugin.app);
  }

  onOpen(): void {
    this.titleEl.setText(`Reprocess · ${this.sessionId}`);
    this.renderBody();
  }

  private renderBody(): void {
    const { contentEl } = this;
    contentEl.empty();

    contentEl.createEl("p", {
      text:
        "Rediarize keeps ASR text and re-runs speaker labels. Retranscribe wipes " +
        "segments and rebuilds text + labels from stored audio. The vault note " +
        "refreshes when the diarize worker finishes.",
    });

    new Setting(contentEl)
      .setName("Mode")
      .setDesc("Rediarize is safer and usually enough for speaker fixes.")
      .addDropdown((dd) => {
        dd.addOption("rediarize", "Rediarize (keep text)");
        dd.addOption("retranscribe", "Retranscribe (wipe + ASR)");
        dd.setValue(this.mode);
        dd.onChange((v) => {
          this.mode = v as ReprocessMode;
          this.renderBody();
        });
      });

    if (this.mode === "retranscribe") {
      contentEl.createEl("p", {
        cls: "mod-warning",
        text:
          "Retranscribe deletes existing segments, analyses, and speaker maps " +
          "for this session before re-running ASR + diarization.",
      });

      new Setting(contentEl)
        .setName("ASR backend")
        .setDesc("Parakeet (nemo) or Whisper. Diarization backend stays from server config.")
        .addDropdown((dd) => {
          dd.addOption("nemo", "Parakeet (nemo)");
          dd.addOption("whisper", "Whisper");
          dd.setValue(this.backend);
          dd.onChange((v) => {
            this.backend = v as AsrBackend;
            this.renderBody();
          });
        });

      if (this.backend === "whisper") {
        new Setting(contentEl)
          .setName("Language")
          .setDesc("Whisper language lock (e.g. it). Leave empty for auto-detect.")
          .addText((t) => {
            t.setPlaceholder("it");
            t.setValue(this.language);
            t.onChange((v) => {
              this.language = v.trim();
            });
          });
      }
    }

    new Setting(contentEl)
      .setName("Identify threshold")
      .setDesc("Gallery cosine threshold (0–1). Default 0.7.")
      .addText((t) => {
        t.setValue(String(this.threshold));
        t.onChange((v) => {
          const n = Number(v);
          if (!Number.isNaN(n)) this.threshold = n;
        });
      });

    new Setting(contentEl)
      .setName("Cross-file threshold")
      .setDesc("Merge same voice across chunk files (0–1). Default 0.55.")
      .addText((t) => {
        t.setValue(String(this.crossFileThreshold));
        t.onChange((v) => {
          const n = Number(v);
          if (!Number.isNaN(n)) this.crossFileThreshold = n;
        });
      });

    if (this.mode === "rediarize") {
      new Setting(contentEl)
        .setName("Identify margin")
        .setDesc("Optional; leave empty to use server speakers.identify_margin.")
        .addText((t) => {
          t.setPlaceholder("0.05");
          t.setValue(this.margin);
          t.onChange((v) => {
            this.margin = v.trim();
          });
        });
    }

    const actions = contentEl.createDiv({ cls: "pawn-job-actions" });
    const cancel = actions.createEl("button", { text: "Cancel" });
    cancel.onclick = () => this.close();
    const ok = actions.createEl("button", {
      text: this.mode === "rediarize" ? "Queue rediarize" : "Queue retranscribe",
      cls: "mod-cta",
    });
    ok.disabled = this.busy;
    ok.onclick = () => void this.submit();
  }

  private async submit(): Promise<void> {
    if (this.busy) return;
    const wipeWarn =
      this.mode === "retranscribe"
        ? `\n\nThis will WIPE and rebuild transcript text for ${this.sessionId}.`
        : "";
    const ok = window.confirm(
      `Queue ${this.mode} for session ${this.sessionId}?${wipeWarn}`,
    );
    if (!ok) return;

    this.busy = true;
    this.renderBody();
    try {
      const body: Parameters<PawnPlugin["client"]["reprocessSession"]>[1] = {
        confirm: true,
        mode: this.mode,
        threshold: this.threshold,
        cross_file_threshold: this.crossFileThreshold,
      };
      if (this.mode === "retranscribe") {
        body.backend = this.backend;
        if (this.backend === "whisper" && this.language) {
          body.language = this.language;
        }
      }
      if (this.mode === "rediarize" && this.margin) {
        const m = Number(this.margin);
        if (!Number.isNaN(m)) body.margin = m;
      }
      const result = await this.plugin.client.reprocessSession(this.sessionId, body);
      new Notice(
        result.result ??
          `Queued ${result.command}. Check Jobs; transcript refreshes when done.`,
        6000,
      );
      this.onDone?.();
      this.close();
    } catch (e) {
      const msg =
        e instanceof HttpError
          ? e.message
          : e instanceof Error
            ? e.message
            : String(e);
      new Notice(msg, 7000);
      this.busy = false;
      this.renderBody();
    }
  }
}

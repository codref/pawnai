import { App, PluginSettingTab, Setting } from "obsidian";
import type PawnPlugin from "./main";

export type ConversationMode = "note" | "global";

export interface PawnSettings {
  serverUrl: string;
  apiToken: string;
  agentRoot: string;
  conversationMode: ConversationMode;
  autoIncludeActiveNote: boolean;
  sendLocalNoteContent: boolean;
  commandsFolder: string;
  notifyOnJobDone: boolean;
  insertCalloutForJobs: boolean;
  resyncOnAgentVaultWrite: boolean;
}

export const DEFAULT_SETTINGS: PawnSettings = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
  agentRoot: "Pawn",
  conversationMode: "note",
  autoIncludeActiveNote: true,
  sendLocalNoteContent: true,
  commandsFolder: "Pawn/Commands",
  notifyOnJobDone: true,
  insertCalloutForJobs: false,
  resyncOnAgentVaultWrite: true,
};

export class PawnSettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: PawnPlugin) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();

    new Setting(containerEl).setName("Connection").setHeading();

    new Setting(containerEl)
      .setName("Server URL")
      .setDesc("pawn-server base URL (no trailing slash).")
      .addText((t) =>
        t
          .setPlaceholder("http://127.0.0.1:8000")
          .setValue(s.serverUrl)
          .onChange(async (v) => {
            s.serverUrl = v.trim().replace(/\/+$/, "");
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("API token")
      .setDesc("Bearer token (pawnai.yaml api.token). Leave empty if the server is open.")
      .addText((t) => {
        t.inputEl.type = "password";
        t.setValue(s.apiToken).onChange(async (v) => {
          s.apiToken = v.trim();
          await this.plugin.saveSettings();
        });
      });

    new Setting(containerEl).setName("Chat").setHeading();

    new Setting(containerEl)
      .setName("Default conversation")
      .setDesc(
        "Per note: each note has its own Pawn conversation (note:<path>). " +
          "Global: one free-standing chat you switch manually.",
      )
      .addDropdown((d) =>
        d
          .addOption("note", "Per note")
          .addOption("global", "Global chat")
          .setValue(s.conversationMode)
          .onChange(async (v) => {
            s.conversationMode = v as ConversationMode;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Include active note")
      .setDesc("Attach the open note by default. Unpin it from the composer chip, and pin it to attach it again.")
      .addToggle((t) =>
        t.setValue(s.autoIncludeActiveNote).onChange(async (v) => {
          s.autoIncludeActiveNote = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Send local note content")
      .setDesc(
        "Send note bodies from this device (includes unsynced edits). " +
          "Off: the server reads notes from the vault bucket.",
      )
      .addToggle((t) =>
        t.setValue(s.sendLocalNoteContent).onChange(async (v) => {
          s.sendLocalNoteContent = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Prompt commands folder")
      .setDesc("Markdown files here become commands (palette, editor menu, / in chat).")
      .addText((t) =>
        t
          .setPlaceholder("Pawn/Commands")
          .setValue(s.commandsFolder)
          .onChange(async (v) => {
            s.commandsFolder = v.trim().replace(/\/+$/, "") || "Pawn/Commands";
            await this.plugin.saveSettings();
            await this.plugin.prompts.reload();
          }),
      )
      .addButton((b) =>
        b.setButtonText("Create defaults").onClick(async () => {
          await this.plugin.prompts.writeDefaults();
        }),
      );

    new Setting(containerEl).setName("Background jobs").setHeading();

    new Setting(containerEl)
      .setName("Agent root")
      .setDesc("Vault folder Pawn owns (task notes live in <root>/Tasks).")
      .addText((t) =>
        t
          .setPlaceholder("Pawn")
          .setValue(s.agentRoot)
          .onChange(async (v) => {
            s.agentRoot = v.trim() || "Pawn";
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Notify when a job finishes")
      .addToggle((t) =>
        t.setValue(s.notifyOnJobDone).onChange(async (v) => {
          s.notifyOnJobDone = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Resync when the agent writes")
      .setDesc(
        "After an agent turn changes vault notes, ask Sync Engine to sync immediately. " +
          "Interval sync still applies when this is off or Sync Engine is disabled.",
      )
      .addToggle((t) =>
        t.setValue(s.resyncOnAgentVaultWrite).onChange(async (v) => {
          s.resyncOnAgentVaultWrite = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Insert callout for background jobs")
      .setDesc("Add a > [!pawn] link to the task note at the cursor when you send a job.")
      .addToggle((t) =>
        t.setValue(s.insertCalloutForJobs).onChange(async (v) => {
          s.insertCalloutForJobs = v;
          await this.plugin.saveSettings();
        }),
      );
  }
}

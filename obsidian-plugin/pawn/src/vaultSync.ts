import { debounce } from "obsidian";
import { PawnClient } from "./api";
import type PawnPlugin from "./main";

const SYNC_COMMAND = "sync-engine:start-non-interactive-sync";
const DEBOUNCE_MS = 1500;
const RETRY_MS = 5000;
const MAX_BACKOFF_MS = 60000;

/** Obsidian's command manager is runtime-public but absent from the plugin typedef. */
interface CommandHost {
  commands: Record<string, unknown>;
  executeCommandById(id: string): boolean;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/** Long-poll vault writes and ask Sync Engine to pull them. */
export class VaultSync {
  private stopped = true;
  private retry: number | null = null;
  private readonly kick = debounce(() => this.runSync(), DEBOUNCE_MS, true);

  constructor(
    private plugin: PawnPlugin,
    private client: PawnClient,
  ) {}

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    void this.loop();
  }

  stop(): void {
    this.stopped = true;
    this.kick.cancel();
    if (this.retry != null) window.clearTimeout(this.retry);
    this.retry = null;
  }

  private async loop(): Promise<void> {
    let since = 0;
    let backoff = 2000;
    while (!this.stopped) {
      if (!this.plugin.settings.resyncOnAgentVaultWrite) {
        await sleep(1000);
        continue;
      }
      const started = Date.now();
      try {
        const page = await this.client.waitForVaultEvents(since);
        since = page.seq;
        if (page.resync || page.events.length > 0) this.kick();
        if (Date.now() - started > 5000) backoff = 2000;
      } catch {
        if (this.stopped) return;
        await sleep(backoff);
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      }
    }
  }

  private runSync(): void {
    if (this.stopped || !this.plugin.settings.resyncOnAgentVaultWrite) return;
    const mgr = (this.plugin.app as unknown as { commands?: CommandHost }).commands;
    if (!mgr?.commands?.[SYNC_COMMAND]) return;
    if (mgr.executeCommandById(SYNC_COMMAND)) return;
    // Sync Engine ignores the command while a sync is already running.
    if (this.retry != null) window.clearTimeout(this.retry);
    this.retry = window.setTimeout(() => {
      this.retry = null;
      if (this.stopped || !this.plugin.settings.resyncOnAgentVaultWrite) return;
      mgr.executeCommandById(SYNC_COMMAND);
    }, RETRY_MS);
  }
}

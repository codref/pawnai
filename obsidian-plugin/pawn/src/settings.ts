export interface PawnSettings {
  serverUrl: string;
  apiToken: string;
  agentRoot: string;
  fastPathTimeoutMs: number;
  alwaysQueue: boolean;
}

export const DEFAULT_SETTINGS: PawnSettings = {
  serverUrl: "http://127.0.0.1:8000",
  apiToken: "",
  agentRoot: "Pawn",
  fastPathTimeoutMs: 60000,
  alwaysQueue: false,
};

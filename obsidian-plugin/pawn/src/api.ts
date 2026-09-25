import { requestUrl } from "obsidian";
import type { PawnSettings } from "./settings";

export interface VaultTaskResponse {
  task_id: string;
  id?: string;
  status: string;
  result?: string;
  conversation_id?: string;
  agent_run_id?: string | null;
  error_code?: string | null;
}

function authHeaders(settings: PawnSettings): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (settings.apiToken) {
    headers.Authorization = `Bearer ${settings.apiToken}`;
  }
  return headers;
}

function baseUrl(settings: PawnSettings): string {
  return settings.serverUrl.replace(/\/+$/, "");
}

export async function healthCheck(settings: PawnSettings): Promise<boolean> {
  try {
    const res = await requestUrl({
      url: `${baseUrl(settings)}/health`,
      method: "GET",
      throw: false,
    });
    return res.status >= 200 && res.status < 300;
  } catch {
    return false;
  }
}

export async function createTask(
  settings: PawnSettings,
  body: {
    id?: string;
    instruction: string;
    note_path?: string;
    context?: string;
    conversation?: string;
    timeout_seconds?: number;
  },
): Promise<{ status: number; data: VaultTaskResponse }> {
  const res = await requestUrl({
    url: `${baseUrl(settings)}/v1/vault/tasks`,
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify(body),
    throw: false,
  });
  let data: VaultTaskResponse;
  try {
    data = res.json as VaultTaskResponse;
  } catch {
    data = { task_id: body.id || "", status: "blocked", error_code: "bad_response" };
  }
  return { status: res.status, data };
}

export async function getTask(
  settings: PawnSettings,
  id: string,
): Promise<VaultTaskResponse | null> {
  try {
    const res = await requestUrl({
      url: `${baseUrl(settings)}/v1/vault/tasks/${encodeURIComponent(id)}`,
      method: "GET",
      headers: authHeaders(settings),
      throw: false,
    });
    if (res.status === 404) return null;
    return res.json as VaultTaskResponse;
  } catch {
    return null;
  }
}

export async function approveTask(
  settings: PawnSettings,
  id: string,
  result?: string,
): Promise<VaultTaskResponse> {
  const res = await requestUrl({
    url: `${baseUrl(settings)}/v1/vault/tasks/${encodeURIComponent(id)}/approve`,
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify(result ? { result } : {}),
    throw: false,
  });
  return res.json as VaultTaskResponse;
}

export async function chatCompletions(
  settings: PawnSettings,
  opts: {
    messages: { role: string; content: string }[];
    user: string;
  },
): Promise<string> {
  const res = await requestUrl({
    url: `${baseUrl(settings)}/v1/chat/completions`,
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify({
      model: "pawn",
      messages: opts.messages,
      user: opts.user,
      stream: false,
    }),
  });
  const data = res.json as {
    choices?: { message?: { content?: string } }[];
  };
  return data.choices?.[0]?.message?.content ?? "";
}

// Single PortalApi instance for the app: the HTTP client, or the in-browser
// mock when the build sets VITE_PAWN_MOCK=1.

import { HttpApi } from "./client";
import type { PortalApi } from "./types";

let current: PortalApi | null = null;

export const MOCK_ENABLED = import.meta.env.VITE_PAWN_MOCK === "1";

export async function initApi(token: () => string, onUnauthorized: () => void): Promise<PortalApi> {
  if (MOCK_ENABLED) {
    const { MockApi } = await import("./mock");
    const mock = new MockApi();
    (window as unknown as { pawnMock: unknown }).pawnMock = mock;
    current = mock;
  } else {
    current = new HttpApi(token, onUnauthorized);
  }
  return current;
}

export function api(): PortalApi {
  if (!current) throw new Error("Portal API used before initApi()");
  return current;
}

export * from "./types";

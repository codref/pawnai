import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SessionSummary, SessionsPage } from "@/api";

const listSessions = vi.fn();

vi.mock("@/api", () => ({
  api: () => ({ listSessions }),
}));

import { useSessionsStore } from "./sessions";

function row(id: string): SessionSummary {
  return { session_id: id, title: id };
}

function page(data: SessionSummary[], total: number, needs = 4): SessionsPage {
  return {
    data,
    facets: { speakers: [{ name: "Ada", count: 2 }], needs_speakers: needs, tags: [], total },
  };
}

describe("sessions store", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    listSessions.mockReset();
  });

  it("pages until the facet total is reached and keeps the unfiltered needs count", async () => {
    listSessions.mockImplementation(async (params: { offset?: number }) => {
      const offset = params.offset ?? 0;
      const data = Array.from({ length: offset === 0 ? 40 : 10 }, (_, i) => row(`s${offset + i}`));
      return page(data, 50, 4);
    });
    const store = useSessionsStore();
    await store.reload();
    expect(store.rows).toHaveLength(40);
    expect(store.done).toBe(false);
    expect(store.needsSpeakers).toBe(4);
    expect(store.facets.speakers[0]?.name).toBe("Ada");

    await store.more();
    expect(store.rows).toHaveLength(50);
    expect(store.rows[49]?.session_id).toBe("s49");
    expect(store.done).toBe(true);
  });

  it("ignores a stale page when a newer reload finishes first", async () => {
    let release: (page: SessionsPage) => void = () => undefined;
    const slow = new Promise<SessionsPage>((resolve) => {
      release = resolve;
    });
    listSessions.mockImplementationOnce(() => slow);
    listSessions.mockImplementationOnce(async () => page([row("new")], 1, 1));
    const store = useSessionsStore();
    const first = store.reload();
    const second = store.reload();
    release(page([row("old")], 1, 9));
    await Promise.all([first, second]);
    expect(store.rows.map((item) => item.session_id)).toEqual(["new"]);
    expect(store.needsSpeakers).toBe(1);
  });

  it("sends the speaker, the needs filter, and today's range", async () => {
    listSessions.mockResolvedValue(page([], 0, 2));
    const store = useSessionsStore();
    store.scope = "needs";
    store.range = "today";
    store.setSpeaker("Ada");
    await vi.waitFor(() => expect(listSessions).toHaveBeenCalled());
    const query = listSessions.mock.calls.at(-1)?.[0] as { speaker?: string; needs_speakers?: boolean; from?: string; to?: string };
    expect(query.speaker).toBe("Ada");
    expect(query.needs_speakers).toBe(true);
    expect(query.from).toBe(query.to);
    expect(query.from).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    store.setSpeaker("Ada");
    await vi.waitFor(() => expect(listSessions.mock.calls.length).toBeGreaterThan(1));
    const cleared = listSessions.mock.calls.at(-1)?.[0] as { speaker?: string };
    expect(cleared.speaker).toBeUndefined();
  });
});

// Mock user boards (VITE_PAWN_MOCK=1): CRUD, pins, note cards and proposals
// over the mock emergence web. A board's graph is the items-board graph cut
// down to its pins (plus one hop of strings), like the server's manual board.
import {
  HttpError,
  type Board,
  type BoardCreate,
  type BoardPatch,
  type BoardProposal,
  type BrainGraph,
  type BrainNode,
} from "./types";

const iso = () => new Date().toISOString();
const short = () => Math.random().toString(16).slice(2, 12);

export class MockBoards {
  private boards = new Map<string, Board>();
  private proposals = new Map<string, BoardProposal>();

  constructor(private readonly base: () => BrainGraph) {
    const now = iso();
    const seed = this.base();
    const pins = seed.nodes.filter((n) => n.kind === "thread").slice(0, 2).map((n) => n.id);
    this.boards.set("b-mockvendor", {
      name: "b-mockvendor",
      title: "Vendor renewal risk",
      kind: "manual",
      spec: {},
      pins,
      hidden: [],
      notes: [],
      folder_id: null,
      tags: ["work"],
      pinned: false,
      archived: false,
      chat_id: null,
      subject_id: null,
      seen_at: new Date(Date.now() - 86_400_000).toISOString(),
      created_at: now,
      updated_at: now,
    });
  }

  private get(name: string): Board {
    const b = this.boards.get(name);
    if (!b) throw new HttpError(404, `no such board '${name}'`);
    return b;
  }

  list(archived = false): Board[] {
    const items: Board = {
      name: "items",
      title: "Items",
      kind: "items",
      spec: {},
      pins: [],
      hidden: [],
      notes: [],
      folder_id: null,
      tags: [],
      pinned: false,
      archived: false,
      chat_id: null,
      subject_id: null,
      seen_at: null,
      created_at: null,
      updated_at: null,
    };
    const rest = [...this.boards.values()]
      .filter((b) => b.archived === archived)
      .sort((a, b) => (b.updated_at ?? "").localeCompare(a.updated_at ?? ""));
    return archived ? rest : [items, ...rest];
  }

  create(body: BoardCreate): Board {
    const title = (body.title || body.question || "").trim();
    if (!title) throw new HttpError(400, "Name the board.");
    const now = iso();
    const b: Board = {
      name: `b-${short()}`,
      title,
      kind: body.kind ?? "manual",
      spec: body.spec ?? {},
      pins: [...new Set(body.nodes ?? [])],
      hidden: [],
      notes: [],
      folder_id: body.folder_id ?? null,
      tags: body.tags ?? [],
      pinned: false,
      archived: false,
      chat_id: body.chat_id ?? null,
      subject_id: body.kind === "question" ? `mockq-${short()}` : null,
      seen_at: now,
      created_at: now,
      updated_at: now,
    };
    this.boards.set(b.name, b);
    return { ...b };
  }

  read(name: string): Board {
    return { ...this.get(name) };
  }

  patch(name: string, p: BoardPatch): Board {
    const b = this.get(name);
    if (p.title?.trim()) b.title = p.title.trim();
    if (p.unfile) b.folder_id = null;
    else if (p.folder_id) b.folder_id = p.folder_id;
    if (p.tags) b.tags = p.tags;
    if (p.pinned !== undefined) b.pinned = p.pinned;
    if (p.archived !== undefined) b.archived = p.archived;
    if (p.spec && b.kind === "lens") b.spec = p.spec;
    b.updated_at = iso();
    return { ...b };
  }

  remove(name: string): { deleted: string } {
    this.get(name);
    this.boards.delete(name);
    for (const [id, p] of this.proposals) if (p.board === name) this.proposals.delete(id);
    return { deleted: name };
  }

  pin(name: string, ids: string[]): Board {
    const b = this.get(name);
    b.pins = [...new Set([...b.pins, ...ids])];
    b.hidden = b.hidden.filter((h) => !ids.includes(h));
    b.updated_at = iso();
    return { ...b };
  }

  unpin(name: string, ids: string[]): Board {
    const b = this.get(name);
    for (const id of ids.filter((i) => i.startsWith("prop:"))) {
      const p = this.proposals.get(id.slice(5));
      if (p) p.status = "rejected";
    }
    b.notes = b.notes.filter((n) => !ids.includes(`note:${n.id}`));
    b.pins = b.pins.filter((p) => !ids.includes(p));
    b.hidden = [...new Set([...b.hidden, ...ids.filter((i) => !i.startsWith("note:") && !i.startsWith("prop:"))])];
    return { ...b };
  }

  seen(name: string): Board {
    const b = this.get(name);
    b.seen_at = iso();
    return { ...b };
  }

  setChat(name: string, chatId: string): void {
    this.get(name).chat_id = chatId;
  }

  chatId(name: string): string | null {
    return this.get(name).chat_id;
  }

  propose(name: string, kind: BoardProposal["kind"], payload: Record<string, unknown>, origin = "chat"): BoardProposal {
    const p: BoardProposal = {
      id: short(),
      board: name,
      chat_id: this.get(name).chat_id,
      run_id: null,
      origin,
      kind,
      payload,
      status: "proposed",
      created_at: iso(),
      decided_at: null,
    };
    this.proposals.set(p.id, p);
    return p;
  }

  decide(name: string, id: string, verdict: "accept" | "reject"): { receipt: string; proposal: BoardProposal } {
    const p = this.proposals.get(id);
    if (!p || p.board !== name) throw new HttpError(404, "no such proposal");
    if (p.status !== "proposed") throw new HttpError(409, `Already ${p.status}.`);
    p.status = verdict === "accept" ? "accepted" : "rejected";
    p.decided_at = iso();
    const b = this.get(name);
    if (verdict === "accept") {
      if (p.kind === "card")
        b.notes.push({
          id: p.id,
          title: String(p.payload.title ?? "Note"),
          body: String(p.payload.body ?? ""),
          sources: (p.payload.sources as string[]) ?? [],
          created_at: iso(),
        });
      if (p.kind === "pin") b.pins.push(String(p.payload.node));
    }
    return { receipt: verdict === "accept" ? "Accepted." : "Burned. Pawn will learn from it.", proposal: { ...p } };
  }

  /** Graph of one user board: pins, their strings, note cards, ghosts. */
  graph(name: string): BrainGraph {
    const b = this.get(name);
    const base = this.base();
    const byId = new Map(base.nodes.map((n) => [n.id, n]));
    const keep = new Set(b.pins.filter((p) => byId.has(p)));
    // One hop: what the pinned cards are tied to.
    for (const e of base.edges) {
      if (keep.has(e.from) && byId.has(e.to)) keep.add(e.to);
      else if (keep.has(e.to) && byId.has(e.from)) keep.add(e.from);
    }
    for (const h of b.hidden) keep.delete(h);
    const seen = b.seen_at ?? "";
    const nodes: BrainNode[] = [...keep].map((id) => byId.get(id)!);
    for (const n of b.notes) {
      nodes.push({
        id: `note:${n.id}`,
        kind: "note",
        title: n.title,
        body: n.body,
        status: "active",
        proposed: false,
        meta: { note_id: n.id, sources: n.sources, new: (n.created_at ?? "") > seen },
        actions: ["remove"],
      });
    }
    const edges = base.edges.filter((e) => keep.has(e.from) && keep.has(e.to));
    for (const n of b.notes)
      for (const s of n.sources)
        if (keep.has(s))
          edges.push({ id: `notesrc:${n.id}:${s}`, from: `note:${n.id}`, to: s, role: "mentions", by: "user", cuttable: true });
    const open = [...this.proposals.values()].filter((p) => p.board === name && p.status === "proposed");
    const suggestions = base.suggestions.filter((s) => keep.has(s.from) && keep.has(s.to));
    for (const p of open) {
      if (p.kind === "card" || p.kind === "group") {
        nodes.push({
          id: `prop:${p.id}`,
          kind: p.kind === "card" ? "note" : "thread",
          title: String(p.payload.title ?? "Proposal"),
          body: String(p.payload.body ?? "") || null,
          status: "proposed",
          origin: p.origin,
          proposed: true,
          meta: { proposal_id: p.id, reason: p.payload.reason },
          actions: ["accept", "dismiss"],
        });
        const members = ((p.payload.sources ?? p.payload.nodes) as string[] | undefined) ?? [];
        for (const m of members)
          if (keep.has(m))
            edges.push({ id: `propsrc:${p.id}:${m}`, from: `prop:${p.id}`, to: m, role: "mentions", by: "pawn", cuttable: false });
      } else if (p.kind === "pin") {
        const n = byId.get(String(p.payload.node));
        if (n && !keep.has(n.id))
          nodes.push({ ...n, proposed: true, meta: { ...n.meta, proposal_id: p.id }, actions: ["accept", "dismiss"] });
      } else if (p.kind === "string") {
        const from = String(p.payload.from ?? "");
        const to = String(p.payload.to ?? "");
        if (keep.has(from) && keep.has(to))
          suggestions.push({
            id: `prop:${p.id}`,
            from,
            to,
            role: String(p.payload.role ?? "mentions"),
            reason: String(p.payload.reason ?? "Proposed in the board chat"),
            score: null,
            by: p.origin,
          });
      }
    }
    return {
      board: name,
      meta: { name, title: b.title, kind: b.kind, chat_id: b.chat_id, subject_id: b.subject_id, seen_at: b.seen_at, spec: b.spec },
      nodes,
      edges,
      suggestions,
      proposals: open,
      emergence_enabled: base.emergence_enabled,
      truncated: false,
    };
  }
}

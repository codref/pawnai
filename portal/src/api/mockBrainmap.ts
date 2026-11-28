// Mock emergence web for the brainmap board (VITE_PAWN_MOCK=1): subjects,
// observation links, ghost suggestions, spillable evidence and a layout kept
// in localStorage so "drag, reload, still there" works without a server.
import {
  HttpError,
  type BrainEdge,
  type BrainGraph,
  type BrainLassoBody,
  type BrainLassoResult,
  type BrainLayout,
  type BrainNode,
  type BrainTieResult,
  type EmergenceObservation,
  type EmergenceStatus,
  type EmergenceSubject,
  type EmergenceSubjectDetail,
  type InboxItem,
} from "./types";

const OPEN_ITEM = ["new", "notified", "snoozed"];
const LAYOUT_KEY = "pawn.mock.brainmap.layout";

interface MockSubject extends EmergenceSubject {
  assumptions?: { text: string; health: string }[];
  direction?: string;
  due?: string;
  by_when?: string;
}

interface MockObs {
  id: string;
  text: string;
  quote: string;
  speaker: string;
  source_ref: string;
  start_sec: number;
  type: string;
}

interface MockLink {
  id: string;
  obs: string;
  subject: string;
  role: string;
  by: "user" | "pawn" | "goal";
  score?: number;
}

interface MockSuggestion {
  obs: string;
  subject: string;
  role: string;
  reason: string;
  score: number;
}

const iso = (daysAgo: number) => new Date(Date.now() - daysAgo * 86_400_000).toISOString();

function subject(
  id: string,
  kind: string,
  title: string,
  extra: Partial<MockSubject> = {},
): MockSubject {
  return {
    id,
    kind,
    title,
    slug: title.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    origin: "extracted",
    status: "active",
    state_md: null,
    fields: {},
    momentum: null,
    parent_id: null,
    note_key: null,
    created_at: iso(12),
    updated_at: iso(1),
    ...extra,
  };
}

/** Observation ids for the mock Inbox items (`it{n}` → `o{n}`). */
export function mockObservationId(itemIndex: number): string {
  return `o${itemIndex}`;
}

export class MockBrainmap {
  private subjects = new Map<string, MockSubject>();
  private links: MockLink[] = [];
  private suggestions: MockSuggestion[] = [];
  private rejected = new Set<string>();
  private scraps: MockObs[] = [];
  private seq = 100;

  constructor(private readonly items: () => InboxItem[]) {
    const s = (x: MockSubject) => this.subjects.set(x.id, x);
    s(
      subject("t1", "thread", "Acme launch", {
        origin: "goal",
        momentum: 0.6,
        state_md:
          "Launch is **on for 1 December**. Budget sign-off is the open risk; legal review is slipping and two budget figures are in circulation.",
        note_key: "Pawn/Threads/acme-launch.md",
      }),
    );
    s(
      subject("t2", "thread", "Platform migration confidence eroding", {
        origin: "emergent",
        momentum: -0.35,
        state_md: "Marco now says *Q2 at the earliest*. The API freeze assumes team B is ready by Monday, which nobody confirmed.",
        note_key: "Pawn/Threads/platform-migration-confidence-eroding.md",
      }),
    );
    s(
      subject("t3", "thread", "Vendor contract doubts", {
        origin: "emergent",
        status: "proposed",
        momentum: 0.4,
        state_md: "Six observations across three meetings question the vendor contract: unsigned, SLA unclear, review not booked.",
      }),
    );
    s(subject("t4", "thread", "On-call load", { origin: "emergent", momentum: 0.2, state_md: "Luca raised on-call load in 3 of the last 4 1:1s." }));
    s(subject("q1", "question", "Is the 1 December date realistic?", { parent_id: "t1", state_md: "2 for / 3 against." }));
    s(
      subject("d1", "decision", "Freeze the public API on Monday", {
        parent_id: "t2",
        assumptions: [
          { text: "Team B is ready by Monday", health: "weak" },
          { text: "Docs are complete", health: "ok" },
          { text: "No partner needs a breaking change", health: "broken" },
        ],
        review_at: iso(-7),
      } as Partial<MockSubject>),
    );
    s(subject("c1", "commitment", "Davide → Ana: Q4 launch plan draft", { parent_id: "t1", direction: "mine", due: "2026-12-05" }));
    s(subject("c2", "commitment", "Ana → finance: revised budget", { parent_id: "t1", direction: "theirs", due: "2026-12-03" }));
    s(subject("p1", "prediction", "API freeze holds through December", { parent_id: "t2", by_when: "2026-12-20" }));
    s(
      subject("h1", "hypothesis", "On-call worries and the vendor delay share a root cause: staging capacity", {
        origin: "dream",
        status: "proposed",
        state_md:
          "Load testing is blocked on staging and the vendor contract covers the staging upgrade. Test: ask whether the upgrade waits on the contract.",
      }),
    );

    const L = (obs: number | string, subj: string, role: string, by: MockLink["by"] = "pawn", score = 0.22) =>
      this.links.push({
        id: `link:${this.seq++}`,
        obs: typeof obs === "number" ? mockObservationId(obs) : obs,
        subject: subj,
        role,
        by,
        score,
      });
    L(1, "c1", "origin");
    L(1, "t1", "mentions");
    L(2, "t1", "evidence_for", "goal");
    L(2, "q1", "evidence_for");
    L(3, "t1", "mentions");
    L(5, "q1", "evidence_against");
    L(7, "c2", "origin");
    L(7, "t1", "mentions");
    L(8, "t4", "evidence_for");
    L(8, "h1", "evidence_for");
    L(9, "q1", "mentions", "user");
    L(10, "t2", "contradicts");
    L(10, "q1", "evidence_against");
    L(11, "d1", "origin");
    L(12, "t2", "mentions");
    L(13, "t4", "mentions");
    L(14, "t3", "evidence_for");
    L(15, "t3", "evidence_for");
    L(15, "h1", "evidence_for");
    L(18, "t4", "mentions");
    L(20, "t1", "contradicts");

    this.suggestions = [
      { obs: "o4", subject: "t3", role: "mentions", reason: "Same vendor named; d=0.21", score: 0.21 },
      { obs: "o16", subject: "t1", role: "mentions", reason: "Launch waitlist — close to Acme launch; d=0.27", score: 0.27 },
      { obs: "o17", subject: "t3", role: "evidence_for", reason: "SLA terms ↔ vendor contract; d=0.18", score: 0.18 },
      { obs: "o19", subject: "q1", role: "evidence_against", reason: "Forecast timing vs the 1 Dec date; d=0.33", score: 0.33 },
      { obs: "o13", subject: "t1", role: "mentions", reason: "Launch week mentioned; d=0.39", score: 0.39 },
      { obs: "o12", subject: "d1", role: "evidence_for", reason: "Runbook needed before the freeze; d=0.31", score: 0.31 },
    ];

    this.scraps = [
      { id: "s1", text: "Team B lead is out until Wednesday", quote: "I'm off Monday and Tuesday.", speaker: "Kai", source_ref: "platform-1on1", start_sec: 812, type: "fact" },
      { id: "s2", text: "Partner X still calls the v1 endpoints", quote: "They hit v1 every night.", speaker: "Marco", source_ref: "platform-1on1", start_sec: 1290, type: "fact" },
      { id: "s3", text: "Migration dry run failed twice", quote: "The dry run fell over again.", speaker: "Mira", source_ref: "weekly-sync", start_sec: 2210, type: "concern" },
      { id: "s4", text: "Pager fired 14 times last week", quote: "Fourteen pages, mostly at night.", speaker: "Luca", source_ref: "platform-1on1", start_sec: 340, type: "fact" },
      { id: "s5", text: "Legal wants the DPA before signing", quote: "No DPA, no signature.", speaker: "Ana", source_ref: "acme-kickoff", start_sec: 1555, type: "block" },
    ];
    L("s1", "t2", "evidence_against");
    L("s2", "t2", "evidence_against");
    L("s3", "t2", "mentions");
    L("s4", "t4", "evidence_for");
    L("s5", "t3", "evidence_for");
  }

  private openItems(): InboxItem[] {
    return this.items().filter((i) => OPEN_ITEM.includes(i.status));
  }

  private itemByObs(): Map<string, InboxItem> {
    const out = new Map<string, InboxItem>();
    for (const i of this.openItems()) {
      const oid = i.payload?.observation_id;
      if (typeof oid === "string") out.set(oid, i);
    }
    return out;
  }

  private nodeIdForObs(obs: string, byObs: Map<string, InboxItem>): string | null {
    const item = byObs.get(obs);
    return item ? `item:${item.id}` : null;
  }

  private subjectNode(s: MockSubject): BrainNode {
    const actions: string[] = [];
    if (s.status === "proposed") actions.push("accept", "dismiss");
    else {
      if (s.kind === "thread") actions.push("spill", "done", "dismiss");
      if (s.kind === "decision") actions.push("council", "done");
      if (s.kind === "question") actions.push("resolved", "dismiss");
      if (s.kind === "commitment" || s.kind === "prediction") actions.push("done", "dismiss");
      if (s.kind === "hypothesis") actions.push("dismiss");
    }
    return {
      id: `subject:${s.id}`,
      kind: s.kind as BrainNode["kind"],
      title: s.title,
      body: s.state_md,
      status: s.status,
      origin: s.origin,
      proposed: s.status === "proposed",
      meta: {
        subject_id: s.id,
        momentum: s.momentum,
        evidence_count: this.links.filter((l) => l.subject === s.id).length,
        assumptions: s.assumptions,
        direction: s.direction,
        due: s.due,
        by_when: s.by_when,
        note_key: s.note_key,
      },
      actions,
    };
  }

  private obsDetail(o: MockObs): Omit<EmergenceObservation, "role" | "linked_at"> {
    return {
      id: o.id,
      type: o.type,
      text: o.text,
      context: "",
      quote: o.quote,
      speaker_name: o.speaker,
      start_sec: o.start_sec,
      fields: {},
      status: "live",
      created_at: iso(3),
      source_ref: o.source_ref,
      source_kind: "session",
    };
  }

  private itemObs(item: InboxItem): MockObs {
    return {
      id: String(item.payload.observation_id),
      text: item.text,
      quote: item.quote || "",
      speaker: item.owner || "",
      source_ref: item.source_ref,
      start_sec: Number(item.payload.start_sec ?? 0),
      type: item.kind,
    };
  }

  graph(): BrainGraph {
    const nodes: BrainNode[] = [];
    const edges: BrainEdge[] = [];
    const byObs = this.itemByObs();
    const people = new Map<string, string>();
    const sessions = new Set<string>();
    for (const item of this.openItems()) {
      const oid = typeof item.payload?.observation_id === "string" ? item.payload.observation_id : undefined;
      nodes.push({
        id: `item:${item.id}`,
        kind: "item",
        title: item.text,
        quote: item.quote,
        status: item.status,
        proposed: false,
        meta: {
          item_id: item.id,
          item_kind: item.kind,
          observation_id: oid,
          source_ref: item.source_ref,
          start_sec: typeof item.payload?.start_sec === "number" ? item.payload.start_sec : null,
          owner: item.owner,
          due: item.due,
          note_key: item.note_key,
        },
        actions: ["todo", "file", "task", "later", "ignore", "delete"],
      });
      if (item.owner) {
        const pid = `person:${item.owner.toLowerCase()}`;
        people.set(pid, item.owner);
        edges.push({ id: `derived:item:${item.id}->${pid}`, from: `item:${item.id}`, to: pid, role: "owner", by: "derived", cuttable: false });
      }
      if (item.source_ref) {
        sessions.add(item.source_ref);
        const sid = `session:${item.source_ref}`;
        edges.push({ id: `derived:item:${item.id}->${sid}`, from: `item:${item.id}`, to: sid, role: "source", by: "derived", cuttable: false });
      }
    }
    for (const [id, name] of people) {
      nodes.push({ id, kind: "person", title: name, status: "active", proposed: false, meta: {}, actions: [] });
    }
    for (const ref of sessions) {
      const title = ref.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
      nodes.push({ id: `session:${ref}`, kind: "session", title, status: "active", proposed: false, meta: { source_ref: ref }, actions: [] });
    }
    for (const s of this.subjects.values()) {
      if (s.status === "dismissed") continue;
      nodes.push(this.subjectNode(s));
      if (s.parent_id && this.subjects.get(s.parent_id)?.status !== "dismissed") {
        edges.push({ id: `parent:${s.id}`, from: `subject:${s.id}`, to: `subject:${s.parent_id}`, role: "parent", by: "pawn", cuttable: true });
      }
    }
    const shown = new Set(nodes.map((n) => n.id));
    for (const l of this.links) {
      const from = this.nodeIdForObs(l.obs, byObs);
      const to = `subject:${l.subject}`;
      if (!from || !shown.has(to)) continue;
      edges.push({ id: l.id, from, to, role: l.role, by: l.by, cuttable: true, score: l.score });
    }
    const suggestions = this.suggestions
      .map((s) => ({ s, from: this.nodeIdForObs(s.obs, byObs), to: `subject:${s.subject}` }))
      .filter((x) => x.from && shown.has(x.to))
      .map(({ s, from, to }) => ({ id: `sug:${s.obs}:${s.subject}`, from: from!, to, role: s.role, reason: s.reason, score: s.score }));
    return { board: "items", nodes, edges, suggestions, emergence_enabled: true, truncated: false };
  }

  private obsOf(nodeId: string): string | null {
    if (nodeId.startsWith("obs:")) return nodeId.slice(4);
    if (nodeId.startsWith("item:")) {
      const item = this.items().find((i) => i.id === nodeId.slice(5));
      const oid = item?.payload?.observation_id;
      return typeof oid === "string" ? oid : null;
    }
    return null;
  }

  private subjectOf(nodeId: string): MockSubject | null {
    return nodeId.startsWith("subject:") ? this.subjects.get(nodeId.slice(8)) ?? null : null;
  }

  private edgeFor(l: MockLink): BrainEdge {
    const byObs = this.itemByObs();
    const from = this.nodeIdForObs(l.obs, byObs) ?? `obs:${l.obs}`;
    return { id: l.id, from, to: `subject:${l.subject}`, role: l.role, by: l.by, cuttable: true, score: l.score ?? null };
  }

  tie(from: string, to: string, role: string): BrainTieResult {
    const a = this.subjectOf(from);
    const b = this.subjectOf(to);
    if (a && b) {
      a.parent_id = b.id;
      return {
        edge: { id: `parent:${a.id}`, from, to, role: "parent", by: "user", cuttable: true },
        receipt: `Filed “${a.title}” under “${b.title}”.`,
      };
    }
    const obs = this.obsOf(from) ?? this.obsOf(to);
    const subj = b ?? a;
    if (!obs || !subj) throw new HttpError(400, "Tie evidence to a subject.");
    if (this.links.some((l) => l.obs === obs && l.subject === subj.id)) throw new HttpError(409, "Already tied.");
    const link: MockLink = { id: `link:${this.seq++}`, obs, subject: subj.id, role, by: "user" };
    this.links.push(link);
    this.rejected.delete(`${subj.id}:${obs}`);
    this.suggestions = this.suggestions.filter((s) => !(s.obs === obs && s.subject === subj.id));
    return { edge: this.edgeFor(link), receipt: `Tied to “${subj.title}”. Pawn will learn from it.` };
  }

  cut(edgeId: string): { receipt: string } {
    if (edgeId.startsWith("parent:")) {
      const s = this.subjects.get(edgeId.slice(7));
      if (!s) throw new HttpError(404, "No such string.");
      s.parent_id = null;
      return { receipt: `Unfiled “${s.title}”.` };
    }
    const link = this.links.find((l) => l.id === edgeId);
    if (!link) throw new HttpError(404, "No such string.");
    this.links = this.links.filter((l) => l !== link);
    this.rejected.add(`${link.subject}:${link.obs}`);
    return { receipt: "Cut. Pawn will not tie these two again." };
  }

  suggestion(id: string, verdict: "confirm" | "snip"): { edge?: BrainEdge | null; receipt: string } {
    const [, obs, subj] = id.split(":");
    const s = this.suggestions.find((x) => x.obs === obs && x.subject === subj);
    if (!s) throw new HttpError(404, "No such suggestion.");
    this.suggestions = this.suggestions.filter((x) => x !== s);
    if (verdict === "snip") {
      this.rejected.add(`${subj}:${obs}`);
      return { edge: null, receipt: "Snipped. Pawn will not suggest this again." };
    }
    const link: MockLink = { id: `link:${this.seq++}`, obs, subject: subj, role: s.role, by: "user" };
    this.links.push(link);
    return { edge: this.edgeFor(link), receipt: "Tied. Pawn will learn from it." };
  }

  lasso(body: BrainLassoBody): BrainLassoResult {
    let target: MockSubject;
    if (body.into) {
      const found = this.subjects.get(body.into);
      if (!found) throw new HttpError(404, "No such thread.");
      target = found;
    } else {
      const title = (body.title || "").trim();
      if (!title) throw new HttpError(400, "Name the thread.");
      target = subject(`u${this.seq++}`, "thread", title, { origin: "user", momentum: 0, state_md: "A thread you started from the board." });
      this.subjects.set(target.id, target);
    }
    const edges: BrainEdge[] = [];
    for (const nid of body.node_ids) {
      const sub = this.subjectOf(nid);
      if (sub && sub.id !== target.id) {
        sub.parent_id = target.id;
        edges.push({ id: `parent:${sub.id}`, from: nid, to: `subject:${target.id}`, role: "parent", by: "user", cuttable: true });
        continue;
      }
      const obs = this.obsOf(nid);
      if (!obs || this.links.some((l) => l.obs === obs && l.subject === target.id)) continue;
      const link: MockLink = { id: `link:${this.seq++}`, obs, subject: target.id, role: "mentions", by: "user" };
      this.links.push(link);
      edges.push({ ...this.edgeFor(link), from: nid });
    }
    return { subject: this.subjectNode(target), edges };
  }

  spill(subjectId: string): { nodes: BrainNode[]; edges: BrainEdge[] } {
    const byObs = this.itemByObs();
    const nodes: BrainNode[] = [];
    const edges: BrainEdge[] = [];
    for (const l of this.links.filter((x) => x.subject === subjectId && !byObs.has(x.obs))) {
      const o = this.scraps.find((s) => s.id === l.obs);
      if (!o) continue;
      nodes.push({
        id: `obs:${o.id}`,
        kind: "scrap",
        title: o.text,
        quote: o.quote,
        status: "live",
        proposed: false,
        meta: { observation_id: o.id, source_ref: o.source_ref, start_sec: o.start_sec, speaker: o.speaker },
        actions: [],
      });
      edges.push({ id: l.id, from: `obs:${o.id}`, to: `subject:${subjectId}`, role: l.role, by: l.by, cuttable: true });
    }
    return { nodes, edges };
  }

  detail(id: string): EmergenceSubjectDetail {
    const s = this.subjects.get(id);
    if (!s) throw new HttpError(404, "no such subject");
    const byObs = this.itemByObs();
    const observations: EmergenceObservation[] = [];
    for (const l of this.links.filter((x) => x.subject === id)) {
      const item = byObs.get(l.obs) ?? this.items().find((i) => i.payload?.observation_id === l.obs);
      const o = item ? this.itemObs(item) : this.scraps.find((x) => x.id === l.obs);
      if (!o) continue;
      observations.push({ ...this.obsDetail(o), role: l.role, linked_at: iso(2) });
    }
    return {
      subject: s,
      observations,
      events: [
        { subject_id: id, kind: "created", at: s.created_at || iso(12), diff: {} },
        { subject_id: id, kind: "state_rewritten", at: iso(1), diff: {} },
      ],
    };
  }

  status(id: string, status: EmergenceStatus): { subject: EmergenceSubject } {
    const s = this.subjects.get(id);
    if (!s) throw new HttpError(404, "no such subject");
    s.status = status;
    s.updated_at = new Date().toISOString();
    return { subject: s };
  }

  getLayout(): BrainLayout {
    try {
      const raw = window.localStorage.getItem(LAYOUT_KEY);
      if (raw) return JSON.parse(raw) as BrainLayout;
    } catch {
      /* storage unavailable */
    }
    return { positions: {}, viewport: null };
  }

  putLayout(layout: BrainLayout): BrainLayout {
    try {
      window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
    } catch {
      /* storage unavailable */
    }
    return layout;
  }
}

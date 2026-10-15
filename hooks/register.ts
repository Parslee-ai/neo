import type {
  EngineInterface,
  HookStream,
  On,
  ProcessSpawnChunk,
  ProcessSpawnResult,
  Register,
  RenderElement,
  RenderInput,
  RenderNode,
  Timer,
} from "claude-code";
import {
  NeoStream,
  followupContext,
  object,
  openQuestions,
  resultText,
  rows,
  strings,
  text,
  validationText,
} from "./protocol.js";
import type { Document } from "./protocol.js";
import {
  AUTO_DEADLINE,
  REVIEW_CONTINUE,
  activityLabel,
  elapsed,
  findingMarkdown,
  voiceLine,
  EDIT_TOOLS,
  MAX_DIAGNOSES,
  RECALL_TIMEOUT,
  rememberBeat,
  architectPrompt,
  autoPayload,
  debugPrompt,
  fingerprint,
  hasFindings,
  hasVerifiableSuggestion,
  injection,
  isCheckCommand,
  recallInjection,
  reviewPrompt,
} from "./auto.js";
import type { AutoKind } from "./auto.js";

const PANE = "neo-workspace";
const DEADLINE = 240000;
const MODES = ["reason", "review", "debug", "architect", "optimize", "pattern"];
const TASKS: Record<string, string> = {
  reason: "explanation",
  review: "refactor",
  debug: "bugfix",
  architect: "explanation",
  optimize: "algorithm",
  pattern: "explanation",
};
const INSTRUCTIONS: Record<string, string> = {
  reason: "Reason about",
  review: "Review for correctness, security, edge cases and maintainability:",
  debug: "Diagnose the root cause and propose a fix for:",
  architect: "Evaluate architecture and concrete trade-offs for:",
  optimize: "Find defensible performance improvements for:",
  pattern: "Identify existing or reusable patterns for:",
};
const DESCRIPTIONS: Record<string, string> = {
  reason: "General reasoning about a question or code",
  review: "Correctness, security, edge cases and maintainability",
  debug: "Root cause and a proposed fix",
  architect: "Architecture and concrete trade-offs",
  optimize: "Defensible performance improvements",
  pattern: "Existing or reusable patterns",
};
type Run = {
  id: string;
  session: string;
  cwd: string;
  binary: string;
  status: string;
  prompt: string;
  mode: string;
  /** The selected context this run was sent with. */
  context: string;
  parser: NeoStream;
  result?: Document;
  error?: string;
  stream?: HookStream<ProcessSpawnChunk, ProcessSpawnResult>;
  timer?: Timer;
  ticker?: Timer;
  /** The session generation the run started in; a later one owns the pane. */
  generation: number;
  /** Asked by a command typed in the conversation, not from the side pane. */
  fromCommand?: boolean;
  started: number;
  elapsed?: number;
  stopped?: string;
};
let current: Run | undefined, previous: Run | undefined;
/** Asked runs by the short id their command row carries, so the row can draw
 * the run live in the conversation. */
const askedRuns = new Map<string, Run>();
let context = "",
  question = "",
  mode = "reason",
  tab = "results",
  details = false;
let memoryQuery = "",
  memoryPage = 0,
  notice = "";
const FACTS_PER_PAGE = 30;
let snapshot: Document = {},
  explanation: Document | undefined,
  inspectorError = "";
let session = "",
  cwd = "",
  interactive = false,
  drawable = false,
  binary = "neo",
  recording = true,
  lastInvocation = "";
const turns = new Map<string, string>();
let generation = 0;
/** Set while start() awaits, so a second press cannot start a second run. */
let starting = false;
/** The clock as of the last tick, so a drawing can show elapsed time. */
let nowMs = 0;
const clockText = (ms: number) => {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};
const JSON_CAP = 20000;
/** Pretty JSON for a Code view, capped: one drawing shares a fixed character
 * budget across every Code block it holds, diffs included. */
function cappedJson(value: unknown): string {
  const out = JSON.stringify(value, null, 2) ?? "";
  return out.length > JSON_CAP
    ? out.slice(0, JSON_CAP) + `\n… truncated (${out.length} characters)`
    : out;
}
function progressLine(run: Run): string {
  const active = run.status === "running" || run.status === "stopping";
  const elapsed = Math.min(
    DEADLINE,
    active ? nowMs - run.started : (run.elapsed ?? nowMs - run.started),
  );
  return active
    ? `${run.status} · ${clockText(elapsed)} of ${clockText(DEADLINE)} · ${run.parser.message}`
    : `${run.status} · took ${clockText(elapsed)}`;
}

function redraw($: EngineInterface) {
  $.ui.invalidate("ui.render");
}
async function record($: EngineInterface, action: string, payload: Document) {
  if (!recording) return;
  try {
    await $.process.run([binary, "hook", action], {
      stdin: JSON.stringify(payload),
      timeoutMs: 5000,
    });
  } catch {
    /* Observability must not replace host behavior. */
  }
}
async function inspect($: EngineInterface, identifier?: string) {
  const startedGeneration = generation;
  const root = await $.session.cwd();
  cwd = root;
  inspectorError = "";
  try {
    const r = await $.process.run(
      [
        binary,
        "workspace",
        identifier ? "explain" : "status",
        ...(identifier ? [identifier] : []),
        "--cwd",
        root,
      ],
      { timeoutMs: DEADLINE },
    );
    if (!r.stdout.trim())
      throw new Error(
        "Neo CLI lacks the workspace protocol or could not start. Install the matching neo-reasoner release or set NEO_MOD_EXECUTABLE to its executable.",
      );
    const data = object(JSON.parse(r.stdout));
    if (generation !== startedGeneration) return;
    if (r.exitCode || data.error || r.isStdoutTruncated)
      throw new Error(
        text(data.error) ||
          "Workspace protocol unavailable. Install the matching neo-reasoner release.",
      );
    if (data.schema_version !== 1)
      throw new Error(
        "Unsupported workspace protocol. Update plugin and CLI together.",
      );
    if (identifier) explanation = data;
    else snapshot = data;
  } catch (error) {
    if (generation !== startedGeneration) return;
    inspectorError = String(error);
  }
  redraw($);
}
async function open($: EngineInterface) {
  if (drawable)
    await $.ui.open({
      id: PANE,
      title: "Neo",
      focus: true,
      rows: 22,
      columns: 66,
    });
}
async function stop($: EngineInterface, reason = "cancelled") {
  const run = current;
  if (!run || run.status !== "running") return;
  run.stopped = reason;
  run.status = "stopping";
  run.timer?.cancel();
  // If closing the stream never ends the loop, the ticker must not redraw
  // forever: give the close ten seconds.
  $.clock.after(10000, () => run.ticker?.cancel());
  redraw($);
  // Close only this invocation's stream; never signal a process group.
  try {
    await run.stream?.return({ code: null, signal: "SIGTERM" });
  } catch {
    /* consumer owns final disposition */
  }
}
async function consume($: EngineInterface, run: Run, payload: Document) {
  try {
    run.stream = $.process.spawn({
      argv: [run.binary, "--stdin-json", "--json", "--mode", "advise"],
      cwd: run.cwd,
      input: JSON.stringify(payload),
    });
    void run.stream.result.catch(() => {});
    run.timer = $.clock.after(DEADLINE, () => {
      void stop($, "timed out after 4 minutes").catch(() => {});
    });
    run.ticker = $.clock.every(1000, () => {
      void $.clock.now().then(
        (t) => {
          nowMs = t;
          redraw($);
        },
        () => {},
      );
    });
    for await (const chunk of run.stream) {
      run.parser.push(chunk.stream, chunk.text);
      redraw($);
    }
    const ended = await run.stream.result;
    run.elapsed = (await $.clock.now()) - run.started;
    if (!run.stopped) {
      run.result = run.parser.finish(ended.code, ended.signal);
      const gates = object(run.result.validation_assessment);
      run.status =
        gates.failed > 0
          ? "checks failed"
          : gates.pending > 0 || gates.unavailable > 0
            ? "validation needed"
            : "findings ready";
    }
  } catch (error) {
    if (!run.stopped) {
      run.error = String(error);
      run.status = "failed";
    }
  } finally {
    run.timer?.cancel();
    run.ticker?.cancel();
    run.elapsed ??= (await $.clock.now()) - run.started;
    if (run.stopped) {
      run.status = "cancelled";
      run.error = run.stopped;
    }
    const ownsPane = run.generation === generation;
    // A run that produced nothing gives its context back for the retry, unless
    // the person has attached something new meanwhile, or the session it
    // belonged to has ended.
    if (
      ownsPane &&
      (run.status === "failed" || run.status === "cancelled") &&
      !context
    )
      context = run.context;
    // A finished analysis may have written memory: the next Memory visit reads
    // the store again.
    if (ownsPane && run.result) snapshot = {};
    // Asked in the conversation and run in the background, the answer reached
    // the person in its card; Claude gets it at the next hook that can carry
    // context, never as a row appended mid-turn (which could land between a
    // tool call and its result). A side-pane ask stays the person's to share.
    if (ownsPane && run.result && run.fromCommand && interactive && drawable)
      pending.push(injection("answer", run.result));
    await record($, "event", {
      event: "neo_finished",
      session_id: run.session,
      invocation_id: run.id,
      cwd: run.cwd,
      status: run.status,
    });
    redraw($);
    if (run.error && ownsPane) $.ui.log("Neo: " + run.error);
  }
}
async function start(
  $: EngineInterface,
  prompt: string,
  selectedMode = mode,
  followup = false,
  fromCommand = false,
) {
  if (
    starting ||
    current?.status === "running" ||
    current?.status === "stopping"
  )
    return "Neo is already working. Cancel that run before starting another.";
  if (!prompt.trim()) return "Enter a question or target first.";
  if (prompt.length > 24000 || context.length > 16000)
    return "Question or selected context exceeds the 24k / 16k character limits.";
  starting = true;
  try {
    session = await $.session.id();
    cwd = await $.session.cwd();
  } finally {
    starting = false;
  }
  const prior = current?.result ? current : previous;
  if (followup && !prior?.result)
    return "Run an analysis before asking a follow-up.";
  if (current?.result) previous = current;
  // A follow-up is about the same code: with no new selection it carries the
  // prior run's context. A fresh question carries only what is attached now.
  const sentContext = context || (followup ? prior!.context : "");
  const id = crypto.randomUUID();
  const run: Run = {
    id,
    session,
    cwd,
    binary,
    status: "running",
    prompt,
    mode: selectedMode,
    context: sentContext,
    generation,
    fromCommand,
    parser: new NeoStream(),
    started: await $.clock.now(),
  };
  nowMs = run.started;
  details = false;
  current = run;
  askedRuns.set(id.slice(0, 8), run);
  if (askedRuns.size > 20) askedRuns.delete(askedRuns.keys().next().value!);
  question = prompt;
  mode = selectedMode;
  lastInvocation = id;
  tab = "results";
  const analysis = followup
    ? followupContext(prior!.result!, prompt)
    : `${INSTRUCTIONS[selectedMode] ?? INSTRUCTIONS.reason} ${prompt}`;
  const payload = {
    prompt:
      analysis +
      (sentContext ? "\n\nUser-selected context:\n" + sentContext : ""),
    working_directory: cwd,
    safe_read_paths: [cwd],
    auto_context: true,
    operating_mode: "advise",
    task_type: TASKS[selectedMode] ?? "explanation",
    role: followup ? "followup_advisor" : "advisor",
    execution_identity: {
      session_id: session,
      task_id: id,
      parent_task_id: followup ? prior!.id : "",
      trace_id: id,
      discovery_source: "claude-code-mod",
      repositories_touched: [cwd],
    },
  };
  // Selected context belongs to this question: keeping it would send it again,
  // unseen, with the next unrelated one.
  context = "";
  notice = "";
  await record($, "event", {
    event: "neo_started",
    session_id: session,
    invocation_id: id,
    cwd,
  });
  redraw($);
  const work = consume($, run, payload);
  if (!interactive || !drawable) await work;
  else void work.catch(() => {}); // The host may unload while cleanup is pending.
  return (!interactive || !drawable) && run.result
    ? resultText(run.result)
    : run.error
      ? "Neo: " + run.error
      : `Asked Neo [${id.slice(0, 8)}].`;
}
/** Start from the pane (Ask, Follow up or Enter). A refusal is shown where
 * the person asked; a started run shows its own progress line. */
async function ask($: EngineInterface, followup: boolean) {
  const before = current;
  const answer = await start($, question, mode, followup);
  notice = current === before ? answer : "";
}
function statusText(): string {
  return current
    ? `Neo ${current.status}: ${current.error ?? current.parser.message}`
    : `Neo ready. ${snapshot.fact_count ?? "Unknown"} project facts. /neo-workspace opens the workspace; /neo-ask <question> starts an analysis.`;
}
function matchingFacts(): Document[] {
  return rows(snapshot.facts).filter((f) =>
    `${f.id} ${f.subject}`.toLowerCase().includes(memoryQuery.toLowerCase()),
  );
}
function lastPage(): number {
  return Math.max(0, Math.ceil(matchingFacts().length / FACTS_PER_PAGE) - 1);
}
function render(
  $: EngineInterface,
  e: RenderInput<"Pane" | "AbovePrompt">,
): RenderElement {
  if (e.surface !== "terminal" && e.surface !== "desktop")
    return $.ui.resolve(e).Text({ children: statusText() });
  const { Box, Text, Button, Input, Select, Code } = $.ui.resolve(e);
  const label = (value: string, color?: string) =>
    Text({ children: value, ...(color ? { color } : {}) });
  const button = (key: string, value: string, hotkey?: string) =>
    Button({
      key,
      label: value,
      onPress: () => {},
      ...(hotkey ? { hotkey } : {}),
    });
  if (e.component === "AbovePrompt")
    return Box({
      flexDirection: "row",
      gap: 1,
      children: [
        label("Neo", "remember"),
        Text({
          children: current ? progressLine(current) : "ready",
          wrap: "truncate-end",
        }),
        ...(current?.status === "running" ? [button("neo-cancel", "Cancel")] : []),
      ],
    });
  const children: RenderNode[] = [
    label(
      "Neo · " + (current?.status ?? "ready"),
      current?.error ? "error" : "remember",
    ),
    Box({
      flexDirection: e.props.bodyColumns < 46 ? "column" : "row",
      gap: 1,
      children: [
        button("neo-results", "Results", "r"),
        button("neo-memory", "Memory", "m"),
        button("neo-evidence", "Evidence", "e"),
        ...(current?.status === "running"
          ? [button("neo-cancel", "Cancel", "c")]
          : []),
      ],
    }),
  ];
  if (tab === "results") {
    const result = current?.result;
    // The question form comes first: after a long answer it would otherwise
    // be scrolled out of reach.
    children.push(
      Select({
        key: "neo-mode",
        label: "Mode",
        value: mode,
        options: MODES.map((value) => ({ value, label: value })),
        onSelect: () => {},
      }),
      label(DESCRIPTIONS[mode] ?? "", "subtle"),
      Input({
        key: "neo-question",
        label: "Question or target",
        value: question,
        onInput: () => {},
        onSubmit: () => {},
      }),
      Input({
        key: "neo-context",
        label: "Selected context (optional)",
        value: context,
        onInput: () => {},
        onSubmit: () => {},
      }),
    );
    if (context)
      children.push(
        label(
          `Context attached · ${context.length} characters · sent with the next question only`,
          "subtle",
        ),
      );
    children.push(
      Box({
        flexDirection: e.props.bodyColumns < 66 ? "column" : "row",
        gap: 1,
        children: [
          button("neo-run", "Ask Neo", "a"),
          // A follow-up needs a finished analysis to follow.
          ...(result || previous?.result
            ? [button("neo-followup", "Follow up", "f")]
            : []),
          button("neo-selection", "Use selection"),
          ...(result ? [button("neo-share", "Share with Claude")] : []),
        ],
      }),
    );
    if (notice) children.push(label(notice, "warning"));
    children.push(
      label(
        "Advisory · proposes changes; does not apply them or claim downstream acceptance.",
        "subtle",
      ),
    );
    if (current) children.push(label(progressLine(current)));
    if (current?.error) children.push(label(current.error, "error"));
    if (result) {
      children.push(label(text(object(result.orchestrator).summary)));
      const beat = text(object(result.orchestrator).personality);
      if (beat) children.push(label(beat, "subtle"));
      for (const caution of strings(object(result.orchestrator).cautions))
        children.push(label("! " + caution, "warning"));
      if (result.notes) children.push(label(text(result.notes)));
      for (const [i, s] of rows(result.code_suggestions).entries()) {
        children.push(label(`${i + 1}. ${s.file_path}: ${s.description}`));
        if (s.unified_diff)
          children.push(
            Code({
              source: text(s.unified_diff),
              language: "diff",
              path: text(s.file_path),
            }),
          );
        for (const t of strings(s.tradeoffs))
          children.push(label("Trade-off: " + t));
      }
      for (const check of rows(result.static_checks))
        children.push(
          label(
            `Check · ${check.tool_name} (${check.status || "status unavailable"}): ${check.summary}`,
          ),
        );
      children.push(label("Validation · " + validationText(result)));
      for (const q of openQuestions(result))
        children.push(label("Open question · " + q, "warning"));
      children.push(
        button(
          "neo-details",
          details ? "Hide reasoning" : "Show reasoning",
          "d",
        ),
      );
      if (details)
        children.push(
          Code({
            source: cappedJson({
              plan: result.plan,
              simulation_traces: result.simulation_traces,
              validation: result.validation_assessment,
              hypotheses: result.hypotheses,
            }),
            language: "json",
          }),
        );
    } else if (!current)
      children.push(
        label(
          "Ask Neo for a second opinion. Findings, cautions and evidence appear here.",
        ),
      );
  } else if (tab === "memory") {
    children.push(
      label(
        `${snapshot.fact_count ?? "?"} project facts · ${snapshot.probation_count ?? "?"} on probation${snapshot.missing_file_count ? ` · ${snapshot.missing_file_count} about files not in this checkout` : ""}`,
      ),
      button("neo-refresh", "Refresh"),
    );
    if (explanation) {
      const f = object(explanation.fact);
      const body = text(f.body);
      children.push(
        label(text(f.subject)),
        label(
          `${f.is_valid ? "Valid" : "Invalid"} · ${strings(f.tags).includes("probation") ? "probation · " : ""}${f.provenance} · ${strings(f.tags).join(", ")}`,
        ),
        label(
          `${rows(explanation.supporting_evidence).length} supporting outcomes · ${rows(explanation.contradicting_evidence).length} conflicting outcomes`,
        ),
        label(
          details ? body : body.slice(0, 400) + (body.length > 400 ? "…" : ""),
        ),
        button(
          "neo-details",
          details ? "Hide provenance details" : "Show provenance details",
          "d",
        ),
      );
      if (details)
        children.push(
          Code({
            language: "json",
            source: cappedJson(
              {
                fact: f,
                supporting: explanation.supporting_evidence,
                conflicting: explanation.contradicting_evidence,
                supersession: explanation.supersession,
                mutations_and_rollback: explanation.mutation_history,
                retrieval: explanation.retrieval_history,
              },
            ),
          }),
        );
      for (const c of strings(explanation.cautions))
        children.push(label(c, "warning"));
    }
    const evidence = rows(current?.result?.memory_evidence);
    if (evidence.length)
      for (const f of evidence) {
        const description = `${f.subject || f.fact_id} · ${f.scope || "scope unknown"} · retrieved · ${f.included_in_context ? "included" : "not included"} · ${f.hard_cited === true ? "hard cited" : "no hard citation"}`;
        children.push(
          f.scope === "project"
            ? button("fact-" + f.fact_id, description)
            : label(description),
        );
      }
    else
      children.push(
        label(
          "No retrieval evidence from this run. Stored facts below are not a claim of use.",
        ),
      );
    const facts = matchingFacts();
    const last = Math.max(0, Math.ceil(facts.length / FACTS_PER_PAGE) - 1);
    const page = Math.min(memoryPage, last);
    children.push(
      Input({
        key: "neo-memory-query",
        label: "Find stored facts",
        value: memoryQuery,
        onInput: () => {},
        onSubmit: () => {},
      }),
      label(
        `${facts.length} matching facts · page ${page + 1} / ${last + 1}`,
      ),
      Box({
        flexDirection: "row",
        gap: 1,
        children: [
          button("neo-prev-page", "Previous"),
          button("neo-next-page", "Next"),
        ],
      }),
    );
    for (const f of facts.slice(
      page * FACTS_PER_PAGE,
      page * FACTS_PER_PAGE + FACTS_PER_PAGE,
    ))
      children.push(
        button(
          "fact-" + f.id,
          `${f.is_valid ? "" : "[invalid] "}${f.probation ? "[probation] " : ""}${text(f.missing_file) ? "[missing file] " : ""}${f.subject || f.id}`,
        ),
      );
    children.push(
      Input({
        key: "neo-fact",
        label: "Explain project fact ID",
        value: "",
        onSubmit: () => {},
      }),
    );
  } else {
    children.push(
      label(
        "Analysis completion is separate from verified application and independent acceptance.",
        "subtle",
      ),
    );
    const result = current?.result;
    if (result) {
      const meta = object(result.metadata);
      const scalars = Object.entries(meta).filter(
        ([, v]) => ["string", "number", "boolean"].includes(typeof v) && v !== "",
      );
      children.push(label("Execution", "remember"));
      for (const [k, v] of scalars) children.push(label(`${k} · ${String(v)}`));
      children.push(label("Validation · " + validationText(result)));
      const checks = rows(result.static_checks);
      children.push(label(`Checks · ${checks.length}`, "remember"));
      for (const c of checks)
        children.push(
          label(`${c.tool_name} · ${c.status || "status unavailable"} · ${text(c.summary)}`),
        );
      const memory = rows(result.memory_evidence);
      const included = memory.filter((m) => m.included_in_context).length;
      const cited = memory.filter((m) => m.hard_cited === true).length;
      children.push(
        label(
          `Memory · ${memory.length} retrieved · ${included} included · ${cited} hard cited`,
          "remember",
        ),
      );
      children.push(
        button("neo-details", details ? "Hide raw evidence" : "Show raw evidence", "d"),
      );
      if (details)
        children.push(
          Code({
            language: "json",
            source: cappedJson({
              execution: result.metadata,
              validation: result.validation_assessment,
              checks: result.static_checks,
              memory: result.memory_evidence,
            }),
          }),
        );
    }
    const events = current?.parser.events ?? [];
    if (events.length) children.push(label("Progress", "remember"));
    for (const event of events)
      children.push(label(`${event.type}: ${event.message ?? ""}`));
    if (current?.parser.malformed)
      children.push(
        label(
          `${current.parser.malformed} malformed progress lines ignored.`,
          "warning",
        ),
      );
  }
  if (inspectorError) children.push(label(inspectorError, "warning"));
  for (const c of strings(snapshot.cautions))
    children.push(label(c, "warning"));
  return Box({ flexDirection: "column", gap: 1, children });
}

// ---------------------------------------------- Neo joining in (see auto.ts)

const field = (value: unknown, key: string): string =>
  text(object(value)[key]);

const autoEdited = new Set<string>();
const reviewed = new Set<string>();
const diagnosed = new Set<string>();
const planned = new Set<string>();
let pending: string[] = [];
let recallMissing = false;
/** Diagnoses running now and started this session: a batch of different
 * failing commands must not fan out into parallel paid runs. */
let diagnosing = 0,
  diagnoses = 0;
/** Bumped by resetAuto: background work started under an older value belongs
 * to a session that has ended and must not deliver into this one. */
let autoGeneration = 0;
/** What the person last typed, and the beats Neo spoke lately: the situation
 * his personality answers to, kept here because each `neo` run forgets it. */
let lastAsk = "";
let spokenBeats: string[] = [];

// ------------------------------------------------ Neo, visible in the session
/** Runs Neo is doing on his own right now, for the band and the spinner. */
const activities = new Map<string, { kind: AutoKind; subject: string; started: number }>();
let activityTicker: Timer | undefined;
/** His latest finding, shown in the band until dismissed or replaced. */
let finding:
  | { kind: AutoKind; result: Document; expanded: boolean; prompts: number }
  | undefined;
/** Lessons recall handed Claude this session, by id: they stay in its context,
 * so recall is told to skip them (`NEO_RECALL_SEEN`) until a compaction drops
 * them. Re-sent on every matching prompt, one lesson reached a session 43
 * times. */
const recalledIds = new Set<string>();
/** What recall gave this prompt, shown until the next one. */
let recalled: { text: string; expanded: boolean } | undefined;
let pendingNotice = "";
/** Subagent ids seen this session, so their spinners keep their own words. */
const subagents = new Set<string>();

const NOTICE_CAP = 400;

/** A line the person sees in the conversation and the model never reads.
 * `$.ui.log`, not a `session.append` system row: the terminal stores those as
 * informational rows and does not draw them, so Neo spoke and nobody saw it
 * (found in the first live session). */
async function noticeRow($: EngineInterface, line: string) {
  if (!line.trim()) return;
  try {
    $.ui.log(line);
  } catch {
    /* a refused line costs the person a line, never the session */
  }
}

/** A row the model reads and the person does not see as typed. False when the
 * engine refused it, so the caller can deliver another way. */
async function modelRow($: EngineInterface, body: string): Promise<boolean> {
  try {
    const stored = await $.session.append({
      message: { type: "user", content: [{ type: "text", text: body }] },
    });
    return !stored?.deny;
  } catch {
    return false;
  }
}

/** A notice held until the prompt it answers is stored: flushed at the
 * turn's first main-thread step, tool call or Stop, whichever comes first. */
async function flushNotice($: EngineInterface) {
  if (!pendingNotice) return;
  const line = pendingNotice;
  pendingNotice = "";
  await noticeRow($, line);
}

/** Neo found something: his line in the conversation, the details in the band. */
async function present($: EngineInterface, kind: AutoKind, result: Document) {
  finding = { kind, result, expanded: false, prompts: 0 };
  const line = voiceLine(result);
  await noticeRow($, line.length > NOTICE_CAP ? line.slice(0, NOTICE_CAP) + "…" : line);
  redraw($);
}

async function autoNeo(
  $: EngineInterface,
  cwd: string,
  session: string,
  prompt: string,
  kind: AutoKind,
  subject = "",
  errorTrace = "",
): Promise<Document | undefined> {
  const id = crypto.randomUUID();
  // What became of each run, so "does Neo help?" is answerable from the
  // ledger: failed, nothing, prose (findings without a diff: a review then
  // stays silent) or change. A failure also says why, and every run how long
  // it took: 85 of 93 reviews once failed, all at the deadline, and the ledger
  // could not tell that from a crash.
  let status = "failed";
  let reason = "";
  const started = await $.clock.now();
  await record($, "event", { event: "auto_started", kind, session_id: session, invocation_id: id, cwd });
  try {
    activities.set(id, { kind, subject, started });
    nowMs = await $.clock.now();
    activityTicker ??= $.clock.every(1000, () => {
      void $.clock.now().then(
        (t) => {
          nowMs = t;
          redraw($);
        },
        () => {},
      );
    });
    redraw($);
    // `--fast`: a prompt new to Neo's memory would otherwise convene the
    // multi-agent panel, which took 126 s on a two-line repository, so every
    // first review there ran out the deadline and delivered nothing.
    const run = await $.process.run(
      [binary, "--stdin-json", "--json", "--mode", "learn", "--fast"],
      {
        cwd,
        stdin: JSON.stringify(
          autoPayload(
            prompt,
            cwd,
            session,
            crypto.randomUUID(),
            { ask: lastAsk, recent: spokenBeats },
            errorTrace,
          ),
        ),
        timeoutMs: AUTO_DEADLINE,
      },
    );
    if (run.exitCode !== 0) {
      reason = `exit ${run.exitCode}`;
      return undefined;
    }
    let result: Document;
    try {
      result = JSON.parse(run.stdout) as Document;
    } catch {
      reason = "unreadable output";
      return undefined;
    }
    if (!result || result.error) {
      reason = "neo error";
      return undefined;
    }
    spokenBeats = rememberBeat(spokenBeats, text(object(result.orchestrator).beat_id));
    status = hasVerifiableSuggestion(result) ? "change" : hasFindings(result) ? "prose" : "nothing";
    return result;
  } catch {
    // `$.process.run` rejects both when the command cannot start and when it
    // is still running at the deadline; only the time taken tells them apart.
    reason = (await $.clock.now()) - started >= AUTO_DEADLINE ? "timed out" : "failed early";
    return undefined; // Neo joining in must never break the session.
  } finally {
    activities.delete(id);
    if (!activities.size) {
      activityTicker?.cancel();
      activityTicker = undefined;
    }
    redraw($);
    await record($, "event", {
      event: "auto_finished", kind, session_id: session, invocation_id: id, cwd, status,
      ...(reason ? { reason } : {}),
      elapsed_ms: Math.round((await $.clock.now()) - started),
    });
  }
}

async function autoRecall(
  $: EngineInterface,
  cwd: string,
  prompt: string,
  transcript: string,
): Promise<{ text: string; beat: string; ids: string[] }> {
  const none = { text: "", beat: "", ids: [] };
  if (recallMissing) return none;
  // A session reset while recall runs makes its answer another session's.
  const owner = autoGeneration;
  try {
    // The transcript lets recall read a short follow-up ("fix it") with the
    // reply it answers. An environment variable, not a flag: an older CLI
    // ignores it, where an unknown flag exits 2 and switches recall off below.
    const run = await $.process.run(
      [binary, "recall", "--cwd", cwd, "--json", "--recent-beats", spokenBeats.join(",")],
      {
        stdin: prompt,
        timeoutMs: RECALL_TIMEOUT,
        env: { NEO_RECALL_TRANSCRIPT: transcript, NEO_RECALL_SEEN: [...recalledIds].join(",") },
      },
    );
    if (owner !== autoGeneration) return none;
    // An older CLI without `recall` answers with an argparse error (exit 2).
    if (run.exitCode === 2) recallMissing = true;
    if (run.exitCode !== 0 || !run.stdout.trim()) return none;
    const result = JSON.parse(run.stdout) as Document;
    spokenBeats = rememberBeat(spokenBeats, text(result.beat_id));
    const ids = Array.isArray(result.fact_ids)
      ? result.fact_ids.filter((i): i is string => typeof i === "string" && i !== "")
      : [];
    return { text: text(result.text).trim(), beat: text(result.beat_id), ids };
  } catch {
    return none;
  }
}

/** What one edited path changed: its diff against HEAD, or, for a file git
 * does not track (new, outside a repository, or a repository with no
 * commits), its whole content as an addition. "" when it matches HEAD. */
async function autoDiffOf($: EngineInterface, cwd: string, path: string) {
  try {
    const tracked = await $.process.run(["git", "diff", "HEAD", "--", path], {
      cwd,
      timeoutMs: 10000,
    });
    if (tracked.exitCode === 0 && tracked.stdout.trim()) return tracked.stdout;
    // `--no-index` exits 1 when the files differ, which is the normal case.
    const added = await $.process.run(
      ["git", "diff", "--no-index", "--", "/dev/null", path],
      { cwd, timeoutMs: 10000 },
    );
    if (added.exitCode === 1 && added.stdout.trim()) {
      // A tracked file that matches HEAD also differs from /dev/null; only an
      // untracked or out-of-repository file is a real addition.
      const known = await $.process.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        { cwd, timeoutMs: 10000 },
      );
      return known.exitCode === 0 ? "" : added.stdout;
    }
  } catch {
    /* no diff for this path */
  }
  return "";
}

function autoDrain(): string[] {
  const out = pending;
  pending = [];
  return out;
}

function withAutoContext<R extends { additionalContext?: string[] }>(
  result: R,
  extra: string[],
): R {
  return extra.length
    ? {
        ...result,
        additionalContext: [...(result.additionalContext ?? []), ...extra],
      }
    : result;
}

/** Called from the mod's one `classic.PostToolUse` hook, after the rest of the
 * chain has answered: the engine allows one unmatched hook per event per
 * module. Optional work never costs the chain its answer. */
async function afterTool(
  $: EngineInterface,
  e: {
    tool_name: string;
    tool_input: unknown;
    tool_response?: unknown;
    cwd: string;
    session_id: string;
    agent_id?: string;
  },
  result: { additionalContext?: string[] },
) {
  try {
    if (EDIT_TOOLS.includes(e.tool_name)) {
      // A subagent's edits are part of the turn the main thread reviews.
      const path =
        field(e.tool_input, "file_path") ||
        field(e.tool_input, "notebook_path");
      if (path) autoEdited.add(path);
    }
    // Findings belong to the main thread: a subagent must not consume them.
    if (e.agent_id) {
      subagents.add(e.agent_id);
      return result;
    }
    await flushNotice($);
    const extra = autoDrain();
    if (e.tool_name === "ExitPlanMode") {
      let plan =
        field(e.tool_response, "plan") || field(e.tool_input, "plan");
      const file = field(e.tool_response, "filePath");
      if (!plan.trim() && file) {
        try {
          plan = String(await $.fs.read(file));
        } catch {
          plan = "";
        }
      }
      const key = fingerprint(plan);
      if (plan.trim() && !planned.has(key)) {
        planned.add(key);
        const found = await autoNeo(
          $, e.cwd, e.session_id, architectPrompt(plan), "architect",
        );
        if (found && hasFindings(found)) {
          extra.push(injection("architect", found));
          await present($, "architect", found);
        }
      }
    }
    return withAutoContext(result, extra);
  } catch {
    return result;
  }
}

/** A new session starts with nothing reviewed, diagnosed or pending. */
function resetAuto() {
  autoGeneration++;
  autoEdited.clear();
  reviewed.clear();
  diagnosed.clear();
  planned.clear();
  pending = [];
  recallMissing = false;
  diagnosing = 0;
  diagnoses = 0;
  lastAsk = "";
  spokenBeats = [];
  activities.clear();
  activityTicker?.cancel();
  activityTicker = undefined;
  finding = undefined;
  recalled = undefined;
  recalledIds.clear();
  pendingNotice = "";
  subagents.clear();
}

function registerAuto(on: On) {
  on("classic.UserPromptSubmit", async ($, e, next) => {
    const result = await next(e);
    try {
      // Only prompts a person typed get memory: task notifications and other
      // machine-injected turns are not questions.
      if ((e.source ?? "user") !== "user") return result;
      lastAsk = e.prompt;
      recalled = undefined;
      // A finding stays in the band through the next prompt, then gives way:
      // its notice row keeps it in the conversation.
      if (finding && ++finding.prompts > 1) finding = undefined;
      const extra = autoDrain();
      const { text: memory, beat, ids } = await autoRecall($, e.cwd, e.prompt, e.transcript_path ?? "");
      if (memory) {
        extra.push(recallInjection(memory));
        // What Claude was handed, by id: a lesson delivered across sessions
        // leaves probation (`FactStore.lift_probation_by_recall`). Recorded
        // from a timer so the prompt does not wait on a process start.
        // Marked seen here, where it is handed over, never before.
        for (const i of ids) recalledIds.add(i);
        if (ids.length) {
          const delivered = { event: "recall_delivered", session_id: e.session_id, cwd: e.cwd, fact_ids: ids };
          $.clock.after(0, () => void record($, "event", delivered));
        }
        recalled = { text: memory, expanded: false };
        // He speaks in the conversation only when a beat earned a line;
        // otherwise the lessons wait quietly in the band. Logged now, the line
        // would sit above the prompt it answers: it waits for the turn's
        // first step, after the prompt is drawn.
        // The host already prefixes a logged line with the plugin's name
        // ("neo: …"), so his own "Neo:" would read twice.
        if (beat) pendingNotice = (memory.split("\n")[0] ?? "").replace(/^Neo:\s*/, "");
      }
      redraw($);
      return withAutoContext(result, extra);
    } catch {
      return result;
    }
  }).catch(async ($, e, next) => next(e));

  on("classic.PostToolUseFailure", async ($, e, next) => {
    const result = await next(e);
    try {
      const command = field(e.tool_input, "command");
      const key = command + "\n" + fingerprint(e.error.slice(-2000));
      if (
        e.tool_name !== "Bash" ||
        e.agent_id ||
        e.is_interrupt ||
        !isCheckCommand(command) ||
        diagnosed.has(key) ||
        diagnosing > 0 ||
        diagnoses >= MAX_DIAGNOSES
      )
        return withAutoContext(result, e.agent_id ? [] : autoDrain());
      diagnosed.add(key);
      diagnosing++;
      diagnoses++;
      const prompt = debugPrompt(command, e.error);
      const trace = e.error.slice(-8000);
      const owner = autoGeneration;
      // In the background: the failure is delivered now, the diagnosis at the
      // next main-thread hook. Work that outlives this dispatch runs from a
      // timer; a call left in flight when the hook returns is aborted.
      $.clock.after(0, () => {
        if (owner !== autoGeneration) return;
        void autoNeo($, e.cwd, e.session_id, prompt, "debug", command, trace)
          .then(
            async (found) => {
              if (owner === autoGeneration && found && hasFindings(found)) {
                pending.push(injection("debug", found, command));
                await present($, "debug", found);
              }
            },
            () => {},
          )
          .finally(() => {
            if (owner === autoGeneration) diagnosing--;
          });
      });
      return withAutoContext(result, autoDrain());
    } catch {
      return result;
    }
  }).catch(async ($, e, next) => next(e));

  on("classic.Stop", async ($, e, next) => {
    const result = await next(e);
    try {
      // Once any Stop hook has continued the turn, let it end; the edits
      // stay recorded and are reviewed at the next Stop that is not a
      // continuation, so nothing goes unreviewed for being continued.
      await flushNotice($);
      if (e.stop_hook_active || !autoEdited.size) return result;
      const files = [...autoEdited].sort();
      autoEdited.clear();
      const diffs: string[] = [];
      for (const path of files) {
        const diff = await autoDiffOf($, e.cwd, path);
        if (diff) diffs.push(diff);
      }
      // Nothing changed against HEAD (an edit undone): nothing to review.
      if (!diffs.length) return result;
      const diff = diffs.join("\n");
      const key = fingerprint(diff);
      if (reviewed.has(key)) return result;
      reviewed.add(key);
      const found = await autoNeo(
        $, e.cwd, e.session_id, reviewPrompt(files, diff), "review",
      );
      // Only a concrete change continues a finishing turn; prose waits.
      if (!found || !hasVerifiableSuggestion(found)) return result;
      await present($, "review", found);
      // The findings go to Claude in a row it reads, so the Stop message the
      // person sees stays one line; if the engine refuses that row, they ride
      // in the Stop message itself.
      const delivered = await modelRow($, injection("review", found));
      return {
        ...result,
        block: [
          ...(result.block ? [result.block] : []),
          ...(delivered
            ? [REVIEW_CONTINUE]
            : [
                injection("review", found),
                "Address what holds up, or say briefly why not, then finish.",
              ]),
        ].join("\n\n"),
      };
    } catch {
      return result;
    }
  }).catch(async ($, e, next) => next(e));
}

/** Neo's rows in the band above the prompt: what he is doing, what he
 * remembered for this prompt, and his latest finding; a dim "listening" line
 * when he has nothing to say, which is most of the time. */
function presence($: EngineInterface, e: RenderInput<"AbovePrompt">): RenderNode[] {
  const { Box, Text, Button, Markdown } = $.ui.resolve(e);
  const out: RenderNode[] = [];
  const neo = Box({ flexShrink: 0, children: [Text({ children: "Neo:", color: "remember" })] });
  for (const a of activities.values())
    out.push(
      Box({
        flexDirection: "row",
        gap: 1,
        children: [
          neo,
          Text({
            children: `${activityLabel(a.kind, a.subject)} · ${elapsed(nowMs - a.started)}`,
            dimColor: true,
            wrap: "truncate-end",
          }),
        ],
      }),
    );
  if (recalled) {
    const [opener = "", ...lessons] = recalled.text.split("\n");
    const subjects = lessons
      .filter((l) => l.startsWith("- "))
      .map((l) => l.slice(2).split(" [")[0]);
    out.push(
      Box({
        flexDirection: "row",
        gap: 1,
        children: [
          Box({ flexShrink: 0, children: [Text({ children: opener, color: "remember" })] }),
          Box({
            flexShrink: 1,
            children: [Text({ children: subjects.join(" · "), dimColor: true, wrap: "truncate-end" })],
          }),
          Button({
            key: "neo-recall-toggle",
            label: recalled.expanded ? "hide" : "lessons",
            onPress: () => {},
          }),
        ],
      }),
    );
    if (recalled.expanded) out.push(Markdown({ text: lessons.join("\n"), dimColor: true }));
  }
  if (finding) {
    out.push(
      Box({
        flexDirection: "row",
        gap: 1,
        children: [
          neo,
          Text({ children: voiceLine(finding.result), wrap: "truncate-end" }),
          Button({
            key: "neo-finding-toggle",
            label: finding.expanded ? "hide" : "details",
            hotkey: "d",
            onPress: () => {},
          }),
          Button({ key: "neo-finding-dismiss", label: "dismiss", hotkey: "x", onPress: () => {} }),
        ],
      }),
    );
    if (finding.expanded) out.push(Markdown({ text: findingMarkdown(finding.result) }));
  }
  // With nothing to say he is still there: one dim line, so a quiet Neo is
  // not mistaken for a missing one.
  if (!out.length)
    out.push(
      Box({
        flexDirection: "row",
        gap: 1,
        children: [
          Box({ flexShrink: 0, children: [Text({ children: "Neo:", dimColor: true })] }),
          Text({
            children: recallMissing ? "this CLI has no recall; update neo" : "listening",
            dimColor: true,
            wrap: "truncate-end",
          }),
        ],
      }),
    );
  return out;
}

/** An asked run, drawn live in its command's row: progress, then the answer. */
function askCard($: EngineInterface, e: RenderInput<"CommandOutput">, run: Run): RenderElement {
  const { Box, Text, Button, Markdown } = $.ui.resolve(e);
  const children: RenderNode[] = [
    Text({ children: "Neo · " + progressLine(run), color: run.error ? "error" : "remember" }),
  ];
  if (run.status === "running")
    children.push(
      Box({
        flexDirection: "row",
        gap: 1,
        children: [
          Button({ key: "neo-card-cancel", label: "Cancel", onPress: () => {} }),
          Text({ children: "/neo-cancel", dimColor: true }),
        ],
      }),
    );
  if (run.error) children.push(Text({ children: run.error, color: "error" }));
  if (run.result)
    children.push(
      Text({ children: voiceLine(run.result) }),
      Markdown({ text: findingMarkdown(run.result) }),
    );
  return Box({ flexDirection: "column", children });
}

/** `/neo-workspace`: what Neo knew about this project when it was asked,
 * inline. Each row keeps its own moment, so an old row does not redraw to
 * the current state. */
const workspaceRows = new Map<string, { snapshot: Document; line: string; error: string }>();
function workspaceCard($: EngineInterface, e: RenderInput<"CommandOutput">): RenderElement {
  const { Box, Text, Button, Markdown } = $.ui.resolve(e);
  const id = /\[([0-9a-f]{8})\]/.exec(e.props.text)?.[1] ?? "";
  const frozen = workspaceRows.get(id);
  if (!frozen) return Text({ children: e.props.text });
  const { snapshot, line, error: inspectorError } = frozen;
  const facts = rows(snapshot.facts).filter((f) => f.is_valid !== false);
  const lines = [
    `**${snapshot.fact_count ?? "?"} project facts** · ${snapshot.probation_count ?? "?"} on probation`,
    ...facts.slice(0, 8).map((f) => `- ${text(f.subject) || text(f.id)}`),
  ];
  const children: RenderNode[] = [Text({ children: "Neo", color: "remember" })];
  if (line) children.push(Text({ children: line }));
  if (inspectorError) children.push(Text({ children: inspectorError, color: "error" }));
  children.push(
    Markdown({ text: lines.join("\n") }),
    Box({
      flexDirection: "row",
      gap: 1,
      children: [
        Button({ key: "neo-open-panel", label: "Open panel", onPress: () => {} }),
        Text({ children: "/neo-workspace panel", dimColor: true }),
      ],
    }),
  );
  return Box({ flexDirection: "column", children });
}

/** `/neo-explain`, readable inline. */
function explanationMarkdown(): string {
  const x = explanation ?? {};
  const f = object(x.fact);
  return [
    `**${text(f.subject) || text(f.id)}**`,
    `${f.is_valid === false ? "Invalid" : "Valid"} · ${text(f.provenance)} · ${strings(f.tags).join(", ")}`,
    `${rows(x.supporting_evidence).length} supporting outcomes · ${rows(x.contradicting_evidence).length} conflicting outcomes`,
    text(f.body),
    ...strings(x.cautions).map((c) => `- ${c}`),
  ]
    .filter(Boolean)
    .join("\n\n");
}

export const register: Register = (on, options) => {
  binary = typeof options.executable === "string" ? options.executable : "neo";
  registerAuto(on);
  on("session.start", async ($, e, next) => {
    generation++;
    resetAuto();
    askedRuns.clear();
    current = undefined;
    previous = undefined;
    snapshot = {};
    explanation = undefined;
    context = "";
    question = "";
    lastInvocation = "";
    inspectorError = "";
    tab = "results";
    details = false;
    mode = "reason";
    memoryQuery = "";
    memoryPage = 0;
    notice = "";
    turns.clear();
    binary = (await $.env.get("NEO_MOD_EXECUTABLE")) || binary;
    session = await $.session.id();
    cwd = e.cwd;
    interactive = e.isInteractive;
    drawable = e.surface === "terminal" || e.surface === "desktop";
    recording = (await $.env.get("NEO_HOOKS")) !== "0";
    await $.command.register({
      name: "neo-workspace",
      description: "Open Neo findings and memory",
      immediate: true,
    });
    await $.command.register({
      name: "neo-status",
      description: "Inspect Neo progress and project status",
      immediate: true,
    });
    await $.command.register({
      name: "neo-ask",
      description: "Ask Neo directly",
      argumentHint:
        "[reason|review|debug|architect|optimize|pattern] <question>",
      immediate: true,
    });
    await $.command.register({
      name: "neo-followup",
      description: "Follow up on Neo’s prior result",
      argumentHint: "<question>",
      immediate: true,
    });
    await $.command.register({
      name: "neo-explain",
      description: "Explain a project memory fact without inference",
      argumentHint: "<fact-id>",
      immediate: true,
    });
    await $.command.register({
      name: "neo-cancel",
      description: "Cancel this workspace’s Neo run",
      immediate: true,
    });
    return next(e);
  });
  // The workspace is a card in the conversation; `/neo-workspace panel`
  // still opens the side pane for whoever wants it.
  on("command.run", { command: "neo-workspace" }, async ($, e) => {
    await inspect($);
    if (e.args.trim() === "panel") await open($);
    const id = crypto.randomUUID().slice(0, 8);
    workspaceRows.set(id, {
      snapshot,
      line: finding ? voiceLine(finding.result) : "",
      error: inspectorError,
    });
    if (workspaceRows.size > 20) workspaceRows.delete(workspaceRows.keys().next().value!);
    return {
      text: `${statusText()} [${id}]` + (inspectorError ? "\n" + inspectorError : ""),
      exitCode: inspectorError ? 1 : 0,
    };
  });
  on("command.run", { command: "neo-status" }, async ($) => {
    if (!current) await inspect($);
    return {
      text: statusText() + (inspectorError ? "\n" + inspectorError : ""),
      exitCode: inspectorError ? 1 : 0,
    };
  });
  on("command.run", { command: "neo-ask" }, async ($, e) => {
    const first = e.args.trim().split(/\s+/)[0] ?? "",
      selected = MODES.includes(first) ? first : "reason";
    const answer = await start(
      $,
      selected === first ? e.args.trim().slice(first.length).trim() : e.args,
      selected,
      false,
      true,
    );
    return {
      text: answer,
      exitCode:
        current?.status === "failed" || current?.status === "cancelled" ? 1 : 0,
    };
  });
  on("command.run", { command: "neo-followup" }, async ($, e) => {
    const answer = await start($, e.args, mode, true, true);
    return {
      text: answer,
      exitCode:
        current?.status === "failed" || current?.status === "cancelled" ? 1 : 0,
    };
  });
  on("command.run", { command: "neo-explain" }, async ($, e) => {
    if (!e.args.trim())
      return { text: "Usage: /neo-explain <project-fact-id>", exitCode: 1 };
    if (!snapshot.schema_version) await inspect($);
    await inspect($, e.args.trim());
    tab = "memory";
    return {
      text:
        inspectorError ||
        (drawable ? explanationMarkdown() : JSON.stringify(explanation, null, 2)),
      exitCode: inspectorError ? 1 : 0,
    };
  });
  on("command.run", { command: "neo-cancel" }, async ($) => {
    await stop($);
    return { text: statusText() };
  });
  on("ui.render", { component: "Pane", requestId: PANE }, async ($, e) =>
    render($, e),
  );
  on("ui.render", { component: "AbovePrompt" }, async ($, e, next) => {
    const existing = await next(e);
    if (e.props.hasSurvey) return existing;
    const parts: RenderNode[] = [];
    if (
      (current?.status === "running" || current?.status === "stopping") &&
      !current.fromCommand
    )
      parts.push(render($, e));
    parts.push(...presence($, e));
    return parts.length
      ? $.ui.resolve(e).Box({ flexDirection: "column", children: [existing, ...parts] })
      : existing;
  });
  on("ui.press", { component: "AbovePrompt", element: "neo-finding-toggle" }, async ($, e) => {
    if (finding) finding.expanded = !finding.expanded;
    redraw($);
    return { element: e.element };
  });
  on("ui.press", { component: "AbovePrompt", element: "neo-finding-dismiss" }, async ($, e) => {
    finding = undefined;
    redraw($);
    return { element: e.element };
  });
  on("ui.press", { component: "AbovePrompt", element: "neo-recall-toggle" }, async ($, e) => {
    if (recalled) recalled.expanded = !recalled.expanded;
    redraw($);
    return { element: e.element };
  });
  // While a review or a plan check holds the turn, the working line says so.
  on("ui.render", { component: "Spinner" }, async ($, e, next) => {
    const holding = [...activities.values()].find((a) => a.kind !== "debug");
    // A subagent's spinner is its own.
    const subagent = subagents.has(e.requestId);
    return holding && !subagent
      ? next({
          ...e,
          props: { ...e.props, message: "Neo is " + activityLabel(holding.kind, holding.subject) },
        })
      : next(e);
  });
  // Asked runs draw live in the conversation, where the command was typed.
  for (const command of ["neo-ask", "neo-followup"])
    on("ui.render", { component: "CommandOutput", props: { command } }, async ($, e, next) => {
      const id = /\[([0-9a-f]{8})\]/.exec(e.props.text)?.[1];
      const run = id ? askedRuns.get(id) : undefined;
      return run ? askCard($, e, run) : next(e);
    });
  on("ui.press", { component: "CommandOutput", element: "neo-card-cancel" }, async ($, e) => {
    await stop($);
    return { element: e.element };
  });
  on("ui.render", { component: "CommandOutput", props: { command: "neo-workspace" } }, async ($, e) =>
    workspaceCard($, e),
  );
  on("ui.render", { component: "CommandOutput", props: { command: "neo-explain" } }, async ($, e) =>
    $.ui.resolve(e).Markdown({ text: e.props.text }),
  );
  on("ui.press", { component: "CommandOutput", element: "neo-open-panel" }, async ($, e) => {
    await open($);
    return { element: e.element };
  });
  on("ui.input", { requestId: PANE }, async ($, e) => {
    if (e.element === "neo-question") question = e.value;
    if (e.element === "neo-context") {
      context = e.value;
      notice = "";
    }
    if (e.element === "neo-memory-query") {
      memoryQuery = e.value;
      memoryPage = 0;
    }
    if (e.kind === "submit" && e.element === "neo-question")
      await ask($, false);
    if (e.kind === "submit" && e.element === "neo-fact")
      await inspect($, e.value.trim());
    redraw($);
    return { element: e.element, value: e.value };
  });
  on("ui.select", { requestId: PANE, element: "neo-mode" }, async ($, e) => {
    if (MODES.includes(e.value)) mode = e.value;
    redraw($);
    return { element: e.element, value: e.value };
  });
  on("ui.press", { requestId: PANE }, async ($, e) => {
    if (["neo-results", "neo-memory", "neo-evidence"].includes(e.element))
      details = false;
    if (e.element === "neo-results") tab = "results";
    if (e.element === "neo-memory") {
      tab = "memory";
      // Status is a full read of the project store; Refresh re-reads it. A
      // read from another checkout, or none, is read again.
      if (!snapshot.schema_version || snapshot.checkout !== (await $.session.cwd()))
        await inspect($);
    }
    if (e.element === "neo-evidence") tab = "evidence";
    if (e.element === "neo-details") details = !details;
    if (e.element === "neo-refresh") await inspect($);
    if (e.element === "neo-run" || e.element === "neo-followup")
      await ask($, e.element === "neo-followup");
    if (e.element === "neo-cancel") await stop($);
    if (e.element === "neo-selection") {
      const selected = (await $.ui.selection())?.text ?? "";
      // Nothing selected keeps what is there: replacing it with "" lost
      // typed context without a word.
      if (selected.trim()) {
        context = selected;
        notice = "";
      } else
        notice =
          "No selection is visible to Claude Code (selection needs fullscreen); context unchanged.";
    }
    if (e.element === "neo-prev-page")
      memoryPage = Math.max(0, Math.min(memoryPage, lastPage()) - 1);
    if (e.element === "neo-next-page")
      memoryPage = Math.min(memoryPage + 1, lastPage());
    if (e.element.startsWith("fact-")) await inspect($, e.element.slice(5));
    if (e.element === "neo-share" && current?.result)
      await $.prompt.fill({
        text:
          "Please consider this Neo analysis:\n\n" + resultText(current.result),
      });
    redraw($);
    return { element: e.element };
  });
  on(
    "ui.press",
    { component: "AbovePrompt", element: "neo-cancel" },
    async ($, e) => {
      await stop($);
      return { element: e.element };
    },
  );
  on("turn.start", async ($, e, next) => {
    session = await $.session.id();
    cwd = await $.session.cwd();
    turns.set("main", e.turnId);
    await record($, "event", {
      event: "turn_started",
      session_id: session,
      turn_id: e.turnId,
      cwd,
    });
    return next(e);
  });
  on("turn.step", async function* ($, e, next) {
    turns.set(e.agentId ?? "main", e.turnId);
    if (e.agentId) subagents.add(e.agentId);
    if (!e.agentId) await flushNotice($);
    return yield* next(e);
  });
  on("turn.complete", async ($, e, next) => {
    await record($, "event", {
      event: "turn_completed",
      session_id: session,
      turn_id: e.turnId,
      agent_id: e.agentId,
      cwd,
      status: e.reason,
    });
    turns.delete(e.agentId ?? "main");
    return next(e);
  });
  on("classic.PostToolUse", async ($, e, next) => {
    // Recency is correlation only. Python still requires git verification and
    // independent outcomes before accepting or promoting any suggestion.
    if (["Edit", "Write", "MultiEdit", "NotebookEdit"].includes(e.tool_name))
      await record($, "record", {
        ...e,
        source: "claude-mod",
        turn_id: turns.get(e.agent_id ?? "main"),
        agent_id: e.agent_id,
        recent_invocation_id: lastInvocation,
      });
    return afterTool($, e, await next(e));
  }).catch(async ($, e, next) => next(e));
  // A compacted conversation no longer holds the lessons recall handed it, so
  // they may be handed again. A subagent's own compaction and a skipped one
  // drop nothing. A `precompute` installs nothing yet, but whether installing
  // it later dispatches again is not documented, so it clears too: the cost is
  // a lesson handed twice, where the other way it could be withheld after the
  // context lost it.
  on("session.compact", async ($, e, next) => {
    const result = await next(e);
    if (!e.agentId && !result.skip) recalledIds.clear();
    return result;
  }).catch(async ($, e, next) => next(e));
  on("session.end", async ($, e, next) => {
    generation++;
    resetAuto();
    askedRuns.clear();
    await stop($, "session ended");
    turns.clear();
    current = undefined;
    previous = undefined;
    explanation = undefined;
    snapshot = {};
    context = "";
    question = "";
    lastInvocation = "";
    inspectorError = "";
    notice = "";
    return next(e);
  });
};

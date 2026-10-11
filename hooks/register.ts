import type {
  EngineInterface,
  HookStream,
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
  started: number;
  elapsed?: number;
  stopped?: string;
};
let current: Run | undefined, previous: Run | undefined;
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
    parser: new NeoStream(),
    started: await $.clock.now(),
  };
  nowMs = run.started;
  details = false;
  current = run;
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
  return !drawable && run.result
    ? resultText(run.result)
    : run.error
      ? "Neo: " + run.error
      : "Neo started. /neo-status shows progress; /neo-cancel stops this run.";
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
        `${snapshot.fact_count ?? "?"} project facts · ${snapshot.probation_count ?? "?"} on probation`,
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
          `${f.is_valid ? "" : "[invalid] "}${f.probation ? "[probation] " : ""}${f.subject || f.id}`,
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

export const register: Register = (on, options) => {
  binary = typeof options.executable === "string" ? options.executable : "neo";
  on("session.start", async ($, e, next) => {
    generation++;
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
  on("command.run", { command: "neo-workspace" }, async ($) => {
    await inspect($);
    await open($);
    return drawable
      ? {}
      : {
          text: statusText() + (inspectorError ? "\n" + inspectorError : ""),
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
    await open($);
    const answer = await start(
      $,
      selected === first ? e.args.trim().slice(first.length).trim() : e.args,
      selected,
    );
    return {
      text: answer,
      exitCode:
        current?.status === "failed" || current?.status === "cancelled" ? 1 : 0,
    };
  });
  on("command.run", { command: "neo-followup" }, async ($, e) => {
    const answer = await start($, e.args, mode, true);
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
    await open($);
    return {
      text:
        inspectorError ||
        (drawable
          ? "Neo: provenance opened in Memory."
          : JSON.stringify(explanation, null, 2)),
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
    return current?.status === "running" || current?.status === "stopping"
      ? $.ui
          .resolve(e)
          .Box({ flexDirection: "column", children: [existing, render($, e)] })
      : existing;
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
    return next(e);
  }).catch(async ($, e, next) => next(e));
  on("session.end", async ($, e, next) => {
    generation++;
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

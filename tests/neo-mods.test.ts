import type { On } from "claude-code";
import { expect, mock, test } from "claude-code/testing";
import {
  NeoStream,
  followupContext,
  openQuestions,
  resultText,
} from "../hooks/protocol.js";

const command = (command: string, args = "") => ({
  command,
  args,
  origin: { kind: "composer" } as const,
  presentation: { isFullscreen: false, columns: 80 },
});
function host(on: On, log: (text: string) => void = () => {}) {
  mock.env(on, {});
  const clock = mock.clock(on);
  on("session.id", () => ({ value: "session1" }));
  on("session.cwd", () => ({ value: "/work" }));
  on("session.start", (_$, e) => ({ cwd: e.cwd }));
  on("command.register", (_$, e) => ({ value: { command: e.name } }));
  on("ui.log", (_$, e: any) => {
    log(String(e?.text ?? e?.message ?? JSON.stringify(e)));
    return { value: undefined };
  });
  on("ui.open", () => {
    opens++;
    return { value: { isPlaced: true } };
  });
  return clock;
}
const SESSION = { cwd: "/work", surface: null, isInteractive: false } as const;
/** Panes opened in this test: Neo must not open one unasked. */
let opens = 0;
const RESULT = {
  orchestrator: {
    summary: "Found an edge case",
    cautions: ["Tests not run", "Check callers"],
  },
  notes: "Return early for empty input.",
  code_suggestions: [],
  static_checks: [],
  memory_evidence: [
    { fact_id: "fact1", included_in_context: true, hard_cited: false },
  ],
};
const PANE = {
  component: "Pane",
  requestId: "neo-workspace",
  props: {
    title: "Neo",
    isFocused: true,
    bodyColumns: 40,
    placement: "inline",
    scroll: { offset: 0, bodyRows: 24, contentRows: 24 },
    view: {},
  },
} as const;

test("partial progress and final JSON require process exit and both contracts", async () => {
  const p = new NeoStream();
  p.push("stderr", 'warning\n{broken}\n{"type":"comp');
  p.push("stderr", 'leted","message":"Reasoned"}\n');
  p.push("stdout", JSON.stringify(RESULT).slice(0, 30));
  p.push("stdout", JSON.stringify(RESULT).slice(30));
  expect(p.malformed).toBe(1);
  expect(p.finish(0, null).orchestrator.summary).toBe("Found an edge case");
  expect(() => p.finish(1, null)).toThrow();
  const incomplete = new NeoStream();
  incomplete.push("stdout", JSON.stringify(RESULT));
  expect(() => incomplete.finish(0, null)).toThrow();
});

test("all cautions survive text fallbacks and bounded follow-ups", async () => {
  expect(resultText(RESULT)).toContain("Tests not run");
  expect(resultText(RESULT)).toContain("Check callers");
  expect(followupContext(RESULT, "Why?")).toContain("Current question:\nWhy?");
  expect(
    followupContext({ ...RESULT, notes: "x".repeat(24000) }, "Why?"),
  ).toContain("[prior analysis truncated]");
});

test("startup does no backend work; status diagnoses CLI version drift", async ($, on) => {
  host(on);
  let calls = 0;
  on("process.run", () => {
    calls++;
    return {
      value: {
        exitCode: 2,
        stdout: "",
        stderr: "unrecognized arguments",
        isStdoutTruncated: false,
        isStderrTruncated: false,
      },
    };
  });
  await $.session.start(SESSION);
  expect(calls).toBe(0);
  const answer = await $.command.run(command("neo-status"));
  expect(answer.text).toContain("matching neo-reasoner");
});

test("headless reasoning uses advise, retains cautions, and follows prior context", async ($, on) => {
  host(on);
  const payloads: any[] = [];
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: "{}",
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  on("process.spawn", async function* (_$, e) {
    payloads.push(JSON.parse(e.input!));
    yield { stream: "stderr", text: '{"type":"completed","message":"Done"}\n' };
    yield { stream: "stdout", text: JSON.stringify(RESULT) };
    return { value: { code: 0, signal: null } };
  });
  await $.session.start(SESSION);
  const first = await $.command.run(command("neo-ask", "debug empty input"));
  expect(first.text).toContain("Check callers");
  expect(payloads[0].operating_mode).toBe("advise");
  expect(payloads[0].task_type).toBe("bugfix");
  const follow = await $.command.run(
    command("neo-followup", "What about None?"),
  );
  expect(follow.text).toContain("Found an edge case");
  expect(payloads[1].prompt).toContain("Return early for empty input");
  expect(payloads[1].execution_identity.parent_task_id).toBe(
    payloads[0].execution_identity.task_id,
  );
});

test("terminal and Desktop panes expose controls and distinguish memory use", async ($, on) => {
  host(on);
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: JSON.stringify({
        schema_version: 1,
        fact_count: 1,
        probation_count: 1,
        facts: [
          {
            id: "fact1",
            subject: "Empty input",
            probation: true,
            is_valid: true,
          },
        ],
      }),
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  await $.session.start(SESSION);
  for (const surface of ["terminal", "desktop"] as const) {
    const ui = await $.ui.mount({ plugin: "neo", surface, ...PANE });
    expect(
      await ui.find({ key: "neo-question" }),
      surface + " question input",
    ).toBeDefined();
    await ui.press({ key: "neo-memory" });
    expect(
      await ui.find({ key: "neo-refresh" }),
      surface + " refresh control",
    ).toBeDefined();
    expect(
      await ui.find({ key: "fact-fact1" }),
      surface + " stored fact",
    ).toBeDefined();
    await ui.press({ key: "neo-results" });
    await ui.unmount();
  }
});

test("successful host edits carry identity without modifying classic behavior", async ($, on) => {
  host(on);
  const records: any[] = [];
  on("process.run", (_$, e) => {
    records.push(JSON.parse(e.init!.stdin!));
    return {
      value: {
        exitCode: 0,
        stdout: "",
        stderr: "",
        isStdoutTruncated: false,
        isStderrTruncated: false,
      },
    };
  });
  on("classic.PostToolUse", () => ({
    additionalContext: ["existing hook context"],
  }));
  await $.session.start(SESSION);
  const result = await $.classic.PostToolUse({
    tool_name: "Edit",
    tool_use_id: "call1",
    tool_input: { file_path: "/work/a.py", new_string: "private" },
    tool_response: {},
  });
  expect(result.additionalContext).toEqual(["existing hook context"]);
  expect(records[0].source).toBe("claude-mod");
  expect(records[0].tool_use_id).toBe("call1");
});

test("process failure is nonzero and a later run can recover", async ($, on) => {
  host(on);
  let attempt = 0;
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: "{}",
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  on("process.spawn", async function* () {
    attempt++;
    if (attempt === 1) {
      yield { stream: "stdout", text: '{"error":"provider unavailable"}' };
      return { value: { code: 1, signal: null } };
    }
    yield { stream: "stderr", text: '{"type":"completed"}\n' };
    yield { stream: "stdout", text: JSON.stringify(RESULT) };
    return { value: { code: 0, signal: null } };
  });
  await $.session.start(SESSION);
  const failed = await $.command.run(command("neo-ask", "Why?"));
  expect(failed.exitCode).toBe(1);
  expect(failed.text).toContain("provider unavailable");
  const recovered = await $.command.run(command("neo-ask", "Try again"));
  expect(recovered.exitCode).toBe(0);
  expect(recovered.text).toContain("Found an edge case");
});

test("deadline closes the owned stream without accepting a partial result", async ($, on) => {
  const clock = host(on);
  let closed = 0;
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: "{}",
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  on("process.spawn", async function* () {
    try {
      while (true) {
        yield {
          stream: "stderr",
          text: '{"type":"planning","message":"Still working"}\n',
        };
        await clock.sleep(1000);
      }
    } finally {
      closed++;
    }
    return { value: { code: null, signal: "SIGTERM" } };
  });
  await $.session.start({
    ...SESSION,
    surface: "terminal",
    isInteractive: true,
  });
  await $.command.run(command("neo-ask", "Long analysis"));
  await clock.advance(241000);
  await clock.settle();
  const status = await $.command.run(command("neo-status"));
  expect(status.text).toContain("cancelled");
  expect(status.text).toContain("4 minutes");
  expect(closed).toBe(1);
});

test("session end clears prior context and follows the host chain", async ($, on) => {
  host(on);
  on("session.end", (_$, e) => ({ sessionId: e.sessionId }));
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: "{}",
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  on("process.spawn", async function* () {
    yield { stream: "stderr", text: '{"type":"completed"}\n' };
    yield { stream: "stdout", text: JSON.stringify(RESULT) };
    return { value: { code: 0, signal: null } };
  });
  await $.session.start(SESSION);
  await $.command.run(command("neo-ask", "Why?"));
  const ended = await $.session.end({
    reason: "clear",
    sessionId: "session1",
    resume: { id: "session1" },
  });
  expect(ended.sessionId).toBe("session1");
  const follow = await $.command.run(command("neo-followup", "Why again?"));
  expect(follow.text).toContain("Run an analysis before");
  expect(follow.text).not.toContain("Found an edge case");
});

test("provenance opens above the fact list with details on demand", async ($, on) => {
  host(on);
  on("process.run", (_$, e) => ({
    value: {
      exitCode: 0,
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
      stdout: JSON.stringify(
        e.argv.includes("explain")
          ? {
              schema_version: 1,
              fact: {
                id: "f",
                subject: "Selected fact",
                body: "Full fact body",
                is_valid: true,
                provenance: "verified",
                tags: ["probation"],
              },
              supporting_evidence: [],
              contradicting_evidence: [],
            }
          : { schema_version: 1, fact_count: 1, probation_count: 1, facts: [] },
      ),
    },
  }));
  await $.session.start({
    ...SESSION,
    surface: "terminal",
    isInteractive: true,
  });
  const answer = await $.command.run(command("neo-explain", "f"));
  expect(answer.text).toContain("**Selected fact**");
  expect(answer.text).toContain("Full fact body");
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  const compact = JSON.stringify(await ui.drawn());
  expect(
    compact.indexOf("Selected fact") < compact.indexOf("No retrieval evidence"),
  ).toBe(true);
  await ui.press({ key: "neo-details" });
  expect(JSON.stringify(await ui.drawn())).toContain("supporting");
  await ui.unmount();
});

test("a caution repeated as an open question is shown once", async () => {
  const result = {
    ...RESULT,
    orchestrator: {
      summary: "s",
      cautions: ["Still open: Simulation found: race in caller", "Tests not run"],
    },
    next_questions: [
      "Simulation found: race in caller",
      "Should empty input raise?",
    ],
  };
  expect(openQuestions(result)).toEqual(["Should empty input raise?"]);
  // Tool-count and confidence questions are voiced as cautions the same way.
  expect(
    openQuestions({
      orchestrator: { cautions: ["Still open: constraint_verifier found 2 issues"] },
      next_questions: ["constraint_verifier found 2 issues"],
    }),
  ).toEqual([]);
  // A question that only ends like a caution is a different question.
  expect(
    openQuestions({
      orchestrator: { cautions: ["Still open: mypy_ruff found 2 issues"] },
      next_questions: ["ruff found 2 issues"],
    }),
  ).toEqual(["ruff found 2 issues"]);
  const text = resultText(result);
  expect(text.split("race in caller").length - 1).toBe(1);
  expect(text).toContain("Open question: Should empty input raise?");
});

function facts(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    id: `fact${i}`,
    subject: `Fact ${i}`,
    probation: false,
    is_valid: true,
  }));
}
function statusHost(on: On, count: number) {
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: JSON.stringify({
        schema_version: 1,
        checkout: "/work",
        fact_count: count,
        probation_count: 0,
        facts: facts(count),
      }),
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
}

test("memory paging stops at the last page so Previous answers at once", async ($, on) => {
  host(on);
  statusHost(on, 45); // two pages of 30
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-memory" });
  for (let i = 0; i < 5; i++) await ui.press({ key: "neo-next-page" });
  expect(JSON.stringify(await ui.drawn())).toContain("page 2 / 2");
  await ui.press({ key: "neo-prev-page" });
  expect(JSON.stringify(await ui.drawn())).toContain("page 1 / 2");
  expect(await ui.find({ key: "fact-fact0" })).toBeDefined();
  await ui.unmount();
});

test("Use selection with nothing selected keeps typed context and says so", async ($, on) => {
  host(on);
  statusHost(on, 0);
  let selection: { text: string } | undefined = { text: "def f(x): ..." };
  on("ui.selection", () => ({ value: selection as any }));
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-selection" });
  expect(JSON.stringify(await ui.drawn())).toContain("Context attached · 13 characters");
  selection = undefined;
  await ui.press({ key: "neo-selection" });
  const drawn = JSON.stringify(await ui.drawn());
  expect(drawn).toContain("Context attached · 13 characters");
  expect(drawn).toContain("No selection is visible to Claude Code");
  await ui.unmount();
});

test("selected context is sent with one question, not silently with the next", async ($, on) => {
  host(on);
  statusHost(on, 0);
  on("ui.selection", () => ({ value: { text: "SECRET_CONTEXT" } as any }));
  const payloads: any[] = [];
  on("process.spawn", async function* (_$, e) {
    payloads.push(JSON.parse(e.input!));
    yield { stream: "stderr", text: '{"type":"completed","message":"Done"}\n' };
    yield { stream: "stdout", text: JSON.stringify(RESULT) };
    return { value: { code: 0, signal: null } };
  });
  await $.session.start(SESSION);
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-selection" });
  await ui.unmount();
  await $.command.run(command("neo-ask", "first question"));
  await $.command.run(command("neo-ask", "unrelated question"));
  expect(payloads[0].prompt).toContain("SECRET_CONTEXT");
  expect(payloads[1].prompt).not.toContain("SECRET_CONTEXT");
});

function spawnHost(on: On, payloads: any[], fail = () => false) {
  on("process.spawn", async function* (_$, e) {
    payloads.push(JSON.parse(e.input!));
    if (fail()) {
      yield { stream: "stdout", text: '{"error":"provider unavailable"}' };
      return { value: { code: 1, signal: null } };
    }
    yield { stream: "stderr", text: '{"type":"completed","message":"Done"}\n' };
    yield { stream: "stdout", text: JSON.stringify(RESULT) };
    return { value: { code: 0, signal: null } };
  });
}

test("a follow-up carries the prior question's context; a new question does not", async ($, on) => {
  host(on);
  statusHost(on, 0);
  on("ui.selection", () => ({ value: { text: "def parse(x): ..." } as any }));
  const payloads: any[] = [];
  spawnHost(on, payloads);
  await $.session.start(SESSION);
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-selection" });
  await ui.unmount();
  await $.command.run(command("neo-ask", "review parse"));
  await $.command.run(command("neo-followup", "and the error path?"));
  await $.command.run(command("neo-ask", "something unrelated"));
  expect(payloads[0].prompt).toContain("def parse(x)");
  expect(payloads[1].prompt).toContain("def parse(x)");
  expect(payloads[2].prompt).not.toContain("def parse(x)");
});

test("a refused or failed run keeps the selected context for the retry", async ($, on) => {
  host(on);
  statusHost(on, 0);
  on("ui.selection", () => ({ value: { text: "KEEP_ME" } as any }));
  const payloads: any[] = [];
  let failNext = true;
  spawnHost(on, payloads, () => {
    const f = failNext;
    failNext = false;
    return f;
  });
  await $.session.start(SESSION);
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-selection" });
  await ui.unmount();
  const refused = await $.command.run(command("neo-ask", "   "));
  expect(refused.text).toContain("Enter a question");
  const failed = await $.command.run(command("neo-ask", "first try"));
  expect(failed.exitCode).toBe(1);
  await $.command.run(command("neo-ask", "retry"));
  expect(payloads[0].prompt).toContain("KEEP_ME");
  expect(payloads[1].prompt).toContain("KEEP_ME");
});

test("typing context by hand clears the selection notice", async ($, on) => {
  host(on);
  statusHost(on, 0);
  on("ui.selection", () => ({ value: undefined as any }));
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-selection" });
  expect(JSON.stringify(await ui.drawn())).toContain("No selection is visible");
  await ui.input({ key: "neo-context", text: "typed by hand" });
  const drawn = JSON.stringify(await ui.drawn());
  expect(drawn).not.toContain("No selection is visible");
  expect(drawn).toContain("Context attached · 13 characters");
  await ui.unmount();
});

test("Follow up appears only once there is a result to follow", async ($, on) => {
  host(on);
  statusHost(on, 0);
  const payloads: any[] = [];
  spawnHost(on, payloads);
  await $.session.start(SESSION);
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  expect(await ui.find({ key: "neo-followup" })).toBeUndefined();
  await $.command.run(command("neo-ask", "first"));
  expect(await ui.find({ key: "neo-followup" })).toBeDefined();
  await ui.unmount();
});

test("a refused Ask is reported in the pane, not the transcript", async ($, on) => {
  const logged: string[] = [];
  host(on, (t) => logged.push(t));
  statusHost(on, 0);
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-run" });
  expect(JSON.stringify(await ui.drawn())).toContain("Enter a question or target first.");
  expect(logged.join(" ")).not.toContain("Enter a question");
  await ui.unmount();
});

test("a running analysis shows elapsed time against the deadline", async ($, on) => {
  const clock = host(on);
  statusHost(on, 0);
  on("process.spawn", async function* () {
    while (true) {
      yield { stream: "stderr", text: '{"type":"planning","message":"Planning"}\n' };
      await clock.sleep(1000);
    }
  });
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  await $.command.run(command("neo-ask", "slow one"));
  await clock.advance(3000);
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  expect(JSON.stringify(await ui.drawn())).toMatch(/running · 0:0[2-3] of 4:00 · Planning/);
  await ui.unmount();
  // End the run the way the deadline does, so the fake process's sleep resolves.
  await clock.advance(241000);
});

test("Evidence is labelled rows, with raw JSON only on request", async ($, on) => {
  host(on);
  statusHost(on, 0);
  const payloads: any[] = [];
  spawnHost(on, payloads);
  await $.session.start(SESSION);
  await $.command.run(command("neo-ask", "q"));
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-evidence" });
  const drawn = JSON.stringify(await ui.drawn());
  expect(drawn).toContain("Memory · 1 retrieved · 1 included · 0 hard cited");
  expect(drawn).not.toContain('\\"memory\\":');
  await ui.press({ key: "neo-details" });
  expect(JSON.stringify(await ui.drawn())).toContain('\\"memory\\":');
  await ui.unmount();
});

test("returning to Memory reuses the store read; Refresh re-reads it", async ($, on) => {
  host(on);
  let reads = 0;
  on("process.run", () => {
    reads++;
    return {
      value: {
        exitCode: 0,
        stdout: JSON.stringify({
          schema_version: 1,
          checkout: "/work",
          fact_count: 0,
          probation_count: 0,
          facts: [],
        }),
        stderr: "",
        isStdoutTruncated: false,
        isStderrTruncated: false,
      },
    };
  });
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-memory" });
  await ui.press({ key: "neo-results" });
  await ui.press({ key: "neo-memory" });
  expect(reads).toBe(1);
  await ui.press({ key: "neo-refresh" });
  expect(reads).toBe(2);
  await ui.unmount();
});

test("the question form is drawn before the findings", async ($, on) => {
  host(on);
  statusHost(on, 0);
  const payloads: any[] = [];
  spawnHost(on, payloads);
  await $.session.start(SESSION);
  await $.command.run(command("neo-ask", "q"));
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  const drawn = JSON.stringify(await ui.drawn());
  expect(drawn.indexOf("neo-question")).toBeLessThan(drawn.indexOf("Found an edge case"));
  await ui.unmount();
});

test("pressing Enter on an empty question reports the refusal in the pane", async ($, on) => {
  host(on);
  statusHost(on, 0);
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.input({ key: "neo-question", text: "   ", kind: "submit" } as any);
  expect(JSON.stringify(await ui.drawn())).toContain("Enter a question or target first.");
  await ui.unmount();
});

test("a finished analysis makes Memory read the store again", async ($, on) => {
  host(on);
  let reads = 0;
  on("process.run", (_$, e) => {
    // Count store reads only; the Mod's hook records also use process.run.
    if (e.argv.includes("workspace")) reads++;
    return {
      value: {
        exitCode: 0,
        stdout: JSON.stringify({ schema_version: 1, checkout: "/work", fact_count: reads, probation_count: 0, facts: [] }),
        stderr: "",
        isStdoutTruncated: false,
        isStderrTruncated: false,
      },
    };
  });
  const payloads: any[] = [];
  spawnHost(on, payloads);
  await $.session.start(SESSION);
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-memory" });
  const before = reads;
  await $.command.run(command("neo-ask", "q"));
  await ui.press({ key: "neo-results" });
  await ui.press({ key: "neo-memory" });
  expect(reads).toBeGreaterThan(before);
  await ui.unmount();
});

test("details opened on one tab stay closed on the others", async ($, on) => {
  host(on);
  statusHost(on, 0);
  const payloads: any[] = [];
  spawnHost(on, payloads);
  await $.session.start(SESSION);
  await $.command.run(command("neo-ask", "q"));
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-evidence" });
  await ui.press({ key: "neo-details" });
  expect(JSON.stringify(await ui.drawn())).toContain('\\"memory\\":');
  await ui.press({ key: "neo-results" });
  expect(JSON.stringify(await ui.drawn())).toContain("Show reasoning");
  expect(JSON.stringify(await ui.drawn())).not.toContain("Hide reasoning");
  await ui.unmount();
});


test("facts about files not in this checkout are marked in Memory", async ($, on) => {
  host(on);
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: JSON.stringify({
        schema_version: 1,
        checkout: "/work",
        fact_count: 2,
        probation_count: 0,
        missing_file_count: 1,
        facts: [
          { id: "gone", subject: "pattern: src/old.py", is_valid: true, missing_file: "src/old.py" },
          { id: "live", subject: "pattern: src/new.py", is_valid: true, missing_file: "" },
        ],
      }),
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  await ui.press({ key: "neo-memory" });
  const drawn = JSON.stringify(await ui.drawn());
  expect(drawn).toContain("1 about files not in this checkout");
  expect(drawn).toContain("[missing file] pattern: src/old.py");
  expect(drawn).not.toContain("[missing file] pattern: src/new.py");
  await ui.unmount();
});

test("an asked run draws live in its command row, and Claude reads the answer", async ($, on) => {
  const clock = host(on);
  statusHost(on, 0);
  const appended: any[] = [];
  on("session.append", async (_$, e: any, next: any) => {
    appended.push(e.message);
    return next(e);
  });
  on("ui.render", { component: "CommandOutput" }, (_$, e) => _$.ui.resolve(e).Text({ children: e.props.text }));
  on("process.spawn", async function* () {
    yield { stream: "stderr", text: '{"type":"completed","message":"Done"}\n' };
    yield { stream: "stdout", text: JSON.stringify(RESULT) };
    return { value: { code: 0, signal: null } };
  });
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const answer = await $.command.run(command("neo-ask", "why?"));
  expect(answer.text).toMatch(/^Neo is on it \[[0-9a-f]{8}\]/);
  const card = await $.ui.mount({
    plugin: "neo",
    surface: "terminal",
    component: "CommandOutput",
    requestId: "row1",
    props: { command: "neo-ask", args: "why?", text: answer.text },
  } as any);
  await clock.settle();
  const shown = JSON.stringify(await card.drawn());
  expect(shown).toContain("Found an edge case");
  expect(shown).toContain("Return early for empty input");
  await card.unmount();
  const toClaude = appended.filter((m) => m.type === "user").map((m) => m.content[0].text);
  expect(toClaude[0]).toContain("Found an edge case");
});

test("the workspace is a card in the conversation, not an unasked pane", async ($, on) => {
  host(on);
  on("ui.render", { component: "CommandOutput" }, (_$, e) => _$.ui.resolve(e).Text({ children: e.props.text }));
  opens = 0;
  on("process.run", () => ({
    value: {
      exitCode: 0,
      stdout: JSON.stringify({ schema_version: 1, fact_count: 2, probation_count: 1, facts: [{ id: "a", subject: "Verify hook failures", is_valid: true }] }),
      stderr: "",
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }));
  await $.session.start({ ...SESSION, surface: "terminal", isInteractive: true });
  const answer = await $.command.run(command("neo-workspace"));
  expect(opens).toBe(0);
  const card = await $.ui.mount({
    plugin: "neo",
    surface: "terminal",
    component: "CommandOutput",
    requestId: "row2",
    props: { command: "neo-workspace", args: "", text: answer.text },
  } as any);
  expect(JSON.stringify(await card.drawn())).toContain("Verify hook failures");
  await card.press({ key: "neo-open-panel" });
  expect(opens).toBe(1);
  await card.unmount();
});

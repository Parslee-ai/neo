import type { On } from "claude-code";
import { expect, mock, test } from "claude-code/testing";
import { NeoStream, followupContext, resultText } from "../hooks/protocol.js";

const command = (command: string, args = "") => ({
  command,
  args,
  origin: { kind: "composer" } as const,
  presentation: { isFullscreen: false, columns: 80 },
});
function host(on: On) {
  mock.env(on, {});
  const clock = mock.clock(on);
  on("session.id", () => ({ value: "session1" }));
  on("session.cwd", () => ({ value: "/work" }));
  on("session.start", (_$, e) => ({ cwd: e.cwd }));
  on("command.register", (_$, e) => ({ value: { command: e.name } }));
  on("ui.log", () => ({ value: undefined }));
  on("ui.open", () => ({ value: { isPlaced: true } }));
  return clock;
}
const SESSION = { cwd: "/work", surface: null, isInteractive: false } as const;
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
  expect(answer.text).toBe("Neo: provenance opened in Memory.");
  const ui = await $.ui.mount({ plugin: "neo", surface: "terminal", ...PANE });
  const compact = JSON.stringify(await ui.drawn());
  expect(
    compact.indexOf("Selected fact") < compact.indexOf("No retrieval evidence"),
  ).toBe(true);
  await ui.press({ key: "neo-details" });
  expect(JSON.stringify(await ui.drawn())).toContain("supporting");
  await ui.unmount();
});

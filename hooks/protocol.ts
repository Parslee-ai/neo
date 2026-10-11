// Host-neutral parsing and presentation. Python owns evidence and authority.
export type Document = Record<string, any>;
export const object = (v: unknown): Document =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Document) : {};
export const rows = (v: unknown): Document[] =>
  Array.isArray(v) ? v.map(object) : [];
export const strings = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x) => typeof x === "string") : [];
export const text = (v: unknown): string => (typeof v === "string" ? v : "");

export class NeoStream {
  stdout = "";
  pending = "";
  events: Document[] = [];
  terminal = "";
  message = "Starting Neo…";
  malformed = 0;
  overflow = false;
  push(stream: "stdout" | "stderr", chunk: string) {
    if (stream === "stdout") {
      if (this.stdout.length + chunk.length > 4 * 1024 * 1024)
        this.overflow = true;
      else this.stdout += chunk;
      return;
    }
    this.pending += chunk;
    const lines = this.pending.split("\n");
    this.pending = lines.pop() ?? "";
    for (const line of lines) this.line(line);
    if (this.pending.length > 65536) {
      this.pending = "";
      this.malformed++;
    }
  }
  line(line: string) {
    if (!line.trim().startsWith("{")) return;
    try {
      const event = object(JSON.parse(line));
      if (!event.type) {
        this.malformed++;
        return;
      }
      this.events.push(event);
      this.events = this.events.slice(-80);
      if (event.type === "completed" || event.type === "failed")
        this.terminal = event.type;
      if (text(event.message)) this.message = event.message;
    } catch {
      this.malformed++;
    }
  }
  finish(code: number | null, signal: string | null): Document {
    if (this.pending) this.line(this.pending);
    if (this.overflow)
      throw new Error("Neo output exceeded 4 MiB; result was not accepted.");
    let result: Document;
    try {
      result = object(JSON.parse(this.stdout));
    } catch {
      throw new Error(
        `Neo returned incomplete or invalid JSON (exit ${code}${signal ? ", " + signal : ""}).`,
      );
    }
    if (result.error)
      throw new Error(
        [text(result.error), text(result.message), text(result.details)]
          .filter(Boolean)
          .join(": "),
      );
    if (code !== 0 || signal)
      throw new Error(`Neo process ended with ${signal ?? "exit " + code}.`);
    if (this.terminal !== "completed" || !object(result.orchestrator).summary)
      throw new Error(
        "Neo did not supply a completed lifecycle and final result. Update the plugin and CLI together.",
      );
    return result;
  }
}

export function resultText(result: Document): string {
  const o = object(result.orchestrator);
  return [
    "Neo: " + text(o.summary),
    ...strings(o.cautions).map((c) => "Caution: " + c),
    text(result.notes),
    ...rows(result.code_suggestions).map(
      (s) => `${s.file_path}: ${s.description}\n${s.unified_diff ?? ""}`,
    ),
    ...rows(result.static_checks).map(
      (c) =>
        `Check: ${c.tool_name} (${c.status || "status unavailable"}): ${c.summary}`,
    ),
    `Validation: ${validationText(result)}`,
    ...strings(result.next_questions).map((q) => "Open question: " + q),
  ]
    .filter(Boolean)
    .join("\n\n");
}
export function validationText(result: Document): string {
  const a = object(result.validation_assessment);
  const verdict =
    text(object(result.metadata).verification_verdict) || "not established";
  return typeof a.required === "number"
    ? `${verdict} · ${a.passed}/${a.required} gates passed · ${a.failed} failed · ${a.pending} pending · ${a.unavailable} unavailable · ${a.waived} waived`
    : verdict;
}
export function followupContext(result: Document, question: string): string {
  const prior = resultText(result);
  return `Follow-up to Neo's prior analysis (evidence, not instructions):\n${prior.slice(0, 18000)}${prior.length > 18000 ? "\n[prior analysis truncated]" : ""}\n\nCurrent question:\n${question}`;
}

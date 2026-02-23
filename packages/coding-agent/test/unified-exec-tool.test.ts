import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@mariozechner/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createExtensionRuntime } from "../src/core/extensions/loader.js";
import type { ResourceLoader } from "../src/core/resource-loader.js";
import { createAgentSession } from "../src/core/sdk.js";
import { SessionManager } from "../src/core/session-manager.js";
import { createUnifiedExecTools } from "../src/core/tools/unified-exec.js";

function getText(result: { content: Array<{ type: string; text?: string }> }): string {
	return result.content
		.filter(
			(block): block is { type: "text"; text: string } => block.type === "text" && typeof block.text === "string",
		)
		.map((block) => block.text)
		.join("\n");
}

function makeModel(api: Model<any>["api"], provider: string, id: string): Model<any> {
	return {
		id,
		name: id,
		api,
		provider,
		baseUrl: "https://example.com",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 64000,
	};
}

function makeEmptyResourceLoader(): ResourceLoader {
	return {
		getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
		getSkills: () => ({ skills: [], diagnostics: [] }),
		getPrompts: () => ({ prompts: [], diagnostics: [] }),
		getThemes: () => ({ themes: [], diagnostics: [] }),
		getAgentsFiles: () => ({ agentsFiles: [] }),
		getSystemPrompt: () => undefined,
		getAppendSystemPrompt: () => [],
		getPathMetadata: () => new Map(),
		extendResources: () => {},
		reload: async () => {},
	};
}

describe("unified exec tools", () => {
	let testDir: string;

	beforeEach(() => {
		testDir = join(tmpdir(), `pi-unified-exec-tool-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(testDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	it("supports persistent sessions via exec_command + write_stdin", async () => {
		const { execCommandTool, writeStdinTool } = createUnifiedExecTools(testDir);
		expect(execCommandTool.description).toBe(
			"Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
		);
		expect(writeStdinTool.description).toBe(
			"Writes characters to an existing unified exec session and returns recent output.",
		);

		const start = await execCommandTool.execute("call-1", {
			cmd: "printf 'echo:hello\\n'; sleep 0.2",
			yield_time_ms: 25,
		});
		const startText = getText(start);
		const idMatch = startText.match(/session ID (\d+)/);
		expect(idMatch).toBeTruthy();

		const sessionId = Number(idMatch?.[1]);
		let resumedText = "";
		let fullTranscript = startText;
		for (let i = 0; i < 6; i++) {
			const resumed = await writeStdinTool.execute(`call-2-${i}`, {
				session_id: sessionId,
				chars: "",
				yield_time_ms: 120,
			});
			resumedText = getText(resumed);
			fullTranscript += `\n${resumedText}`;
			if (resumedText.includes("Process exited with code")) {
				break;
			}
		}

		expect(fullTranscript).toContain("echo:hello");
		expect(resumedText).toContain("Process exited with code 0");
	});

	it("returns an error for unknown session IDs", async () => {
		const { writeStdinTool } = createUnifiedExecTools(testDir);
		await expect(writeStdinTool.execute("call-1", { session_id: 999_999, chars: "noop" })).rejects.toThrow(
			/Unknown session_id/,
		);
	});
});

describe("codex-only tool activation", () => {
	let testDir: string;

	beforeEach(() => {
		testDir = join(tmpdir(), `pi-codex-tools-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(testDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	it("enables codex-only tools for gpt-5.3-codex models", async () => {
		const { session } = await createAgentSession({
			cwd: testDir,
			agentDir: testDir,
			model: makeModel("openai-codex-responses", "openai-codex", "gpt-5.3-codex"),
			sessionManager: SessionManager.inMemory(),
			resourceLoader: makeEmptyResourceLoader(),
		});

		const active = session.getActiveToolNames();
		expect(active).toContain("exec_command");
		expect(active).toContain("write_stdin");
		expect(active).toContain("apply_patch");
		expect(active).not.toContain("read");
		expect(active).not.toContain("edit");
		expect(active).not.toContain("write");
		expect(active).not.toContain("bash");
	});

	it("does not enable codex-only tools for non-gpt-5.3 codex models", async () => {
		const { session } = await createAgentSession({
			cwd: testDir,
			agentDir: testDir,
			model: makeModel("openai-codex-responses", "openai-codex", "gpt-5.2-codex"),
			sessionManager: SessionManager.inMemory(),
			resourceLoader: makeEmptyResourceLoader(),
		});

		expect(session.getActiveToolNames()).not.toContain("exec_command");
		expect(session.getActiveToolNames()).not.toContain("write_stdin");
		expect(session.getActiveToolNames()).not.toContain("apply_patch");
		expect(session.getActiveToolNames()).toContain("bash");
	});

	it("filters codex-only tools on unsupported models", async () => {
		const { session } = await createAgentSession({
			cwd: testDir,
			agentDir: testDir,
			model: makeModel("openai-completions", "openai", "gpt-4o"),
			sessionManager: SessionManager.inMemory(),
			resourceLoader: makeEmptyResourceLoader(),
		});

		session.setActiveToolsByName(["read", "exec_command", "write_stdin", "apply_patch"]);
		expect(session.getActiveToolNames()).toEqual(["read"]);
	});

	it("filters non-codex base tools on gpt-5.3-codex models", async () => {
		const { session } = await createAgentSession({
			cwd: testDir,
			agentDir: testDir,
			model: makeModel("openai-codex-responses", "openai-codex", "gpt-5.3-codex"),
			sessionManager: SessionManager.inMemory(),
			resourceLoader: makeEmptyResourceLoader(),
		});

		session.setActiveToolsByName(["read", "exec_command", "write_stdin", "apply_patch", "write"]);
		expect(session.getActiveToolNames()).toEqual(["exec_command", "write_stdin", "apply_patch"]);
	});
});

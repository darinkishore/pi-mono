import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@mariozechner/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createExtensionRuntime } from "../src/core/extensions/loader.js";
import type { ResourceLoader } from "../src/core/resource-loader.js";
import { createAgentSession } from "../src/core/sdk.js";
import { SessionManager } from "../src/core/session-manager.js";
import { createApplyPatchTool } from "../src/core/tools/apply-patch.js";

function hasCodexCli(): boolean {
	const check = spawnSync("codex", ["--version"], { encoding: "utf8" });
	return check.status === 0;
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

describe("apply_patch tool", () => {
	let testDir: string;

	beforeEach(() => {
		testDir = join(tmpdir(), `pi-apply-patch-tool-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(testDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	it("applies file changes through the official codex apply_patch path", async () => {
		if (!hasCodexCli()) return;

		const testFile = join(testDir, "demo.txt");
		writeFileSync(testFile, "old\n", "utf8");

		const patch = ["*** Begin Patch", "*** Update File: demo.txt", "@@", "-old", "+new", "*** End Patch"].join("\n");

		const tool = createApplyPatchTool(testDir);
		expect(tool.description).toBe(
			"Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
		);
		const result = await tool.execute("call-1", patch);
		const updated = readFileSync(testFile, "utf8");

		expect(updated).toBe("new\n");
		const textBlock = result.content.find((block): block is { type: "text"; text: string } => block.type === "text");
		expect(textBlock?.text.length ?? 0).toBeGreaterThan(0);
	});
});

describe("apply_patch tool activation", () => {
	let testDir: string;

	beforeEach(() => {
		testDir = join(tmpdir(), `pi-apply-patch-activation-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(testDir, { recursive: true });
	});

	afterEach(() => {
		rmSync(testDir, { recursive: true, force: true });
	});

	it("enables apply_patch by default for codex responses models", async () => {
		const { session } = await createAgentSession({
			cwd: testDir,
			agentDir: testDir,
			model: makeModel("openai-codex-responses", "openai-codex", "gpt-5.3-codex"),
			sessionManager: SessionManager.inMemory(),
			resourceLoader: makeEmptyResourceLoader(),
		});

		expect(session.getActiveToolNames()).toContain("apply_patch");
	});

	it("does not allow apply_patch for non-codex models", async () => {
		const { session } = await createAgentSession({
			cwd: testDir,
			agentDir: testDir,
			model: makeModel("openai-completions", "openai", "gpt-4o"),
			sessionManager: SessionManager.inMemory(),
			resourceLoader: makeEmptyResourceLoader(),
		});

		session.setActiveToolsByName(["read", "apply_patch"]);
		expect(session.getActiveToolNames()).toEqual(["read"]);
	});
});

import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, Message, Model } from "@mariozechner/pi-ai";
import * as ai from "@mariozechner/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import * as compactionModule from "../src/core/compaction/index.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

function makeAssistant(model: Model<any>, text: string, totalTokens: number): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: "stop",
		timestamp: Date.now(),
		usage: {
			input: totalTokens,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	};
}

function createSession(model: Model<any>, tempDir: string, providerApiKey = "test-key"): AgentSession {
	const agent = new Agent({
		initialState: {
			model,
			systemPrompt: "Test",
			tools: [],
		},
	});
	const sessionManager = SessionManager.inMemory();
	const settingsManager = SettingsManager.create(tempDir, tempDir);
	settingsManager.applyOverrides({
		compaction: {
			enabled: true,
			keepRecentTokens: 1,
			reserveTokens: 16_384,
		},
	});
	const authStorage = new AuthStorage(join(tempDir, "auth.json"));
	authStorage.setRuntimeApiKey(model.provider, providerApiKey);
	const modelRegistry = new ModelRegistry(authStorage, tempDir);

	return new AgentSession({
		agent,
		sessionManager,
		settingsManager,
		cwd: tempDir,
		modelRegistry,
		resourceLoader: createTestResourceLoader(),
	});
}

function seedConversation(session: AgentSession, model: Model<any>): void {
	const userMessage: Message = {
		role: "user",
		content: [{ type: "text", text: "user prompt" }],
		timestamp: Date.now(),
	};
	const assistantMessage = makeAssistant(model, "assistant response", 220_000);
	session.agent.replaceMessages([userMessage, assistantMessage]);
	session.sessionManager.appendMessage(userMessage);
	session.sessionManager.appendMessage(assistantMessage);
}

describe("AgentSession Codex remote compaction", () => {
	let tempDir: string | undefined;

	afterEach(() => {
		vi.restoreAllMocks();
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
		tempDir = undefined;
	});

	it("uses remote codex compaction when model provider is openai-codex", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);

		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockResolvedValue([
			{
				role: "user",
				content: [{ type: "text", text: "Another language model started to solve this problem\n\nRemote summary" }],
				timestamp: Date.now(),
			},
			{
				role: "assistant",
				content: [{ type: "text", text: "Kept assistant context" }],
				api: model.api,
				provider: model.provider,
				model: model.id,
				stopReason: "stop",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				timestamp: Date.now(),
			},
		]);
		const localCompactSpy = vi.spyOn(compactionModule, "compact");

		const result = await session.compact();

		expect(remoteSpy).toHaveBeenCalledTimes(1);
		expect(localCompactSpy).not.toHaveBeenCalled();
		expect(result.summary).toContain("Another language model started to solve this problem");
		expect((result.details as { replacementMessages?: Message[] }).replacementMessages).toBeDefined();

		const ctx = session.sessionManager.buildSessionContext();
		expect(ctx.messages[0].role).toBe("user");
		expect((ctx.messages[0] as Message).content).toEqual([
			{ type: "text", text: "Another language model started to solve this problem\n\nRemote summary" },
		]);

		session.dispose();
	});

	it("keeps codex remote compaction even when custom instructions are provided", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);

		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockResolvedValue([
			{
				role: "user",
				content: [{ type: "text", text: "Another language model started to solve this problem\n\nRemote summary" }],
				timestamp: Date.now(),
			},
		]);
		const localCompactSpy = vi.spyOn(compactionModule, "compact");

		const result = await session.compact("focus on unresolved TODOs");

		expect(remoteSpy).toHaveBeenCalledTimes(1);
		expect(localCompactSpy).not.toHaveBeenCalled();
		expect(result.summary).toContain("Another language model started to solve this problem");

		session.dispose();
	});

	it("trims old context and retries codex remote compaction when /compact overflows", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);

		const seenMessageCounts: number[] = [];
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (...args) => {
			seenMessageCounts.push(args[1].messages.length);
			if (seenMessageCounts.length === 1) {
				throw new Error(
					"Your input exceeds the context window of this model. Please adjust your input and try again.",
				);
			}
			return [
				{
					role: "user",
					content: [
						{ type: "text", text: "Another language model started to solve this problem\n\nRecovered summary" },
					],
					timestamp: Date.now(),
				},
			];
		});

		const result = await session.compact();

		expect(remoteSpy).toHaveBeenCalledTimes(2);
		expect(seenMessageCounts.length).toBe(2);
		const secondCount = seenMessageCounts[1];
		if (secondCount === undefined) {
			throw new Error("Expected a second compaction attempt with trimmed context");
		}
		expect(seenMessageCounts[0]).toBeGreaterThan(secondCount);
		expect(result.summary).toContain("Recovered summary");

		session.dispose();
	});
});

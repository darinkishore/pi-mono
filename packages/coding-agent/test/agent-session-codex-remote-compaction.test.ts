import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, Message, Model } from "@mariozechner/pi-ai";
import * as ai from "@mariozechner/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentSession, type AgentSessionEvent } from "../src/core/agent-session.js";
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

function seedLargeAssistantConversation(session: AgentSession, model: Model<any>, assistantCount: number): void {
	const messages: Message[] = [
		{
			role: "user",
			content: [{ type: "text", text: "large user prompt" }],
			timestamp: Date.now(),
		},
	];
	for (let index = 0; index < assistantCount; index++) {
		// Mirror real provider usage behavior where later assistant turns report cumulative context usage.
		messages.push(makeAssistant(model, `assistant context ${index}`, (index + 1) * 20_000));
	}
	session.agent.replaceMessages(messages);
	for (const message of messages) {
		session.sessionManager.appendMessage(message);
	}
}

function makeToolCallAssistant(model: Model<any>, callIds: string[]): AssistantMessage {
	return {
		role: "assistant",
		content: callIds.map((callId) => ({
			type: "toolCall" as const,
			id: callId,
			name: `tool-${callId}`,
			arguments: { callId },
		})),
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: "toolUse",
		timestamp: Date.now(),
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	};
}

function makeToolResult(toolCallId: string): Message {
	return {
		role: "toolResult",
		toolCallId,
		toolName: `tool-${toolCallId}`,
		content: [{ type: "text", text: `result-${toolCallId}` }],
		isError: false,
		timestamp: Date.now(),
	};
}

function expectNoOrphanToolResults(messages: Message[]): void {
	const toolCallIds = new Set<string>();
	for (const message of messages) {
		if (message.role !== "assistant") {
			continue;
		}
		for (const block of message.content) {
			if (block.type === "toolCall") {
				toolCallIds.add(block.id);
			}
		}
	}

	const orphanToolResultIds = messages
		.filter((message): message is Extract<Message, { role: "toolResult" }> => message.role === "toolResult")
		.filter((message) => !toolCallIds.has(message.toolCallId))
		.map((message) => message.toolCallId);
	expect(orphanToolResultIds).toEqual([]);
}

function expectNoUnpairedToolCalls(messages: Message[]): void {
	const toolCallIds = new Set<string>();
	for (const message of messages) {
		if (message.role !== "assistant") {
			continue;
		}
		for (const block of message.content) {
			if (block.type === "toolCall") {
				toolCallIds.add(block.id);
			}
		}
	}
	const toolResultIds = new Set(
		messages
			.filter((message): message is Extract<Message, { role: "toolResult" }> => message.role === "toolResult")
			.map((message) => message.toolCallId),
	);
	const unpairedCallIds = [...toolCallIds].filter((id) => !toolResultIds.has(id));
	expect(unpairedCallIds).toEqual([]);
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

	it("pre-trims old context locally before codex remote compaction when /compact would overflow", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedLargeAssistantConversation(session, model, 24);

		const seenMessageCounts: number[] = [];
		const overflowError = "Your input exceeds the context window of this model. Please adjust your input and try again.";
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (_model, context) => {
			seenMessageCounts.push(context.messages.length);
			if (context.messages.length > 18) {
				throw new Error(overflowError);
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

		expect(remoteSpy).toHaveBeenCalledTimes(1);
		const firstCount = seenMessageCounts[0];
		if (firstCount === undefined) {
			throw new Error("Expected one compaction attempt");
		}
		expect(firstCount).toBeLessThanOrEqual(18);
		expect(firstCount).toBeLessThan(25);
		expect(result.summary).toContain("Recovered summary");

		session.dispose();
	});

	it("keeps compaction API retries low for large contexts that require heavy trimming", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedLargeAssistantConversation(session, model, 24);

		const seenMessageCounts: number[] = [];
		const overflowError = "Your input exceeds the context window of this model. Please adjust your input and try again.";
		let retryThreshold: number | undefined;
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (_model, context) => {
			seenMessageCounts.push(context.messages.length);
			if (retryThreshold === undefined) {
				retryThreshold = context.messages.length - 1;
			}
			if (context.messages.length > retryThreshold) {
				throw new Error(overflowError);
			}
			return [
				{
					role: "user",
					content: [{ type: "text", text: "Another language model started to solve this problem\n\nRecovered summary" }],
					timestamp: Date.now(),
				},
			];
		});

		await session.compact();

		expect(remoteSpy.mock.calls.length).toBeLessThanOrEqual(3);
		expect(remoteSpy.mock.calls.length).toBeGreaterThanOrEqual(2);
		const firstCount = seenMessageCounts[0];
		const secondCount = seenMessageCounts[1];
		if (firstCount === undefined || secondCount === undefined) {
			throw new Error("Expected overflow retries with locally trimmed contexts");
		}
		expect(firstCount).toBeGreaterThan(1);
		expect(firstCount).toBeGreaterThan(secondCount);
		const finalCount = seenMessageCounts[seenMessageCounts.length - 1];
		if (finalCount === undefined) {
			throw new Error("Expected final compaction retry count");
		}
		expect(finalCount).toBeLessThan(firstCount);
		for (let index = 1; index < seenMessageCounts.length; index++) {
			const previous = seenMessageCounts[index - 1];
			const current = seenMessageCounts[index];
			if (previous === undefined || current === undefined) {
				throw new Error("Expected compact retry count values");
			}
			expect(previous).toBeGreaterThan(current);
		}

		session.dispose();
	});

	it("trims trailing assistant tool call and toolResult together without leaving orphans", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const messages: Message[] = [
			{
				role: "user",
				content: [{ type: "text", text: "tool-heavy context" }],
				timestamp: Date.now(),
			},
			makeToolCallAssistant(model, ["call-a"]),
			makeToolResult("call-a"),
			makeToolCallAssistant(model, ["call-b"]),
			makeToolResult("call-b"),
		];
		session.agent.replaceMessages(messages);
		for (const message of messages) {
			session.sessionManager.appendMessage(message);
		}

		const attemptedMessages: Message[][] = [];
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (_model, context) => {
			attemptedMessages.push([...context.messages]);
			if (attemptedMessages.length === 1) {
				throw new Error(
					"Your input exceeds the context window of this model. Please adjust your input and try again.",
				);
			}
			return [
				{
					role: "user",
					content: [{ type: "text", text: "Another language model started to solve this problem\n\nRecovered summary" }],
					timestamp: Date.now(),
				},
			];
		});

		await session.compact();

		expect(remoteSpy).toHaveBeenCalledTimes(2);
		const retryMessages = attemptedMessages[1];
		if (!retryMessages) {
			throw new Error("Expected retry attempt messages");
		}
		expect(retryMessages.some((message) => message.role === "toolResult" && message.toolCallId === "call-b")).toBe(
			false,
		);
		expect(
			retryMessages.some(
				(message) =>
					message.role === "assistant" &&
					message.content.some((block) => block.type === "toolCall" && block.id === "call-b"),
			),
		).toBe(false);
		expect(retryMessages.some((message) => message.role === "toolResult" && message.toolCallId === "call-a")).toBe(
			true,
		);
		expect(
			retryMessages.some(
				(message) =>
					message.role === "assistant" &&
					message.content.some((block) => block.type === "toolCall" && block.id === "call-a"),
			),
		).toBe(true);
		expectNoOrphanToolResults(retryMessages);

		session.dispose();
	});

	it("trims one toolResult from a multi-tool-call assistant and preserves remaining pairs", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const messages: Message[] = [
			{
				role: "user",
				content: [{ type: "text", text: "tool-heavy context" }],
				timestamp: Date.now(),
			},
			makeToolCallAssistant(model, ["call-1", "call-2", "call-3"]),
			makeToolResult("call-1"),
			makeToolResult("call-2"),
			makeToolResult("call-3"),
		];
		session.agent.replaceMessages(messages);
		for (const message of messages) {
			session.sessionManager.appendMessage(message);
		}

		const attemptedMessages: Message[][] = [];
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (_model, context) => {
			attemptedMessages.push([...context.messages]);
			if (attemptedMessages.length === 1) {
				throw new Error(
					"Your input exceeds the context window of this model. Please adjust your input and try again.",
				);
			}
			return [
				{
					role: "user",
					content: [{ type: "text", text: "Another language model started to solve this problem\n\nRecovered summary" }],
					timestamp: Date.now(),
				},
			];
		});

		await session.compact();

		expect(remoteSpy).toHaveBeenCalledTimes(2);
		const compactMessages = attemptedMessages[1];
		if (!compactMessages) {
			throw new Error("Expected retry compaction attempt messages");
		}
		const retryAssistant = compactMessages.find((message) => message.role === "assistant");
		if (!retryAssistant || retryAssistant.role !== "assistant") {
			throw new Error("Expected assistant message in retry compaction attempt");
		}
		const retryCallIds: string[] = [];
		for (const block of retryAssistant.content) {
			if (block.type === "toolCall") {
				retryCallIds.push(block.id);
			}
		}
		expect(retryCallIds).toEqual(["call-1", "call-2"]);
		expect(compactMessages.some((message) => message.role === "toolResult" && message.toolCallId === "call-1")).toBe(
			true,
		);
		expect(compactMessages.some((message) => message.role === "toolResult" && message.toolCallId === "call-2")).toBe(
			true,
		);
		expect(compactMessages.some((message) => message.role === "toolResult" && message.toolCallId === "call-3")).toBe(
			false,
		);
		expectNoOrphanToolResults(compactMessages);

		session.dispose();
	});

	it("synthesizes aborted toolResult stubs for remaining tool calls without outputs", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const orphanedCallId = "call-orphan";
		const messages: Message[] = [
			{
				role: "user",
				content: [{ type: "text", text: "tool-heavy context" }],
				timestamp: Date.now(),
			},
			makeToolCallAssistant(model, [orphanedCallId]),
			makeToolCallAssistant(model, ["call-trimmed"]),
			makeToolResult("call-trimmed"),
		];
		session.agent.replaceMessages(messages);
		for (const message of messages) {
			session.sessionManager.appendMessage(message);
		}

		const attemptedMessages: Message[][] = [];
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (_model, context) => {
			attemptedMessages.push([...context.messages]);
			return [
				{
					role: "user",
					content: [{ type: "text", text: "Another language model started to solve this problem\n\nRecovered summary" }],
					timestamp: Date.now(),
				},
			];
		});

		await session.compact();

		expect(remoteSpy).toHaveBeenCalledTimes(1);
		const compactMessages = attemptedMessages[0];
		if (!compactMessages) {
			throw new Error("Expected compaction attempt messages");
		}
		const assistantIndex = compactMessages.findIndex(
			(message) =>
				message.role === "assistant" &&
				message.content.some((block) => block.type === "toolCall" && block.id === orphanedCallId),
		);
		expect(assistantIndex).toBeGreaterThanOrEqual(0);
		const syntheticResult = compactMessages[assistantIndex + 1];
		expect(syntheticResult?.role).toBe("toolResult");
		if (!syntheticResult || syntheticResult.role !== "toolResult") {
			throw new Error("Expected synthetic toolResult after assistant tool call");
		}
		expect(syntheticResult.toolCallId).toBe(orphanedCallId);
		expect(syntheticResult.content).toEqual([{ type: "text", text: "aborted" }]);
		expect(syntheticResult.isError).toBe(true);
		expectNoUnpairedToolCalls(compactMessages);
		expectNoOrphanToolResults(compactMessages);

		session.dispose();
	});

	it("stops trimming at the user boundary and throws after no further shrink is possible", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);

		const seenMessageCounts: number[] = [];
		const overflowError = "Your input exceeds the context window of this model. Please adjust your input and try again.";
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses").mockImplementation(async (_model, context) => {
			seenMessageCounts.push(context.messages.length);
			throw new Error(overflowError);
		});

		await expect(session.compact()).rejects.toThrow(`Codex remote compaction failed: ${overflowError}`);
		expect(remoteSpy).toHaveBeenCalledTimes(2);
		expect(seenMessageCounts).toEqual([2, 1]);

		session.dispose();
	});

	it("throws when codex remote compaction fails without local fallback", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);

		const remoteSpy = vi
			.spyOn(ai, "compactOpenAICodexResponses")
			.mockRejectedValue(new Error("Codex compaction request failed"));
		const localCompactSpy = vi.spyOn(compactionModule, "compact");

		await expect(session.compact()).rejects.toThrow(
			"Codex remote compaction failed: Codex compaction request failed",
		);

		expect(remoteSpy).toHaveBeenCalledTimes(1);
		expect(localCompactSpy).not.toHaveBeenCalled();

		session.dispose();
	});

	it("uses local compaction path for non-codex models", async () => {
		tempDir = join(tmpdir(), `pi-non-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("anthropic", "claude-sonnet-4-5")!;
		const session = createSession(model, tempDir, "anthropic-key");

		const localResult = {
			summary: "Local summary",
			firstKeptEntryId: "entry-1",
			tokensBefore: 123,
			details: { readFiles: [], modifiedFiles: [] },
		};
		const localCompactSpy = vi.spyOn(compactionModule, "compact").mockResolvedValue(localResult);
		const remoteSpy = vi.spyOn(ai, "compactOpenAICodexResponses");

		const result = await (
			session as unknown as {
				_runCompactionWithCodexFallback: (
					preparation: unknown,
					apiKey: string,
					customInstructions: string | undefined,
					signal?: AbortSignal,
				) => Promise<typeof localResult>;
			}
		)._runCompactionWithCodexFallback({} as unknown, "anthropic-key", undefined);

		expect(localCompactSpy).toHaveBeenCalledTimes(1);
		expect(remoteSpy).not.toHaveBeenCalled();
		expect(result).toEqual(localResult);

		session.dispose();
	});

	it("emits auto-compaction error when codex remote auto-compaction fails without local fallback", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);

		const events: AgentSessionEvent[] = [];
		const unsubscribe = session.subscribe((event) => events.push(event));

		const remoteSpy = vi
			.spyOn(ai, "compactOpenAICodexResponses")
			.mockRejectedValue(new Error("Too many requests, please wait before trying again."));
		const localCompactSpy = vi.spyOn(compactionModule, "compact");

		const outcome = await (
			session as unknown as {
				_runAutoCompaction: (
					reason: "overflow" | "threshold",
					willRetry: boolean,
				) => Promise<{ ok: boolean; errorMessage?: string }>;
			}
		)._runAutoCompaction("threshold", false);

		const autoCompactionEnd = events.find((event) => event.type === "auto_compaction_end");
		expect(remoteSpy).toHaveBeenCalledTimes(1);
		expect(localCompactSpy).not.toHaveBeenCalled();
		expect(outcome.ok).toBe(false);
		expect(outcome.errorMessage).toContain("Auto-compaction failed:");
		expect(outcome.errorMessage).toContain("Codex remote compaction failed:");
		expect(outcome.errorMessage).toContain("Too many requests, please wait before trying again.");
		expect(autoCompactionEnd?.type).toBe("auto_compaction_end");
		if (autoCompactionEnd?.type !== "auto_compaction_end") {
			throw new Error("expected auto_compaction_end event");
		}
		expect(autoCompactionEnd.errorMessage).toContain("Auto-compaction failed:");
		expect(autoCompactionEnd.errorMessage).toContain("Codex remote compaction failed:");
		expect(autoCompactionEnd.result).toBeUndefined();

		unsubscribe();
		session.dispose();
	});

	it("fails auto-compaction when compaction succeeds but context still exceeds threshold", async () => {
		tempDir = join(tmpdir(), `pi-codex-compact-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		seedConversation(session, model);
		const branchHead = session.sessionManager.getBranch()[0];
		if (!branchHead) {
			throw new Error("expected seeded conversation entry");
		}

		const threshold = (
			session as unknown as {
				_getAutoCompactLimit: (contextWindow: number, settings: { reserveTokens: number }) => number;
			}
		)._getAutoCompactLimit(model.contextWindow ?? 0, { reserveTokens: 16_384 });
		const targetTokens = Number.isFinite(threshold) ? Math.floor(threshold) + 1_000 : 300_000;
		const oversizedAssistant = {
			role: "assistant" as const,
			content: [{ type: "text" as const, text: "x".repeat(targetTokens * 4) }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			stopReason: "stop" as const,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: Date.now(),
		};

		const fallbackSpy = vi
			.spyOn(
				session as unknown as {
					_runCompactionWithCodexFallback: (
						preparation: unknown,
						apiKey: string,
						customInstructions: string | undefined,
						signal?: AbortSignal,
					) => Promise<{
						summary: string;
						firstKeptEntryId: string;
						tokensBefore: number;
						details: unknown;
					}>;
				},
				"_runCompactionWithCodexFallback",
			)
			.mockResolvedValue({
				summary: "remote summary",
				firstKeptEntryId: branchHead.id,
				tokensBefore: 220_000,
				details: {
					readFiles: [],
					modifiedFiles: [],
					replacementMessages: [oversizedAssistant],
				},
			});

		const events: AgentSessionEvent[] = [];
		const unsubscribe = session.subscribe((event) => events.push(event));

		const outcome = await (
			session as unknown as {
				_runAutoCompaction: (
					reason: "overflow" | "threshold",
					willRetry: boolean,
				) => Promise<{ ok: boolean; errorMessage?: string }>;
			}
		)._runAutoCompaction("threshold", false);

		expect(fallbackSpy).toHaveBeenCalledTimes(1);
		expect(outcome.ok).toBe(false);
		expect(outcome.errorMessage).toContain("Auto-compaction failed:");
		expect(outcome.errorMessage).toContain("Compaction succeeded but context still exceeds threshold");

		const autoCompactionEnd = events.find((event) => event.type === "auto_compaction_end");
		expect(autoCompactionEnd?.type).toBe("auto_compaction_end");
		if (autoCompactionEnd?.type !== "auto_compaction_end") {
			throw new Error("expected auto_compaction_end event");
		}
		expect(autoCompactionEnd.errorMessage).toContain("Compaction succeeded but context still exceeds threshold");
		expect(autoCompactionEnd.result).toBeUndefined();

		unsubscribe();
		session.dispose();
	});
});

import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, Message, Model } from "@mariozechner/pi-ai";
import * as ai from "@mariozechner/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSession, type AgentSessionEvent } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import * as compactionModule from "../src/core/compaction/index.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

function makeAssistant(
	model: Model<any>,
	text: string,
	totalTokens: number,
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason,
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

function makeOverflowError(model: Model<any>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: "error",
		errorMessage:
			'Codex error: {"type":"error","error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again.","param":"input"},"sequence_number":2}',
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

describe("AgentSession Codex compaction double failure", () => {
	let tempDir: string | undefined;

	beforeEach(() => {
		vi.useFakeTimers();
	});

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
		tempDir = undefined;
	});

	it("leaves codex session unrecovered when remote auto-compaction fails", async () => {
		tempDir = join(tmpdir(), `pi-codex-double-failure-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const userMessage: Message = {
			role: "user",
			content: [{ type: "text", text: "large prompt context" }],
			timestamp: Date.now(),
		};
		const largeAssistant = makeAssistant(model, "assistant with huge context", 250_000);
		const overflowAssistant = makeOverflowError(model);

		session.agent.replaceMessages([userMessage, largeAssistant, overflowAssistant]);
		session.sessionManager.appendMessage(userMessage);
		session.sessionManager.appendMessage(largeAssistant);
		session.sessionManager.appendMessage(overflowAssistant);

		const events: AgentSessionEvent[] = [];
		const unsubscribe = session.subscribe((event) => events.push(event));

		const remoteSpy = vi
			.spyOn(ai, "compactOpenAICodexResponses")
			.mockRejectedValue(new Error("Too many requests, please wait before trying again."));
		const localCompactSpy = vi.spyOn(compactionModule, "compact");
		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(overflowAssistant, true, "postTurn");

		await vi.runAllTimersAsync();

		expect(remoteSpy).toHaveBeenCalledTimes(1);
		expect(localCompactSpy).not.toHaveBeenCalled();
		expect(continueSpy).not.toHaveBeenCalled();

		const starts = events.filter((event) => event.type === "auto_compaction_start");
		expect(starts).toHaveLength(1);
		if (starts[0]?.type !== "auto_compaction_start") {
			throw new Error("expected auto_compaction_start");
		}
		expect(starts[0].reason).toBe("overflow");

		const ends = events.filter((event) => event.type === "auto_compaction_end");
		expect(ends).toHaveLength(1);
		if (ends[0]?.type !== "auto_compaction_end") {
			throw new Error("expected auto_compaction_end");
		}
		expect(ends[0].aborted).toBe(false);
		expect(ends[0].willRetry).toBe(false);
		expect(ends[0].result).toBeUndefined();
		expect(ends[0].errorMessage).toContain("Context overflow recovery failed:");
		expect(ends[0].errorMessage).toContain("Codex remote compaction failed:");
		expect(ends[0].errorMessage).toContain("Too many requests, please wait before trying again.");

		const compactionEntries = session.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
		expect(compactionEntries).toHaveLength(0);

		const remainingMessages = session.agent.state.messages;
		expect(remainingMessages).toHaveLength(2);
		const lastMessage = remainingMessages[remainingMessages.length - 1];
		expect(lastMessage?.role).toBe("assistant");
		if (!lastMessage || lastMessage.role !== "assistant") {
			throw new Error("expected trailing assistant message");
		}
		expect(lastMessage.usage.totalTokens).toBe(250_000);
		expect(lastMessage.content).toEqual([{ type: "text", text: "assistant with huge context" }]);

		unsubscribe();
		session.dispose();
	});

	it("can fail threshold compaction first, then fail overflow recovery, leaving no retry path", async () => {
		tempDir = join(tmpdir(), `pi-codex-threshold-overflow-double-failure-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const userMessage: Message = {
			role: "user",
			content: [{ type: "text", text: "large prompt context" }],
			timestamp: Date.now(),
		};
		const largeAssistant = makeAssistant(model, "assistant with huge context", 250_000, "toolUse");
		const overflowAssistant = makeOverflowError(model);

		session.agent.replaceMessages([userMessage, largeAssistant]);
		session.sessionManager.appendMessage(userMessage);
		session.sessionManager.appendMessage(largeAssistant);

		const events: AgentSessionEvent[] = [];
		const unsubscribe = session.subscribe((event) => events.push(event));

		const remoteSpy = vi
			.spyOn(ai, "compactOpenAICodexResponses")
			.mockRejectedValue(new Error("Too many requests, please wait before trying again."));
		const localCompactSpy = vi.spyOn(compactionModule, "compact");
		const abortSpy = vi.spyOn(session.agent, "abort");
		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
					hasImmediateContinuation?: boolean,
				) => Promise<void>;
			}
		)._checkCompaction(largeAssistant, true, "postTurn", true);

		session.agent.replaceMessages([...session.agent.state.messages, overflowAssistant]);
		session.sessionManager.appendMessage(overflowAssistant);

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(overflowAssistant, true, "postTurn");

		await vi.runAllTimersAsync();

		expect(remoteSpy).toHaveBeenCalledTimes(2);
		expect(localCompactSpy).not.toHaveBeenCalled();
		expect(abortSpy).toHaveBeenCalledTimes(1);
		expect(continueSpy).not.toHaveBeenCalled();

		const starts = events.filter((event) => event.type === "auto_compaction_start");
		expect(starts).toHaveLength(2);
		if (starts[0]?.type !== "auto_compaction_start" || starts[1]?.type !== "auto_compaction_start") {
			throw new Error("expected two auto_compaction_start events");
		}
		expect(starts[0].reason).toBe("threshold");
		expect(starts[1].reason).toBe("overflow");

		const ends = events.filter((event) => event.type === "auto_compaction_end");
		expect(ends).toHaveLength(2);
		if (ends[0]?.type !== "auto_compaction_end" || ends[1]?.type !== "auto_compaction_end") {
			throw new Error("expected two auto_compaction_end events");
		}
		expect(ends[0].willRetry).toBe(false);
		expect(ends[0].result).toBeUndefined();
		expect(ends[0].errorMessage).toContain("Auto-compaction failed:");
		expect(ends[0].errorMessage).toContain("Codex remote compaction failed:");
		expect(ends[1].willRetry).toBe(false);
		expect(ends[1].result).toBeUndefined();
		expect(ends[1].errorMessage).toContain("Context overflow recovery failed:");
		expect(ends[1].errorMessage).toContain("Codex remote compaction failed:");

		const compactionEntries = session.sessionManager.getEntries().filter((entry) => entry.type === "compaction");
		expect(compactionEntries).toHaveLength(0);

		unsubscribe();
		session.dispose();
	});

	it("skips overlapping auto-compaction and avoids undefined signal crashes", async () => {
		tempDir = join(tmpdir(), `pi-codex-compaction-signal-race-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const userMessage: Message = {
			role: "user",
			content: [{ type: "text", text: "large prompt context" }],
			timestamp: Date.now(),
		};
		const largeAssistant = makeAssistant(model, "assistant with huge context", 250_000);

		session.agent.replaceMessages([userMessage, largeAssistant]);
		session.sessionManager.appendMessage(userMessage);
		session.sessionManager.appendMessage(largeAssistant);

		const firstEntry = session.sessionManager.getBranch()[0];
		if (!firstEntry) {
			throw new Error("expected seeded session entry");
		}

		const events: AgentSessionEvent[] = [];
		const unsubscribe = session.subscribe((event) => events.push(event));

		const pendingResolvers: Array<
			(value: {
				summary: string;
				firstKeptEntryId: string;
				tokensBefore: number;
				details: { readFiles: string[]; modifiedFiles: string[] };
			}) => void
		> = [];
		const fallbackSpy = vi
			.spyOn(
				session as unknown as {
					_runCompactionWithCodexFallback: (...args: unknown[]) => Promise<{
						summary: string;
						firstKeptEntryId: string;
						tokensBefore: number;
						details: { readFiles: string[]; modifiedFiles: string[] };
					}>;
				},
				"_runCompactionWithCodexFallback",
			)
			.mockImplementation(
				async () =>
					await new Promise((resolve) => {
						pendingResolvers.push(resolve);
					}),
			);

		const runAutoCompaction = (
			session as unknown as {
				_runAutoCompaction: (
					reason: "overflow" | "threshold",
					willRetry: boolean,
				) => Promise<{ ok: boolean; errorMessage?: string }>;
			}
		)._runAutoCompaction.bind(session);

		const runA = runAutoCompaction("threshold", false);

		for (let i = 0; i < 20 && pendingResolvers.length < 1; i++) {
			await Promise.resolve();
		}
		expect(pendingResolvers).toHaveLength(1);

		const runBOutcome = await runAutoCompaction("threshold", false);
		expect(runBOutcome.ok).toBe(true);
		expect(pendingResolvers).toHaveLength(1);

		const compactionResult = {
			summary: "compacted",
			firstKeptEntryId: firstEntry.id,
			tokensBefore: 250_000,
			details: { readFiles: [], modifiedFiles: [] },
		};

		const firstResolver = pendingResolvers[0];
		if (!firstResolver) {
			throw new Error("expected pending compaction resolver");
		}

		firstResolver(compactionResult);
		const runAOutcome = await runA;
		expect(runAOutcome.ok).toBe(true);

		expect(fallbackSpy).toHaveBeenCalledTimes(1);

		const ends = events.filter((event) => event.type === "auto_compaction_end");
		expect(ends).toHaveLength(1);
		if (ends[0]?.type !== "auto_compaction_end") {
			throw new Error("expected auto_compaction_end event");
		}
		expect(ends[0].errorMessage).toBeUndefined();
		expect(ends[0].result?.summary).toBe("compacted");

		const starts = events.filter((event) => event.type === "auto_compaction_start");
		expect(starts).toHaveLength(1);

		unsubscribe();
		session.dispose();
	});

	it("passes through codex remote abort errors without wrapping", async () => {
		tempDir = join(tmpdir(), `pi-codex-abort-pass-through-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const runCompactionWithFallback = (
			session as unknown as {
				_runCompactionWithCodexFallback: (
					preparation: unknown,
					apiKey: string,
					customInstructions: string | undefined,
					signal?: AbortSignal,
				) => Promise<unknown>;
			}
		)._runCompactionWithCodexFallback.bind(session);

		const remoteSpy = vi.spyOn(
			session as unknown as {
				_runCodexRemoteCompaction: (preparation: unknown, apiKey: string, signal?: AbortSignal) => Promise<unknown>;
			},
			"_runCodexRemoteCompaction",
		);

		const signalAbortedError = Object.assign(new Error("request was aborted"), { name: "AbortError" });
		remoteSpy.mockRejectedValueOnce(signalAbortedError);
		const abortedController = new AbortController();
		abortedController.abort();

		await expect(
			runCompactionWithFallback({} as unknown, "codex-key", undefined, abortedController.signal),
		).rejects.toBe(signalAbortedError);

		const abortLikeError = new Error("Request was aborted by caller");
		remoteSpy.mockRejectedValueOnce(abortLikeError);

		await expect(runCompactionWithFallback({} as unknown, "codex-key", undefined)).rejects.toBe(abortLikeError);

		session.dispose();
	});

	it("emits aborted auto_compaction_end for codex remote abort errors", async () => {
		tempDir = join(tmpdir(), `pi-codex-auto-abort-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = ai.getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");

		const userMessage: Message = {
			role: "user",
			content: [{ type: "text", text: "large prompt context" }],
			timestamp: Date.now(),
		};
		const largeAssistant = makeAssistant(model, "assistant with huge context", 250_000);
		session.agent.replaceMessages([userMessage, largeAssistant]);
		session.sessionManager.appendMessage(userMessage);
		session.sessionManager.appendMessage(largeAssistant);

		const events: AgentSessionEvent[] = [];
		const unsubscribe = session.subscribe((event) => events.push(event));

		const runAutoCompaction = (
			session as unknown as {
				_runAutoCompaction: (
					reason: "overflow" | "threshold",
					willRetry: boolean,
				) => Promise<{ ok: boolean; errorMessage?: string }>;
			}
		)._runAutoCompaction.bind(session);

		const remoteAbortError = Object.assign(new Error("request was aborted"), { name: "AbortError" });
		const fallbackSpy = vi
			.spyOn(
				session as unknown as {
					_runCompactionWithCodexFallback: (
						preparation: unknown,
						apiKey: string,
						customInstructions: string | undefined,
						signal?: AbortSignal,
					) => Promise<unknown>;
				},
				"_runCompactionWithCodexFallback",
			)
			.mockRejectedValue(remoteAbortError);

		const outcome = await runAutoCompaction("threshold", false);

		expect(fallbackSpy).toHaveBeenCalledTimes(1);
		expect(outcome.ok).toBe(true);

		const starts = events.filter((event) => event.type === "auto_compaction_start");
		expect(starts).toHaveLength(1);
		const ends = events.filter((event) => event.type === "auto_compaction_end");
		expect(ends).toHaveLength(1);
		if (ends[0]?.type !== "auto_compaction_end") {
			throw new Error("expected auto_compaction_end event");
		}
		expect(ends[0].aborted).toBe(true);
		expect(ends[0].errorMessage).toBeUndefined();
		expect(ends[0].result).toBeUndefined();

		unsubscribe();
		session.dispose();
	});
});

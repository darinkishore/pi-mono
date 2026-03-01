import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, Model } from "@mariozechner/pi-ai";
import { getModel } from "@mariozechner/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

function makeAssistant(model: Model<any>, totalTokens: number, timestamp = Date.now(), text = "ok"): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		stopReason: "stop",
		timestamp,
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

function createSession(
	model: Model<any>,
	tempDir: string,
	providerApiKey = "test-key",
	compactionEnabled = true,
): AgentSession {
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
			enabled: compactionEnabled,
			reserveTokens: 16_384,
			keepRecentTokens: 20_000,
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

const AUTO_COMPACTION_OK = { ok: true } as const;

describe("AgentSession compaction semantics", () => {
	let tempDir: string | undefined;

	afterEach(() => {
		vi.restoreAllMocks();
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
		tempDir = undefined;
	});

	it("defers threshold compaction until pre-prompt when there is no queued work", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const session = createSession(model, tempDir, "anthropic-key");
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(model, 186_000);

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
				_pendingThresholdCompaction: boolean;
			}
		)._checkCompaction(assistant, true, "postTurn");

		expect(runAutoCompaction).not.toHaveBeenCalled();
		expect(
			(
				session as unknown as {
					_pendingThresholdCompaction: boolean;
				}
			)._pendingThresholdCompaction,
		).toBe(true);

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(assistant, false, "prePrompt");

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("threshold", false);
		expect(
			(
				session as unknown as {
					_pendingThresholdCompaction: boolean;
				}
			)._pendingThresholdCompaction,
		).toBe(false);

		session.dispose();
	});

	it("runs threshold compaction at post-turn when tool-use implies immediate continuation", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(model, 250_000);

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
					hasImmediateContinuation?: boolean,
				) => Promise<void>;
				_pendingThresholdCompaction: boolean;
			}
		)._checkCompaction(assistant, true, "postTurn", true);

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("threshold", true);
		expect(
			(
				session as unknown as {
					_pendingThresholdCompaction: boolean;
				}
			)._pendingThresholdCompaction,
		).toBe(false);

		session.dispose();
	});

	it("applies Codex 90% auto-compaction limit for gpt-5.3", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const codexModel = getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(codexModel, tempDir, "codex-key");
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(codexModel, 250_000);
		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(assistant, false, "prePrompt");

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("threshold", false);
		session.dispose();
	});

	it("applies Codex 90% auto-compaction limit for gpt-5.3-codex-spark", () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const sparkModel = getModel("openai-codex", "gpt-5.3-codex-spark")!;
		const session = createSession(sparkModel, tempDir, "codex-key");

		const limit = (
			session as unknown as {
				_getAutoCompactLimit: (contextWindow: number, settings: { reserveTokens: number }) => number;
			}
		)._getAutoCompactLimit(128_000, { reserveTokens: 16_384 });

		// 90% of Spark's 128k context window; reserveTokens should not lower Codex limit.
		expect(limit).toBe(115_200);
		session.dispose();
	});

	it("uses reserve-tokens threshold for non-codex providers", () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const anthropicModel = getModel("anthropic", "claude-sonnet-4-5")!;
		const session = createSession(anthropicModel, tempDir, "anthropic-key");

		const limit = (
			session as unknown as {
				_getAutoCompactLimit: (contextWindow: number, settings: { reserveTokens: number }) => number;
			}
		)._getAutoCompactLimit(272_000, { reserveTokens: 16_384 });

		expect(limit).toBe(255_616);
		session.dispose();
	});

	it("still runs overflow compaction for Codex when global auto-compaction is disabled", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const codexModel = getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(codexModel, tempDir, "codex-key", false);
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(codexModel, 0);
		assistant.stopReason = "error";
		assistant.errorMessage = 'Codex error: {"error":{"code":"context_length_exceeded"}}';

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(assistant, true, "postTurn");

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("overflow", true);
		session.dispose();
	});

	it("falls back to overflow compaction when native Anthropic compaction hits context overflow", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const opusModel = getModel("anthropic", "claude-opus-4-6")!;
		const session = createSession(opusModel, tempDir, "anthropic-key", true);
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(opusModel, 0);
		assistant.stopReason = "error";
		assistant.errorMessage = "prompt is too long: 213462 tokens > 200000 maximum";

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(assistant, true, "postTurn");

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("overflow", true);
		session.dispose();
	});

	it("still runs threshold compaction for Codex when global auto-compaction is disabled", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const codexModel = getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(codexModel, tempDir, "codex-key", false);
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(codexModel, 250_000);
		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(assistant, false, "prePrompt");

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("threshold", false);
		session.dispose();
	});

	it("falls back to estimated context tokens when assistant usage is zero", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const codexModel = getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(codexModel, tempDir, "codex-key", true);
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const threshold = (
			session as unknown as {
				_getAutoCompactLimit: (contextWindow: number, settings: { reserveTokens: number }) => number;
			}
		)._getAutoCompactLimit(codexModel.contextWindow ?? 0, { reserveTokens: 16_384 });
		const targetTokens = Number.isFinite(threshold) ? Math.floor(threshold) + 1_000 : 300_000;
		const largeAssistant = makeAssistant(codexModel, 0, Date.now(), "x".repeat(targetTokens * 4));

		session.agent.replaceMessages([largeAssistant]);

		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(largeAssistant, false, "prePrompt");

		expect(runAutoCompaction).toHaveBeenCalledTimes(1);
		expect(runAutoCompaction).toHaveBeenCalledWith("threshold", false);
		session.dispose();
	});

	it("keeps non-codex models disabled when auto-compaction is turned off", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const anthropicModel = getModel("anthropic", "claude-sonnet-4-5")!;
		const session = createSession(anthropicModel, tempDir, "anthropic-key", false);
		const runAutoCompaction = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (
						r: "overflow" | "threshold",
						w: boolean,
					) => Promise<{ ok: boolean; errorMessage?: string }>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue(AUTO_COMPACTION_OK);

		const assistant = makeAssistant(anthropicModel, 300_000);
		await (
			session as unknown as {
				_checkCompaction: (
					message: AssistantMessage,
					skipAbortedCheck: boolean,
					phase: "postTurn" | "prePrompt",
				) => Promise<void>;
			}
		)._checkCompaction(assistant, false, "prePrompt");

		expect(runAutoCompaction).not.toHaveBeenCalled();
		session.dispose();
	});

	it("checks compaction on turn_end and does not re-check same assistant on agent_end", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const session = createSession(model, tempDir, "anthropic-key");
		const checkCompaction = vi
			.spyOn(
				session as unknown as {
					_checkCompaction: (
						message: AssistantMessage,
						skipAbortedCheck: boolean,
						phase: "postTurn" | "prePrompt",
						hasImmediateContinuation?: boolean,
					) => Promise<void>;
				},
				"_checkCompaction",
			)
			.mockResolvedValue();

		const assistant = makeAssistant(model, 120_000);
		const handleAgentEvent = (
			session as unknown as {
				_handleAgentEvent: (event: unknown) => Promise<void>;
			}
		)._handleAgentEvent;

		await handleAgentEvent({
			type: "message_end",
			message: assistant,
		});
		await handleAgentEvent({
			type: "turn_end",
			message: assistant,
			toolResults: [],
		});
		await handleAgentEvent({
			type: "agent_end",
			messages: [assistant],
		});

		expect(checkCompaction).toHaveBeenCalledTimes(1);
		expect(checkCompaction).toHaveBeenCalledWith(assistant, true, "postTurn", false);
		session.dispose();
	});

	it("aborts synchronously on tool-use turn_end when threshold is exceeded", async () => {
		tempDir = join(tmpdir(), `pi-compaction-semantics-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });

		const model = getModel("openai-codex", "gpt-5.3-codex")!;
		const session = createSession(model, tempDir, "codex-key");
		const assistant = makeAssistant(model, 250_000);
		assistant.stopReason = "toolUse";
		session.agent.replaceMessages([assistant]);

		const abortSpy = vi.spyOn(session.agent, "abort");
		const checkCompaction = vi
			.spyOn(
				session as unknown as {
					_checkCompaction: (
						message: AssistantMessage,
						skipAbortedCheck: boolean,
						phase: "postTurn" | "prePrompt",
						hasImmediateContinuation?: boolean,
					) => Promise<void>;
				},
				"_checkCompaction",
			)
			.mockResolvedValue();

		const handleAgentEvent = (
			session as unknown as {
				_handleAgentEvent: (event: unknown) => Promise<void>;
			}
		)._handleAgentEvent;

		const eventPromise = handleAgentEvent({
			type: "turn_end",
			message: assistant,
			toolResults: [],
		});

		expect(abortSpy).toHaveBeenCalledTimes(1);

		await eventPromise;

		expect(checkCompaction).toHaveBeenCalledTimes(1);
		expect(checkCompaction).toHaveBeenCalledWith(assistant, true, "postTurn", true);
		session.dispose();
	});
});

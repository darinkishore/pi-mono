import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { isAbsolute, resolve } from "node:path";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { type Static, Type } from "@sinclair/typebox";
import type { IPty } from "node-pty";
import * as nodePty from "node-pty";
import { getShellConfig, getShellEnv, killProcessTree, sanitizeBinaryOutput } from "../../utils/shell.js";

const DEFAULT_EXEC_YIELD_TIME_MS = 10_000;
const DEFAULT_WRITE_STDIN_YIELD_TIME_MS = 250;
const DEFAULT_LOGIN_SHELL = true;
const SESSION_RETENTION_MS = 45 * 60 * 1000;
const SESSION_IDLE_TIMEOUT_MS = 45 * 60 * 1000;
const STDIN_CLOSED_ERROR = "stdin is closed for this session; rerun exec_command with tty=true to keep stdin open";

const execCommandSchema = Type.Object(
	{
		cmd: Type.String({ description: "Shell command to execute." }),
		workdir: Type.Optional(
			Type.String({
				description: "Optional working directory to run the command in; defaults to the turn cwd.",
			}),
		),
		shell: Type.Optional(
			Type.String({ description: "Shell binary to launch. Defaults to the user's default shell." }),
		),
		tty: Type.Optional(
			Type.Boolean({
				description:
					"Whether to allocate a TTY for the command. Defaults to false (plain pipes); set to true to open a PTY and access TTY process.",
			}),
		),
		login: Type.Optional(
			Type.Boolean({
				description: "Whether to run the shell with -l/-i semantics. Defaults to true.",
			}),
		),
		yield_time_ms: Type.Optional(
			Type.Number({ description: "How long to wait (in milliseconds) for output before yielding." }),
		),
		max_output_tokens: Type.Optional(
			Type.Number({
				description: "Maximum number of tokens to return. Excess output will be truncated.",
			}),
		),
		sandbox_permissions: Type.Optional(
			Type.String({
				description:
					'Whether to run without sandbox restrictions. Set to "require_escalated" to request it; defaults to "use_default".',
			}),
		),
		justification: Type.Optional(
			Type.String({
				description:
					"Only specify when sandbox_permissions is require_escalated. Explain why escalated execution is required.",
			}),
		),
		prefix_rule: Type.Optional(
			Type.Array(
				Type.String({
					description:
						"Only specify when sandbox_permissions is require_escalated. Prefix pattern to allow similar future commands.",
				}),
			),
		),
	},
	{ additionalProperties: false },
);

const writeStdinSchema = Type.Object(
	{
		session_id: Type.Number({ description: "Identifier of the running unified exec session." }),
		chars: Type.Optional(
			Type.String({
				description: "Bytes to write to stdin (may be empty to poll).",
			}),
		),
		yield_time_ms: Type.Optional(
			Type.Number({ description: "How long to wait (in milliseconds) for output before yielding." }),
		),
		max_output_tokens: Type.Optional(
			Type.Number({
				description: "Maximum number of tokens to return. Excess output will be truncated.",
			}),
		),
	},
	{ additionalProperties: false },
);

export type ExecCommandToolInput = Static<typeof execCommandSchema>;
export type WriteStdinToolInput = Static<typeof writeStdinSchema>;

interface UnifiedExecResponse {
	chunkId: string;
	wallTimeMs: number;
	output: string;
	exitCode?: number | null;
	sessionId?: number;
	originalTokenCount?: number;
}

interface UnifiedExecSession {
	id: number;
	backend: "pty" | "pipe";
	child?: ChildProcessWithoutNullStreams;
	pty?: IPty;
	supportsStdin: boolean;
	pendingOutput: string;
	closed: boolean;
	exitCode: number | null;
	waiters: Set<() => void>;
	cleanupTimer?: NodeJS.Timeout;
	idleTimer?: NodeJS.Timeout;
}

function createChunkId(): string {
	return randomBytes(3).toString("hex");
}

function estimateTokenCount(text: string): number {
	const matches = text.match(/\S+/g);
	return matches ? matches.length : 0;
}

function truncateToLastTokens(
	output: string,
	maxOutputTokens: number | undefined,
): { output: string; originalTokenCount?: number } {
	if (!maxOutputTokens || !Number.isFinite(maxOutputTokens) || maxOutputTokens <= 0) {
		return { output };
	}

	const tokenLimit = Math.floor(maxOutputTokens);
	const originalTokenCount = estimateTokenCount(output);
	if (originalTokenCount <= tokenLimit) {
		return { output };
	}

	const parts = output.split(/(\s+)/);
	const kept: string[] = [];
	let keptTokens = 0;

	for (let i = parts.length - 1; i >= 0; i--) {
		const part = parts[i] ?? "";
		kept.unshift(part);
		if (/\S/.test(part)) {
			keptTokens++;
			if (keptTokens >= tokenLimit) {
				break;
			}
		}
	}

	return {
		output: `[output truncated to last ${tokenLimit} tokens]\n${kept.join("").trimStart()}`,
		originalTokenCount,
	};
}

function formatResponse(response: UnifiedExecResponse): string {
	const lines: string[] = [];

	if (response.chunkId.length > 0) {
		lines.push(`Chunk ID: ${response.chunkId}`);
	}

	lines.push(`Wall time: ${(response.wallTimeMs / 1000).toFixed(4)} seconds`);

	if (response.exitCode !== undefined) {
		lines.push(`Process exited with code ${response.exitCode}`);
	}

	if (response.sessionId !== undefined) {
		lines.push(`Process running with session ID ${response.sessionId}`);
	}

	if (response.originalTokenCount !== undefined) {
		lines.push(`Original token count: ${response.originalTokenCount}`);
	}

	lines.push("Output:");
	lines.push(response.output);
	return lines.join("\n");
}

function resolveWorkdir(baseCwd: string, workdir?: string): string {
	if (!workdir || workdir.trim().length === 0) {
		return baseCwd;
	}
	return isAbsolute(workdir) ? workdir : resolve(baseCwd, workdir);
}

function shellSupportsLoginFlag(shellPath: string): boolean {
	const lower = shellPath.toLowerCase();
	return lower.endsWith("bash") || lower.endsWith("zsh");
}

function shellLooksLikePowerShell(shellPath: string): boolean {
	const lower = shellPath.toLowerCase();
	return lower.includes("powershell") || lower.includes("pwsh");
}

function resolveShellCommand(
	cmd: string,
	shellPath: string | undefined,
	login: boolean,
): { shell: string; args: string[] } {
	if (shellPath && shellLooksLikePowerShell(shellPath)) {
		return {
			shell: shellPath,
			args: ["-NoProfile", "-Command", cmd],
		};
	}

	const shellConfig = shellPath ? { shell: shellPath } : getShellConfig();
	const shell = shellConfig.shell;
	const canUseLogin = login && shellSupportsLoginFlag(shell);
	const commandFlag = canUseLogin ? "-lc" : "-c";
	return { shell, args: [commandFlag, cmd] };
}

function getPtyEnv(): Record<string, string> {
	const env = getShellEnv();
	const normalizedEnv: Record<string, string> = {};
	for (const [key, value] of Object.entries(env)) {
		if (typeof value === "string") {
			normalizedEnv[key] = value;
		}
	}
	return normalizedEnv;
}

export class UnifiedExecManager {
	private _sessions = new Map<number, UnifiedExecSession>();
	private _nextSessionId = 1;
	private _cwd: string;

	constructor(cwd: string) {
		this._cwd = cwd;
	}

	private _wakeWaiters(session: UnifiedExecSession): void {
		const waiters = [...session.waiters];
		for (const waiter of waiters) {
			waiter();
		}
	}

	private _appendOutput(session: UnifiedExecSession, data: string): void {
		if (data.length === 0) {
			return;
		}
		session.pendingOutput += data;
		this._markSessionActive(session);
		this._wakeWaiters(session);
	}

	private _clearSessionTimers(session: UnifiedExecSession): void {
		if (session.cleanupTimer) {
			clearTimeout(session.cleanupTimer);
			session.cleanupTimer = undefined;
		}
		if (session.idleTimer) {
			clearTimeout(session.idleTimer);
			session.idleTimer = undefined;
		}
	}

	private _killBackend(session: UnifiedExecSession): void {
		if (session.backend === "pty" && session.pty) {
			try {
				session.pty.kill();
			} catch {
				// Ignore kill failures; process may already be gone.
			}
			return;
		}
		if (session.child?.pid) {
			killProcessTree(session.child.pid);
		}
	}

	private _markSessionActive(session: UnifiedExecSession): void {
		if (session.closed) {
			return;
		}
		if (session.idleTimer) {
			clearTimeout(session.idleTimer);
		}
		session.idleTimer = setTimeout(() => {
			if (session.closed) {
				return;
			}
			this._appendOutput(session, "\nunified exec session closed after 45 minutes of inactivity\n");
			this._finalizeSession(session, null, true);
		}, SESSION_IDLE_TIMEOUT_MS);
		session.idleTimer.unref?.();
	}

	private _finalizeSession(session: UnifiedExecSession, exitCode: number | null, terminateProcess = false): void {
		if (session.closed) {
			return;
		}
		session.closed = true;
		session.exitCode = exitCode;
		if (session.idleTimer) {
			clearTimeout(session.idleTimer);
			session.idleTimer = undefined;
		}
		if (terminateProcess) {
			this._killBackend(session);
		}
		this._wakeWaiters(session);

		session.cleanupTimer = setTimeout(() => {
			this._sessions.delete(session.id);
		}, SESSION_RETENTION_MS);
		session.cleanupTimer.unref?.();
	}

	private async _waitForActivity(
		session: UnifiedExecSession,
		yieldTimeMs: number,
		signal?: AbortSignal,
	): Promise<void> {
		if (session.pendingOutput.length > 0 || session.closed) {
			return;
		}

		await new Promise<void>((resolve, reject) => {
			const onWake = () => {
				cleanup();
				resolve();
			};

			const timeout = setTimeout(onWake, yieldTimeMs);

			const onAbort = () => {
				cleanup();
				reject(new Error("Operation aborted"));
			};

			const cleanup = () => {
				clearTimeout(timeout);
				session.waiters.delete(onWake);
				signal?.removeEventListener("abort", onAbort);
			};

			session.waiters.add(onWake);

			if (signal) {
				if (signal.aborted) {
					onAbort();
					return;
				}
				signal.addEventListener("abort", onAbort, { once: true });
			}
		});
	}

	private _takePendingOutput(session: UnifiedExecSession): string {
		const output = session.pendingOutput;
		session.pendingOutput = "";
		return output;
	}

	private _buildResponse(
		session: UnifiedExecSession,
		startedAt: number,
		maxOutputTokens?: number,
	): UnifiedExecResponse {
		const output = this._takePendingOutput(session);
		const truncated = truncateToLastTokens(output, maxOutputTokens);
		return {
			chunkId: createChunkId(),
			wallTimeMs: Date.now() - startedAt,
			output: truncated.output,
			exitCode: session.closed ? session.exitCode : undefined,
			sessionId: session.closed ? undefined : session.id,
			originalTokenCount: truncated.originalTokenCount,
		};
	}

	private _destroySession(session: UnifiedExecSession): void {
		this._clearSessionTimers(session);
		if (!session.closed) {
			this._killBackend(session);
		}
		this._sessions.delete(session.id);
	}

	private _attachPipeProcess(session: UnifiedExecSession, shell: string, args: string[], cwd: string): void {
		session.backend = "pipe";
		session.supportsStdin = false;
		const child = spawn(shell, args, {
			cwd,
			env: getShellEnv(),
			stdio: ["pipe", "pipe", "pipe"],
			detached: true,
		});
		session.child = child;

		const onOutput = (chunk: Buffer): void => {
			const text = sanitizeBinaryOutput(chunk.toString("utf8")).replace(/\r/g, "");
			this._appendOutput(session, text);
		};

		child.stdout.on("data", onOutput);
		child.stderr.on("data", onOutput);
		child.on("close", (code) => {
			this._finalizeSession(session, code);
		});
		child.on("error", (error) => {
			this._appendOutput(session, `${error.message}\n`);
			this._finalizeSession(session, 1);
		});

		child.stdin.end();
	}

	async execCommand(input: ExecCommandToolInput, signal?: AbortSignal): Promise<UnifiedExecResponse> {
		const startedAt = Date.now();
		const sessionId = this._nextSessionId++;
		const yieldTimeMs = Math.max(0, Math.floor(input.yield_time_ms ?? DEFAULT_EXEC_YIELD_TIME_MS));
		const login = input.login ?? DEFAULT_LOGIN_SHELL;
		const { shell, args } = resolveShellCommand(input.cmd, input.shell, login);
		const cwd = resolveWorkdir(this._cwd, input.workdir);
		const requestedPty = input.tty === true;

		const session: UnifiedExecSession = {
			id: sessionId,
			backend: requestedPty ? "pty" : "pipe",
			supportsStdin: requestedPty,
			pendingOutput: "",
			closed: false,
			exitCode: null,
			waiters: new Set(),
		};
		this._sessions.set(sessionId, session);
		this._markSessionActive(session);

		if (requestedPty) {
			try {
				const pty = nodePty.spawn(shell, args, {
					name: process.platform === "win32" ? "xterm" : "xterm-256color",
					cols: 120,
					rows: 40,
					cwd,
					env: getPtyEnv(),
				});
				session.pty = pty;
				pty.onData((data) => {
					const text = sanitizeBinaryOutput(data).replace(/\r/g, "");
					this._appendOutput(session, text);
				});
				pty.onExit(({ exitCode }) => {
					this._finalizeSession(session, exitCode);
				});
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				this._appendOutput(session, `Failed to start PTY, falling back to pipes: ${message}\n`);
				this._attachPipeProcess(session, shell, args, cwd);
			}
		} else {
			this._attachPipeProcess(session, shell, args, cwd);
		}

		try {
			await this._waitForActivity(session, yieldTimeMs, signal);
			return this._buildResponse(session, startedAt, input.max_output_tokens);
		} catch (error) {
			this._destroySession(session);
			throw error;
		}
	}

	async writeStdin(input: WriteStdinToolInput, signal?: AbortSignal): Promise<UnifiedExecResponse> {
		const startedAt = Date.now();
		const sessionId = Math.floor(input.session_id);
		const session = this._sessions.get(sessionId);
		if (!session) {
			throw new Error(`Unknown session_id ${input.session_id}`);
		}

		if (input.chars && input.chars.length > 0) {
			if (session.closed || !session.supportsStdin) {
				throw new Error(STDIN_CLOSED_ERROR);
			}
			if (session.backend === "pty") {
				session.pty?.write(input.chars);
			} else if (session.child?.stdin && !session.child.stdin.destroyed && session.child.stdin.writable) {
				await new Promise<void>((resolve, reject) => {
					session.child?.stdin.write(input.chars ?? "", (error) => {
						if (error) {
							reject(error);
							return;
						}
						resolve();
					});
				});
			} else {
				throw new Error(STDIN_CLOSED_ERROR);
			}
		}

		this._markSessionActive(session);
		const yieldTimeMs = Math.max(0, Math.floor(input.yield_time_ms ?? DEFAULT_WRITE_STDIN_YIELD_TIME_MS));
		await this._waitForActivity(session, yieldTimeMs, signal);
		return this._buildResponse(session, startedAt, input.max_output_tokens);
	}
}

export interface UnifiedExecToolOptions {
	manager?: UnifiedExecManager;
}

function createExecCommandToolWithManager(manager: UnifiedExecManager): AgentTool<typeof execCommandSchema> {
	return {
		name: "exec_command",
		label: "exec_command",
		description: "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
		parameters: execCommandSchema,
		execute: async (_toolCallId: string, input: ExecCommandToolInput, signal?: AbortSignal) => {
			const response = await manager.execCommand(input, signal);
			return {
				content: [{ type: "text", text: formatResponse(response) }],
				details: response,
			};
		},
	};
}

function createWriteStdinToolWithManager(manager: UnifiedExecManager): AgentTool<typeof writeStdinSchema> {
	return {
		name: "write_stdin",
		label: "write_stdin",
		description: "Writes characters to an existing unified exec session and returns recent output.",
		parameters: writeStdinSchema,
		execute: async (_toolCallId: string, input: WriteStdinToolInput, signal?: AbortSignal) => {
			const response = await manager.writeStdin(input, signal);
			return {
				content: [{ type: "text", text: formatResponse(response) }],
				details: response,
			};
		},
	};
}

export function createUnifiedExecTools(
	cwd: string,
	options?: UnifiedExecToolOptions,
): { execCommandTool: AgentTool<typeof execCommandSchema>; writeStdinTool: AgentTool<typeof writeStdinSchema> } {
	const manager = options?.manager ?? new UnifiedExecManager(cwd);
	return {
		execCommandTool: createExecCommandToolWithManager(manager),
		writeStdinTool: createWriteStdinToolWithManager(manager),
	};
}

export function createExecCommandTool(
	cwd: string,
	options?: UnifiedExecToolOptions,
): AgentTool<typeof execCommandSchema> {
	return createUnifiedExecTools(cwd, options).execCommandTool;
}

export function createWriteStdinTool(
	cwd: string,
	options?: UnifiedExecToolOptions,
): AgentTool<typeof writeStdinSchema> {
	return createUnifiedExecTools(cwd, options).writeStdinTool;
}

/** Default unified-exec tools using process.cwd() - for backwards compatibility */
const defaultUnifiedExecManager = new UnifiedExecManager(process.cwd());
export const execCommandTool = createExecCommandTool(process.cwd(), { manager: defaultUnifiedExecManager });
export const writeStdinTool = createWriteStdinTool(process.cwd(), { manager: defaultUnifiedExecManager });

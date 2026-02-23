import { spawn, spawnSync } from "node:child_process";
import fsp from "node:fs/promises";
import path from "node:path";

const DEFAULT_EXEC_YIELD_TIME_MS = 10_000;
const DEFAULT_WRITE_STDIN_YIELD_TIME_MS = 250;
const MIN_YIELD_TIME_MS = 250;
const MIN_EMPTY_YIELD_TIME_MS = 5_000;
const MAX_YIELD_TIME_MS = 30_000;
const DEFAULT_TEST_SYNC_TIMEOUT_MS = 1_000;
const DEFAULT_MAX_OUTPUT_TOKENS = 10_000;
const APPROX_BYTES_PER_TOKEN = 4;
const UNIFIED_EXEC_OUTPUT_MAX_BYTES = 1024 * 1024;
const MAX_ENTRY_LENGTH = 500;
const MAX_LINE_LENGTH = 500;
const TAB_WIDTH = 4;
const COMMENT_PREFIXES = ["#", "//", "--"] as const;
const GREP_TIMEOUT_MS = 30_000;

export interface ExecCommandArgs {
	cmd: string;
	workdir?: string;
	shell?: string;
	login?: boolean;
	tty?: boolean;
	yield_time_ms?: number;
	max_output_tokens?: number;
}

export interface WriteStdinArgs {
	session_id: number;
	chars?: string;
	yield_time_ms?: number;
	max_output_tokens?: number;
}

export interface ShellArgs {
	command: string[];
	workdir?: string;
	timeout_ms?: number;
}

export interface ShellCommandArgs {
	command: string;
	workdir?: string;
	login?: boolean;
	timeout_ms?: number;
}

export interface GrepFilesArgs {
	pattern: string;
	include?: string;
	path?: string;
	limit?: number;
}

export interface ReadFileIndentationArgs {
	anchor_line?: number;
	max_levels?: number;
	include_siblings?: boolean;
	include_header?: boolean;
	max_lines?: number;
}

export interface ReadFileArgs {
	file_path: string;
	offset?: number;
	limit?: number;
	mode?: "slice" | "indentation";
	indentation?: ReadFileIndentationArgs;
}

export interface ListDirArgs {
	dir_path: string;
	offset?: number;
	limit?: number;
	depth?: number;
}

export interface ViewImageArgs {
	path: string;
}

export interface McpResourcesArgs {
	server?: string;
	cursor?: string;
}

export interface ReadMcpResourceArgs {
	server: string;
	uri: string;
}

export interface UpdatePlanArgs {
	explanation?: string;
	plan: Array<{ step: string; status: "pending" | "in_progress" | "completed" }>;
}

export interface RequestUserInputArgs {
	questions: Array<{
		id: string;
		header: string;
		question: string;
		options: Array<{
			label: string;
			description: string;
		}>;
	}>;
}

export interface SpawnAgentArgs {
	message?: string;
	items?: ToolInputItem[];
	agent_type?: string;
}

export interface SendInputArgs {
	id: string;
	message?: string;
	items?: ToolInputItem[];
	interrupt?: boolean;
}

export interface ResumeAgentArgs {
	id: string;
}

export interface WaitArgs {
	ids: string[];
	timeout_ms?: number;
}

export interface CloseAgentArgs {
	id: string;
}

export interface SearchToolBm25Args {
	query: string;
	limit?: number;
}

export interface TestSyncToolArgs {
	sleep_before_ms?: number;
	sleep_after_ms?: number;
	barrier?: {
		id: string;
		participants: number;
		timeout_ms?: number;
	};
}

export interface ToolInputItem {
	type?: "text" | "image" | "local_image" | "skill" | "mention";
	text?: string;
	image_url?: string;
	path?: string;
	name?: string;
}

export type CodexToolName =
	| "apply_patch"
	| "exec_command"
	| "write_stdin"
	| "shell"
	| "local_shell"
	| "container.exec"
	| "shell_command"
	| "grep_files"
	| "read_file"
	| "list_dir"
	| "view_image"
	| "list_mcp_resources"
	| "list_mcp_resource_templates"
	| "read_mcp_resource"
	| "search_tool_bm25"
	| "js_repl"
	| "js_repl_reset"
	| "web_search"
	| "test_sync_tool"
	| "update_plan"
	| "request_user_input"
	| "spawn_agent"
	| "send_input"
	| "resume_agent"
	| "wait"
	| "close_agent";

export interface CodexToolAdapters {
	listMcpResources?: (args: McpResourcesArgs) => Promise<unknown> | unknown;
	listMcpResourceTemplates?: (args: McpResourcesArgs) => Promise<unknown> | unknown;
	readMcpResource?: (args: ReadMcpResourceArgs) => Promise<unknown> | unknown;
	searchToolBm25?: (args: SearchToolBm25Args) => Promise<unknown> | unknown;
	jsRepl?: (source: string) => Promise<unknown> | unknown;
	jsReplReset?: (args: Record<string, never>) => Promise<unknown> | unknown;
	webSearch?: (args: unknown) => Promise<unknown> | unknown;
	updatePlan?: (args: UpdatePlanArgs) => Promise<unknown> | unknown;
	requestUserInput?: (args: RequestUserInputArgs) => Promise<unknown> | unknown;
	spawnAgent?: (args: SpawnAgentArgs) => Promise<unknown> | unknown;
	sendInput?: (args: SendInputArgs) => Promise<unknown> | unknown;
	resumeAgent?: (args: ResumeAgentArgs) => Promise<unknown> | unknown;
	wait?: (args: WaitArgs) => Promise<unknown> | unknown;
	closeAgent?: (args: CloseAgentArgs) => Promise<unknown> | unknown;
}

export interface CodexToolboxOptions {
	cwd?: string;
	adapters?: CodexToolAdapters;
	applyPatch?: (patch: string) => Promise<string> | string;
}

type ExecSession = {
	id: number;
	process: ReturnType<typeof spawn>;
	tty: boolean;
	output: HeadTailChunkBuffer;
	exitCode: number | null;
	exited: boolean;
	waiters: Set<() => void>;
};

type TestSyncBarrierWaiter = {
	resolve: () => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
};

type TestSyncBarrierState = {
	participants: number;
	waiters: TestSyncBarrierWaiter[];
};

const testSyncBarriers = new Map<string, TestSyncBarrierState>();

class HeadTailChunkBuffer {
	private readonly maxBytes: number;
	private readonly headBudget: number;
	private readonly tailBudget: number;
	private readonly head: Buffer[] = [];
	private readonly tail: Buffer[] = [];
	private headBytes = 0;
	private tailBytes = 0;

	constructor(maxBytes: number) {
		this.maxBytes = Math.max(0, maxBytes);
		this.headBudget = Math.floor(this.maxBytes / 2);
		this.tailBudget = this.maxBytes - this.headBudget;
	}

	pushChunk(chunk: Buffer): void {
		if (this.maxBytes === 0) {
			return;
		}

		if (this.headBytes < this.headBudget) {
			const remainingHead = this.headBudget - this.headBytes;
			if (chunk.length <= remainingHead) {
				this.head.push(chunk);
				this.headBytes += chunk.length;
				return;
			}

			if (remainingHead > 0) {
				const headPart = chunk.subarray(0, remainingHead);
				this.head.push(headPart);
				this.headBytes += headPart.length;
			}

			const tailPart = chunk.subarray(remainingHead);
			if (tailPart.length > 0) {
				this.pushToTail(tailPart);
			}
			return;
		}

		this.pushToTail(chunk);
	}

	drainToBuffer(): Buffer {
		if (this.head.length === 0 && this.tail.length === 0) {
			return Buffer.alloc(0);
		}

		const chunks = [...this.head, ...this.tail];
		this.head.length = 0;
		this.tail.length = 0;
		this.headBytes = 0;
		this.tailBytes = 0;
		return Buffer.concat(chunks);
	}

	private pushToTail(chunk: Buffer): void {
		if (this.tailBudget === 0) {
			return;
		}

		if (chunk.length >= this.tailBudget) {
			const start = chunk.length - this.tailBudget;
			const kept = chunk.subarray(start);
			this.tail.length = 0;
			this.tail.push(kept);
			this.tailBytes = kept.length;
			return;
		}

		this.tail.push(chunk);
		this.tailBytes += chunk.length;
		this.trimTailToBudget();
	}

	private trimTailToBudget(): void {
		let excess = this.tailBytes - this.tailBudget;
		while (excess > 0 && this.tail.length > 0) {
			const front = this.tail[0]!;
			if (excess >= front.length) {
				excess -= front.length;
				this.tail.shift();
				this.tailBytes -= front.length;
				continue;
			}

			this.tail[0] = front.subarray(excess);
			this.tailBytes -= excess;
			excess = 0;
		}
	}
}

export class CodexToolbox {
	private readonly cwd: string;
	private readonly adapters: CodexToolAdapters;
	private readonly applyPatchExecutor?: (patch: string) => Promise<string> | string;
	private readonly sessions = new Map<number, ExecSession>();
	private nextSessionId = 1_000;

	constructor(options: CodexToolboxOptions = {}) {
		this.cwd = options.cwd ?? process.cwd();
		this.adapters = options.adapters ?? {};
		this.applyPatchExecutor = options.applyPatch;
	}

	async dispatch(name: CodexToolName, args: unknown): Promise<unknown> {
		switch (name) {
			case "apply_patch": {
				if (typeof args === "string") {
					return this.applyPatch(args);
				}
				if (isObject(args) && typeof args.input === "string") {
					return this.applyPatch(args.input);
				}
				throw new Error("apply_patch expects a patch string or { input: string }");
			}
			case "exec_command":
				return this.execCommand(assertType<ExecCommandArgs>(args));
			case "write_stdin":
				return this.writeStdin(assertType<WriteStdinArgs>(args));
			case "shell":
			case "local_shell":
			case "container.exec":
				return this.shell(assertType<ShellArgs>(args));
			case "shell_command":
				return this.shellCommand(assertType<ShellCommandArgs>(args));
			case "grep_files":
				return this.grepFiles(assertType<GrepFilesArgs>(args));
			case "read_file":
				return this.readFile(assertType<ReadFileArgs>(args));
			case "list_dir":
				return this.listDir(assertType<ListDirArgs>(args));
			case "view_image":
				return this.viewImage(assertType<ViewImageArgs>(args));
			case "list_mcp_resources":
				return this.runAdapter("listMcpResources", assertType<McpResourcesArgs>(args));
			case "list_mcp_resource_templates":
				return this.runAdapter("listMcpResourceTemplates", assertType<McpResourcesArgs>(args));
			case "read_mcp_resource":
				return this.runAdapter("readMcpResource", assertType<ReadMcpResourceArgs>(args));
			case "search_tool_bm25":
				return this.runAdapter("searchToolBm25", assertType<SearchToolBm25Args>(args));
			case "js_repl":
				if (typeof args !== "string") {
					throw new Error("js_repl expects raw JavaScript source text");
				}
				return this.runAdapter("jsRepl", args);
			case "js_repl_reset":
				return this.runAdapter("jsReplReset", isObject(args) ? (args as Record<string, never>) : {});
			case "web_search":
				return this.runAdapter("webSearch", args);
			case "test_sync_tool":
				return this.testSyncTool(assertType<TestSyncToolArgs>(args));
			case "update_plan":
				return this.runAdapter("updatePlan", assertType<UpdatePlanArgs>(args));
			case "request_user_input":
				return this.runAdapter("requestUserInput", assertType<RequestUserInputArgs>(args));
			case "spawn_agent":
				return this.runAdapter("spawnAgent", assertType<SpawnAgentArgs>(args));
			case "send_input":
				return this.runAdapter("sendInput", assertType<SendInputArgs>(args));
			case "resume_agent":
				return this.runAdapter("resumeAgent", assertType<ResumeAgentArgs>(args));
			case "wait":
				return this.runAdapter("wait", assertType<WaitArgs>(args));
			case "close_agent":
				return this.runAdapter("closeAgent", assertType<CloseAgentArgs>(args));
			default:
				return exhaustive(name);
		}
	}

	async applyPatch(patch: string): Promise<string> {
		if (!this.applyPatchExecutor) {
			throw new Error("apply_patch executor is not configured");
		}

		const result = await this.applyPatchExecutor(patch);
		return typeof result === "string" ? result : JSON.stringify(result);
	}

	async execCommand(args: ExecCommandArgs): Promise<string> {
		if (!args || typeof args.cmd !== "string") {
			throw new Error("exec_command requires cmd");
		}

		const login = args.login ?? true;
		const tty = args.tty ?? false;
		const yieldTimeMs = clampYieldTime(positiveInt(args.yield_time_ms, DEFAULT_EXEC_YIELD_TIME_MS));
		const maxOutputTokens = resolveMaxTokens(args.max_output_tokens);
		const workdir = this.resolvePath(args.workdir);
		const command = this.deriveShellCommand(args.cmd, args.shell, login);

		const startedAt = Date.now();
		const sessionId = this.allocateSessionId();

		const child = spawn(command[0]!, command.slice(1), {
			cwd: workdir,
			stdio: [tty ? "pipe" : "ignore", "pipe", "pipe"],
			env: process.env,
			shell: false,
		});

		const session: ExecSession = {
			id: sessionId,
			process: child,
			tty,
			output: new HeadTailChunkBuffer(UNIFIED_EXEC_OUTPUT_MAX_BYTES),
			exitCode: null,
			exited: false,
			waiters: new Set(),
		};

		child.stdout?.on("data", (chunk: Buffer | string) => {
			const asBuffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, "utf8");
			const normalized = tty ? Buffer.from(normalizeTtyLineEndings(asBuffer.toString("utf8")), "utf8") : asBuffer;
			session.output.pushChunk(normalized);
			notifySession(session);
		});
		child.stderr?.on("data", (chunk: Buffer | string) => {
			const asBuffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, "utf8");
			const normalized = tty ? Buffer.from(normalizeTtyLineEndings(asBuffer.toString("utf8")), "utf8") : asBuffer;
			session.output.pushChunk(normalized);
			notifySession(session);
		});
		child.on("error", (error: Error) => {
			session.output.pushChunk(Buffer.from(`${error.message}\n`, "utf8"));
			session.exited = true;
			session.exitCode = session.exitCode ?? 1;
			notifySession(session);
		});
		child.on("close", (code: number | null) => {
			session.exited = true;
			session.exitCode = code ?? -1;
			notifySession(session);
		});

		this.sessions.set(session.id, session);

		await waitUntilYieldOrExit(session, yieldTimeMs);

		const wallTimeMs = Date.now() - startedAt;
		const collected = session.output.drainToBuffer().toString("utf8");
		const output = formattedTruncateText(collected, maxOutputTokens);

		const response = this.formatUnifiedExecResponse({
			chunkId: generateChunkId(),
			wallTimeMs,
			exitCode: session.exited ? session.exitCode : null,
			processId: session.exited ? null : session.id,
			output,
			originalTokenCount: approxTokenCount(collected),
		});

		if (session.exited) {
			this.sessions.delete(session.id);
		}

		return response;
	}

	async writeStdin(args: WriteStdinArgs): Promise<string> {
		if (!args || typeof args.session_id !== "number") {
			throw new Error("write_stdin requires session_id");
		}

		const session = this.sessions.get(args.session_id);
		if (!session) {
			throw new Error(`write_stdin failed: Unknown process id ${args.session_id}`);
		}

		const input = args.chars ?? "";
		if (input.length > 0) {
			if (!session.tty) {
				throw new Error(
					"write_stdin failed: stdin is closed for this session; rerun exec_command with tty=true to keep stdin open",
				);
			}
			// PTY-backed unified exec echoes typed input back to the terminal stream.
			session.output.pushChunk(Buffer.from(normalizeTtyLineEndings(input), "utf8"));
			session.process.stdin?.write(input);
			await sleep(100);
		}

		const yieldTimeMs = clampYieldTime(positiveInt(args.yield_time_ms, DEFAULT_WRITE_STDIN_YIELD_TIME_MS));
		const boundedYield =
			input.length === 0 ? clamp(yieldTimeMs, MIN_EMPTY_YIELD_TIME_MS, MAX_YIELD_TIME_MS) : yieldTimeMs;
		const maxOutputTokens = resolveMaxTokens(args.max_output_tokens);

		const startedAt = Date.now();
		await waitUntilYieldOrExit(session, boundedYield);
		const wallTimeMs = Date.now() - startedAt;

		const collected = session.output.drainToBuffer().toString("utf8");
		const output = formattedTruncateText(collected, maxOutputTokens);

		const response = this.formatUnifiedExecResponse({
			chunkId: generateChunkId(),
			wallTimeMs,
			exitCode: session.exited ? session.exitCode : null,
			processId: session.exited ? null : session.id,
			output,
			originalTokenCount: approxTokenCount(collected),
		});

		if (session.exited) {
			this.sessions.delete(session.id);
		}

		return response;
	}

	async shell(args: ShellArgs): Promise<string> {
		if (!args || !Array.isArray(args.command) || args.command.length === 0) {
			throw new Error("shell requires non-empty command array");
		}

		const cwd = this.resolvePath(args.workdir);
		const timeoutMs = positiveInt(args.timeout_ms, DEFAULT_EXEC_YIELD_TIME_MS);
		const { content, durationMs, exitCode } = await runShellCommandArray(args.command, cwd, timeoutMs);
		const durationSeconds = roundToOneDecimal(durationMs / 1000);
		const output = formattedTruncateText(content, DEFAULT_MAX_OUTPUT_TOKENS);

		return JSON.stringify({
			output,
			metadata: {
				exit_code: exitCode,
				duration_seconds: durationSeconds,
			},
		});
	}

	async shellCommand(args: ShellCommandArgs): Promise<string> {
		if (!args || typeof args.command !== "string") {
			throw new Error("shell_command requires command");
		}

		const login = args.login ?? true;
		const command = this.deriveShellCommand(args.command, undefined, login);
		const cwd = this.resolvePath(args.workdir);
		const timeoutMs = positiveInt(args.timeout_ms, DEFAULT_EXEC_YIELD_TIME_MS);
		const { content, durationMs, exitCode } = await runShellCommandArray(command, cwd, timeoutMs);
		const durationSeconds = roundToOneDecimal(durationMs / 1000);
		const totalLines = rustLineCount(content);
		const output = truncateTextByTokens(content, DEFAULT_MAX_OUTPUT_TOKENS);

		const sections: string[] = [];
		sections.push(`Exit code: ${exitCode}`);
		sections.push(`Wall time: ${durationSeconds} seconds`);
		if (totalLines !== rustLineCount(output)) {
			sections.push(`Total output lines: ${totalLines}`);
		}
		sections.push("Output:");
		sections.push(output);

		return sections.join("\n");
	}

	async grepFiles(args: GrepFilesArgs): Promise<string> {
		const pattern = `${args.pattern ?? ""}`.trim();
		if (pattern.length === 0) {
			throw new Error("pattern must not be empty");
		}

		const rawLimit = args.limit ?? 100;
		if (rawLimit <= 0) {
			throw new Error("limit must be greater than zero");
		}
		const limit = Math.min(rawLimit, 2000);

		const searchPath = this.resolvePath(args.path);
		await fsp.access(searchPath).catch((error: unknown) => {
			throw new Error(`unable to access \`${searchPath}\`: ${formatErrorMessage(error)}`);
		});

		const include = args.include?.trim();

		const rgArgs = ["--files-with-matches", "--sortr=modified", "--regexp", pattern, "--no-messages"];
		if (include) {
			rgArgs.push("--glob", include);
		}
		rgArgs.push("--", searchPath);

		const result = spawnSync("rg", rgArgs, {
			cwd: this.cwd,
			encoding: "utf8",
			timeout: GREP_TIMEOUT_MS,
			maxBuffer: 16 * 1024 * 1024,
		});

		if (result.error) {
			const err = result.error as NodeJS.ErrnoException;
			if (err.code === "ETIMEDOUT") {
				throw new Error("rg timed out after 30 seconds");
			}
			throw new Error(`failed to launch rg: ${err.message}. Ensure ripgrep is installed and on PATH.`);
		}

		if (result.status === null && result.signal === "SIGTERM") {
			throw new Error("rg timed out after 30 seconds");
		}

		if (result.status === 1) {
			return "No matches found.";
		}

		if (result.status !== 0) {
			const stderr = result.stderr ?? "";
			throw new Error(`rg failed: ${stderr}`);
		}

		const output = result.stdout ?? "";
		const results = output
			.split("\n")
			.filter((line) => line.length > 0)
			.slice(0, limit);

		return results.length === 0 ? "No matches found." : results.join("\n");
	}

	async readFile(args: ReadFileArgs): Promise<string> {
		const offset = positiveInt(args.offset, 1);
		const limit = positiveInt(args.limit, 2000);

		if (offset <= 0) {
			throw new Error("offset must be a 1-indexed line number");
		}
		if (limit <= 0) {
			throw new Error("limit must be greater than zero");
		}
		if (!path.isAbsolute(args.file_path)) {
			throw new Error("file_path must be an absolute path");
		}

		const mode = args.mode ?? "slice";

		if (mode === "slice") {
			const lines = await readFileSlice(args.file_path, offset, limit);
			return lines.join("\n");
		}

		const indentation: Required<ReadFileIndentationArgs> = {
			anchor_line: args.indentation?.anchor_line ?? offset,
			max_levels: args.indentation?.max_levels ?? 0,
			include_siblings: args.indentation?.include_siblings ?? false,
			include_header: args.indentation?.include_header ?? true,
			max_lines: args.indentation?.max_lines ?? limit,
		};

		const lines = await readFileIndentation(args.file_path, limit, indentation);
		return lines.join("\n");
	}

	async listDir(args: ListDirArgs): Promise<string> {
		const offset = positiveInt(args.offset, 1);
		const limit = positiveInt(args.limit, 25);
		const depth = positiveInt(args.depth, 2);

		if (offset <= 0) {
			throw new Error("offset must be a 1-indexed entry number");
		}
		if (limit <= 0) {
			throw new Error("limit must be greater than zero");
		}
		if (depth <= 0) {
			throw new Error("depth must be greater than zero");
		}
		if (!path.isAbsolute(args.dir_path)) {
			throw new Error("dir_path must be an absolute path");
		}

		const entries = await listDirSlice(args.dir_path, offset, limit, depth);
		return [`Absolute path: ${args.dir_path}`, ...entries].join("\n");
	}

	async viewImage(args: ViewImageArgs): Promise<string> {
		const resolved = this.resolvePath(args.path);
		const stat = await fsp.stat(resolved).catch((error: unknown) => {
			throw new Error(`unable to locate image at \`${resolved}\`: ${formatErrorMessage(error)}`);
		});

		if (!stat.isFile()) {
			throw new Error(`image path \`${resolved}\` is not a file`);
		}

		return "attached local image path";
	}

	async testSyncTool(args: TestSyncToolArgs): Promise<string> {
		const beforeMs = positiveInt(args?.sleep_before_ms, 0);
		const afterMs = positiveInt(args?.sleep_after_ms, 0);
		if (beforeMs > 0) {
			await sleep(beforeMs);
		}

		if (args?.barrier) {
			await waitOnTestSyncBarrier(args.barrier);
		}

		if (afterMs > 0) {
			await sleep(afterMs);
		}

		return "ok";
	}

	async dispose(): Promise<void> {
		for (const [id, session] of this.sessions) {
			try {
				session.process.kill("SIGKILL");
			} catch {
				// ignore cleanup errors
			}
			this.sessions.delete(id);
		}
	}

	private async runAdapter<K extends keyof CodexToolAdapters>(
		key: K,
		args: Parameters<NonNullable<CodexToolAdapters[K]>>[0],
	): Promise<unknown> {
		const adapter = this.adapters[key];
		if (!adapter) {
			throw new Error(`tool adapter '${String(key)}' is not configured`);
		}
		return await (adapter as (...toolArgs: unknown[]) => unknown)(args);
	}

	private formatUnifiedExecResponse(params: {
		chunkId: string;
		wallTimeMs: number;
		exitCode: number | null;
		processId: number | null;
		output: string;
		originalTokenCount: number;
	}): string {
		const lines: string[] = [];
		if (params.chunkId.length > 0) {
			lines.push(`Chunk ID: ${params.chunkId}`);
		}
		lines.push(`Wall time: ${(params.wallTimeMs / 1000).toFixed(4)} seconds`);

		if (params.exitCode !== null) {
			lines.push(`Process exited with code ${params.exitCode}`);
		}

		if (params.processId !== null) {
			lines.push(`Process running with session ID ${params.processId}`);
		}

		lines.push(`Original token count: ${params.originalTokenCount}`);
		lines.push("Output:");
		lines.push(params.output);

		return lines.join("\n");
	}

	private deriveShellCommand(command: string, explicitShell: string | undefined, login: boolean): string[] {
		const shell = (
			explicitShell ??
			process.env.SHELL ??
			(process.platform === "win32" ? "powershell" : "bash")
		).trim();
		const lower = path.basename(shell).toLowerCase();

		if (lower.includes("powershell") || lower === "pwsh" || lower === "pwsh.exe") {
			if (login) {
				return [shell, "-Command", command];
			}
			return [shell, "-NoProfile", "-Command", command];
		}

		if (lower === "cmd" || lower === "cmd.exe") {
			return [shell, "/c", command];
		}

		return [shell, login ? "-lc" : "-c", command];
	}

	private resolvePath(maybeRelative?: string): string {
		if (!maybeRelative || maybeRelative.length === 0) {
			return this.cwd;
		}
		if (path.isAbsolute(maybeRelative)) {
			return maybeRelative;
		}
		return path.join(this.cwd, maybeRelative);
	}

	private allocateSessionId(): number {
		while (true) {
			const id = this.nextSessionId;
			this.nextSessionId += 1;
			if (!this.sessions.has(id)) {
				return id;
			}
		}
	}
}

async function readFileSlice(filePath: string, offset: number, limit: number): Promise<string[]> {
	const content = await fsp.readFile(filePath).catch((error: unknown) => {
		throw new Error(`failed to read file: ${formatErrorMessage(error)}`);
	});
	const normalized = content.toString("utf8").replaceAll("\r\n", "\n");
	const allLines = normalized.split("\n");

	if (allLines.length > 0 && allLines[allLines.length - 1] === "") {
		allLines.pop();
	}

	if (offset > allLines.length) {
		throw new Error("offset exceeds file length");
	}

	const out: string[] = [];
	for (let lineNumber = offset; lineNumber <= allLines.length && out.length < limit; lineNumber += 1) {
		const raw = allLines[lineNumber - 1] ?? "";
		out.push(`L${lineNumber}: ${truncateLine(raw)}`);
	}
	return out;
}

type LineRecord = {
	number: number;
	raw: string;
	display: string;
	indent: number;
};

async function readFileIndentation(
	filePath: string,
	limit: number,
	indentation: Required<ReadFileIndentationArgs>,
): Promise<string[]> {
	const lines = await collectLineRecords(filePath);
	if (indentation.anchor_line <= 0) {
		throw new Error("anchor_line must be a 1-indexed line number");
	}

	if (lines.length === 0 || indentation.anchor_line > lines.length) {
		throw new Error("anchor_line exceeds file length");
	}

	const guardLimit = indentation.max_lines;
	if (guardLimit <= 0) {
		throw new Error("max_lines must be greater than zero");
	}

	const anchorIndex = indentation.anchor_line - 1;
	const effectiveIndents = computeEffectiveIndents(lines);
	const anchorIndent = effectiveIndents[anchorIndex] ?? 0;
	const minIndent = indentation.max_levels === 0 ? 0 : Math.max(anchorIndent - indentation.max_levels * TAB_WIDTH, 0);

	const finalLimit = Math.min(limit, guardLimit, lines.length);
	if (finalLimit === 1) {
		const anchor = lines[anchorIndex]!;
		return [`L${anchor.number}: ${anchor.display}`];
	}

	let i = anchorIndex - 1;
	let j = anchorIndex + 1;
	let iCounterMinIndent = 0;
	let jCounterMinIndent = 0;

	const out: LineRecord[] = [lines[anchorIndex]!];

	while (out.length < finalLimit) {
		let progressed = 0;

		if (i >= 0) {
			const record = lines[i]!;
			const indent = effectiveIndents[i] ?? 0;
			if (indent >= minIndent) {
				out.unshift(record);
				progressed += 1;
				i -= 1;

				if (indent === minIndent && !indentation.include_siblings) {
					const allowHeaderComment = indentation.include_header && isComment(record.raw);
					const canTake = allowHeaderComment || iCounterMinIndent === 0;
					if (canTake) {
						iCounterMinIndent += 1;
					} else {
						out.shift();
						progressed -= 1;
						i = -1;
					}
				}

				if (out.length >= finalLimit) {
					break;
				}
			} else {
				i = -1;
			}
		}

		if (j < lines.length) {
			const record = lines[j]!;
			const indent = effectiveIndents[j] ?? 0;
			if (indent >= minIndent) {
				out.push(record);
				progressed += 1;
				j += 1;

				if (indent === minIndent && !indentation.include_siblings) {
					if (jCounterMinIndent > 0) {
						out.pop();
						progressed -= 1;
						j = lines.length;
					}
					jCounterMinIndent += 1;
				}
			} else {
				j = lines.length;
			}
		}

		if (progressed === 0) {
			break;
		}
	}

	trimEdgeBlankLines(out);
	return out.map((record) => `L${record.number}: ${record.display}`);
}

async function collectLineRecords(filePath: string): Promise<LineRecord[]> {
	const content = await fsp.readFile(filePath).catch((error: unknown) => {
		throw new Error(`failed to read file: ${formatErrorMessage(error)}`);
	});
	const normalized = content.toString("utf8").replaceAll("\r\n", "\n");
	const sourceLines = normalized.split("\n");
	if (sourceLines.length > 0 && sourceLines[sourceLines.length - 1] === "") {
		sourceLines.pop();
	}

	return sourceLines.map((raw, index) => ({
		number: index + 1,
		raw,
		display: truncateLine(raw),
		indent: measureIndent(raw),
	}));
}

function computeEffectiveIndents(records: LineRecord[]): number[] {
	const out: number[] = [];
	let previous = 0;
	for (const record of records) {
		if (record.raw.trim().length === 0) {
			out.push(previous);
		} else {
			previous = record.indent;
			out.push(previous);
		}
	}
	return out;
}

function measureIndent(line: string): number {
	let total = 0;
	for (const ch of line) {
		if (ch === " ") {
			total += 1;
			continue;
		}
		if (ch === "\t") {
			total += TAB_WIDTH;
			continue;
		}
		break;
	}
	return total;
}

function isComment(line: string): boolean {
	const trimmed = line.trim();
	return COMMENT_PREFIXES.some((prefix) => trimmed.startsWith(prefix));
}

function trimEdgeBlankLines(records: LineRecord[]): void {
	while (records.length > 0 && records[0]!.raw.trim().length === 0) {
		records.shift();
	}
	while (records.length > 0 && records[records.length - 1]!.raw.trim().length === 0) {
		records.pop();
	}
}

function truncateLine(line: string): string {
	return safeSlice(line, MAX_LINE_LENGTH);
}

type DirEntryKind = "directory" | "file" | "symlink" | "other";

type DirEntry = {
	name: string;
	displayName: string;
	depth: number;
	kind: DirEntryKind;
};

async function listDirSlice(absolutePath: string, offset: number, limit: number, depth: number): Promise<string[]> {
	const entries: DirEntry[] = [];
	await collectEntries(absolutePath, "", depth, entries);

	if (entries.length === 0) {
		return [];
	}

	entries.sort((lhs, rhs) => lhs.name.localeCompare(rhs.name));

	const startIndex = offset - 1;
	if (startIndex >= entries.length) {
		throw new Error("offset exceeds directory entry count");
	}

	const remaining = entries.length - startIndex;
	const cappedLimit = Math.min(limit, remaining);
	const endIndex = startIndex + cappedLimit;
	const selected = entries.slice(startIndex, endIndex);
	const formatted = selected.map(formatDirEntryLine);

	if (endIndex < entries.length) {
		formatted.push(`More than ${cappedLimit} entries found`);
	}

	return formatted;
}

async function collectEntries(
	currentDir: string,
	relativePrefix: string,
	depth: number,
	out: DirEntry[],
): Promise<void> {
	const queue: Array<{ currentDir: string; relativePrefix: string; remainingDepth: number }> = [
		{ currentDir, relativePrefix, remainingDepth: depth },
	];

	while (queue.length > 0) {
		const next = queue.shift()!;
		const readDir = await fsp.readdir(next.currentDir, { withFileTypes: true }).catch((error: unknown) => {
			throw new Error(`failed to read directory: ${formatErrorMessage(error)}`);
		});

		const staged: Array<{
			absolute: string;
			relativePath: string;
			kind: DirEntryKind;
			entry: DirEntry;
		}> = [];

		for (const entry of readDir) {
			const absolute = path.join(next.currentDir, entry.name);
			const relativePath = next.relativePrefix.length === 0 ? entry.name : `${next.relativePrefix}/${entry.name}`;
			const depthLevel = next.relativePrefix.length === 0 ? 0 : next.relativePrefix.split("/").length;
			const kind: DirEntryKind = entry.isSymbolicLink()
				? "symlink"
				: entry.isDirectory()
					? "directory"
					: entry.isFile()
						? "file"
						: "other";

			staged.push({
				absolute,
				relativePath,
				kind,
				entry: {
					name: safeSlice(relativePath.replaceAll("\\", "/"), MAX_ENTRY_LENGTH),
					displayName: safeSlice(entry.name, MAX_ENTRY_LENGTH),
					depth: depthLevel,
					kind,
				},
			});
		}

		staged.sort((lhs, rhs) => lhs.entry.name.localeCompare(rhs.entry.name));

		for (const item of staged) {
			if (item.kind === "directory" && next.remainingDepth > 1) {
				queue.push({
					currentDir: item.absolute,
					relativePrefix: item.relativePath,
					remainingDepth: next.remainingDepth - 1,
				});
			}
			out.push(item.entry);
		}
	}
}

function formatDirEntryLine(entry: DirEntry): string {
	const indent = " ".repeat(entry.depth * 2);
	let name = entry.displayName;
	if (entry.kind === "directory") {
		name += "/";
	} else if (entry.kind === "symlink") {
		name += "@";
	} else if (entry.kind === "other") {
		name += "?";
	}
	return `${indent}${name}`;
}

async function waitOnTestSyncBarrier(args: { id: string; participants: number; timeout_ms?: number }): Promise<void> {
	const barrierId = `${args.id ?? ""}`.trim();
	if (barrierId.length === 0) {
		throw new Error("barrier id must not be empty");
	}

	const participants = Math.floor(args.participants);
	if (!Number.isFinite(participants) || participants <= 0) {
		throw new Error("barrier participants must be greater than zero");
	}

	const timeoutMs = positiveInt(args.timeout_ms, DEFAULT_TEST_SYNC_TIMEOUT_MS);
	if (timeoutMs <= 0) {
		throw new Error("barrier timeout must be greater than zero");
	}

	let state = testSyncBarriers.get(barrierId);
	if (!state) {
		state = {
			participants,
			waiters: [],
		};
		testSyncBarriers.set(barrierId, state);
	} else if (state.participants !== participants) {
		throw new Error(`barrier ${barrierId} already registered with ${state.participants} participants`);
	}

	const outcome = await new Promise<void>((resolve, reject) => {
		const waiter: TestSyncBarrierWaiter = {
			resolve: () => {
				clearTimeout(waiter.timer);
				resolve();
			},
			reject: (error: Error) => {
				clearTimeout(waiter.timer);
				reject(error);
			},
			timer: setTimeout(() => {
				const existing = testSyncBarriers.get(barrierId);
				if (existing) {
					existing.waiters = existing.waiters.filter((item) => item !== waiter);
				}
				reject(new Error("test_sync_tool barrier wait timed out"));
			}, timeoutMs),
		};

		state.waiters.push(waiter);

		if (state.waiters.length === state.participants) {
			const waiters = [...state.waiters];
			testSyncBarriers.delete(barrierId);
			for (const item of waiters) {
				item.resolve();
			}
		}
	});

	return outcome;
}

async function waitUntilYieldOrExit(session: ExecSession, yieldTimeMs: number): Promise<void> {
	if (session.exited) {
		return;
	}

	await new Promise<void>((resolve) => {
		const timeout = setTimeout(
			() => {
				cleanup();
				resolve();
			},
			Math.max(1, yieldTimeMs),
		);

		const waiter = () => {
			if (session.exited) {
				cleanup();
				resolve();
			}
		};

		function cleanup(): void {
			clearTimeout(timeout);
			session.waiters.delete(waiter);
		}

		session.waiters.add(waiter);
	});
}

function notifySession(session: ExecSession): void {
	for (const waiter of session.waiters) {
		waiter();
	}
}

async function waitForProcessExit(child: ReturnType<typeof spawn>, timeoutMs: number): Promise<[number, boolean]> {
	return await new Promise<[number, boolean]>((resolve) => {
		let done = false;
		const timer = setTimeout(
			() => {
				if (done) {
					return;
				}
				done = true;
				child.kill("SIGKILL");
				resolve([124, true]);
			},
			Math.max(1, timeoutMs),
		);

		child.on("close", (code: number | null) => {
			if (done) {
				return;
			}
			done = true;
			clearTimeout(timer);
			resolve([code ?? -1, false]);
		});

		child.on("error", () => {
			if (done) {
				return;
			}
			done = true;
			clearTimeout(timer);
			resolve([1, false]);
		});
	});
}

async function runShellCommandArray(
	command: string[],
	cwd: string,
	timeoutMs: number,
): Promise<{ content: string; durationMs: number; exitCode: number }> {
	const startedAt = Date.now();
	const child = spawn(command[0]!, command.slice(1), {
		cwd,
		stdio: ["ignore", "pipe", "pipe"],
		env: process.env,
		shell: false,
	});

	let aggregated = "";
	child.stdout?.on("data", (chunk: Buffer | string) => {
		aggregated += chunk.toString();
	});
	child.stderr?.on("data", (chunk: Buffer | string) => {
		aggregated += chunk.toString();
	});

	const [exitCode, timedOut] = await waitForProcessExit(child, timeoutMs);
	const durationMs = Date.now() - startedAt;
	const content = timedOut ? `command timed out after ${durationMs} milliseconds\n${aggregated}` : aggregated;
	return { content, durationMs, exitCode };
}

function positiveInt(value: number | undefined, fallback: number): number {
	if (value === undefined) {
		return fallback;
	}
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(0, Math.floor(value));
}

function roundToOneDecimal(value: number): number {
	return Math.round(value * 10) / 10;
}

function normalizeTtyLineEndings(text: string): string {
	return text.replaceAll("\r\n", "\n").replaceAll("\n", "\r\n");
}

function clamp(value: number, min: number, max: number): number {
	if (value < min) {
		return min;
	}
	if (value > max) {
		return max;
	}
	return value;
}

function clampYieldTime(value: number): number {
	return clamp(value, MIN_YIELD_TIME_MS, MAX_YIELD_TIME_MS);
}

function resolveMaxTokens(maxOutputTokens: number | undefined): number {
	if (maxOutputTokens === undefined) {
		return DEFAULT_MAX_OUTPUT_TOKENS;
	}
	if (!Number.isFinite(maxOutputTokens)) {
		return DEFAULT_MAX_OUTPUT_TOKENS;
	}
	return Math.max(0, Math.floor(maxOutputTokens));
}

function approxTokenCount(text: string): number {
	return approxTokensFromByteCount(Buffer.byteLength(text, "utf8"));
}

function approxBytesForTokens(tokens: number): number {
	return Math.max(0, tokens) * APPROX_BYTES_PER_TOKEN;
}

function approxTokensFromByteCount(bytes: number): number {
	if (bytes <= 0) {
		return 0;
	}
	return Math.floor((bytes + (APPROX_BYTES_PER_TOKEN - 1)) / APPROX_BYTES_PER_TOKEN);
}

function rustLineCount(text: string): number {
	if (text.length === 0) {
		return 0;
	}
	const normalized = text.replaceAll("\r\n", "\n");
	const parts = normalized.split("\n");
	if (parts.length > 0 && parts[parts.length - 1] === "") {
		parts.pop();
	}
	return parts.length;
}

function truncateTextByTokens(text: string, maxTokens: number): string {
	if (text.length === 0) {
		return "";
	}

	const budgetBytes = approxBytesForTokens(maxTokens);
	if (Buffer.byteLength(text, "utf8") <= budgetBytes) {
		return text;
	}

	if (budgetBytes === 0) {
		return `…${approxTokensFromByteCount(Buffer.byteLength(text, "utf8"))} tokens truncated…`;
	}

	const totalBytes = Buffer.byteLength(text, "utf8");
	const [leftBudget, rightBudget] = splitBudget(budgetBytes);
	const { removedChars, left, right } = splitStringForTruncation(text, leftBudget, rightBudget);
	const removedBytes = Math.max(0, totalBytes - budgetBytes);
	const marker = `…${approxTokensFromByteCount(removedBytes)} tokens truncated…`;
	if (removedChars === 0) {
		return text;
	}
	return `${left}${marker}${right}`;
}

function formattedTruncateText(text: string, maxTokens: number): string {
	if (Buffer.byteLength(text, "utf8") <= approxBytesForTokens(maxTokens)) {
		return text;
	}
	const totalLines = rustLineCount(text);
	const truncated = truncateTextByTokens(text, maxTokens);
	return `Total output lines: ${totalLines}\n\n${truncated}`;
}

function splitBudget(budget: number): [number, number] {
	const left = Math.floor(budget / 2);
	return [left, budget - left];
}

function splitStringForTruncation(
	text: string,
	beginningBytes: number,
	endBytes: number,
): { removedChars: number; left: string; right: string } {
	if (text.length === 0) {
		return { removedChars: 0, left: "", right: "" };
	}

	const totalBytes = Buffer.byteLength(text, "utf8");
	const tailStartTarget = Math.max(0, totalBytes - endBytes);
	let prefixEndByte = 0;
	let suffixStartByte = totalBytes;
	let removedChars = 0;
	let suffixStarted = false;
	let byteIdx = 0;

	for (const ch of text) {
		const charBytes = Buffer.byteLength(ch, "utf8");
		const charEnd = byteIdx + charBytes;
		if (charEnd <= beginningBytes) {
			prefixEndByte = charEnd;
			byteIdx = charEnd;
			continue;
		}

		if (byteIdx >= tailStartTarget) {
			if (!suffixStarted) {
				suffixStartByte = byteIdx;
				suffixStarted = true;
			}
			byteIdx = charEnd;
			continue;
		}

		removedChars += 1;
		byteIdx = charEnd;
	}

	if (suffixStartByte < prefixEndByte) {
		suffixStartByte = prefixEndByte;
	}

	const bytes = Buffer.from(text, "utf8");
	const left = bytes.subarray(0, prefixEndByte).toString("utf8");
	const right = bytes.subarray(suffixStartByte).toString("utf8");
	return { removedChars, left, right };
}

function safeSlice(value: string, maxBytes: number): string {
	if (Buffer.byteLength(value, "utf8") <= maxBytes) {
		return value;
	}

	let out = "";
	let used = 0;
	for (const ch of value) {
		const charBytes = Buffer.byteLength(ch, "utf8");
		if (used + charBytes > maxBytes) {
			break;
		}
		out += ch;
		used += charBytes;
	}
	return out;
}

function formatErrorMessage(error: unknown): string {
	if (error instanceof Error) {
		return error.message;
	}
	return String(error);
}

function generateChunkId(): string {
	const alphabet = "0123456789abcdef";
	let out = "";
	for (let i = 0; i < 6; i += 1) {
		out += alphabet[Math.floor(Math.random() * alphabet.length)];
	}
	return out;
}

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function assertType<T>(value: unknown): T {
	return value as T;
}

function exhaustive(value: never): never {
	throw new Error(`unreachable value: ${String(value)}`);
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export function createDefaultToolbox(options: CodexToolboxOptions = {}): CodexToolbox {
	return new CodexToolbox(options);
}

import path from "node:path";

export type JsonSchema =
	| {
			type: "boolean";
			description?: string;
	  }
	| {
			type: "string";
			description?: string;
	  }
	| {
			type: "number" | "integer";
			description?: string;
	  }
	| {
			type: "array";
			items: JsonSchema;
			description?: string;
	  }
	| {
			type: "object";
			properties: Record<string, JsonSchema>;
			required?: string[];
			additionalProperties?: boolean | JsonSchema;
	  };

export type FunctionToolSpec = {
	type: "function";
	name: string;
	description: string;
	strict: boolean;
	parameters: JsonSchema;
};

export type FreeformToolSpec = {
	type: "custom";
	name: string;
	description: string;
	format: {
		type: "grammar";
		syntax: "lark";
		definition: string;
	};
};

export type LocalShellToolSpec = {
	type: "local_shell";
};

export type WebSearchToolSpec = {
	type: "web_search";
	external_web_access?: boolean;
};

export type CodexToolSpec = FunctionToolSpec | FreeformToolSpec | LocalShellToolSpec | WebSearchToolSpec;

export type ShellToolKind = "shell" | "shell_command" | "local_shell" | "unified_exec" | "disabled";
export type ApplyPatchToolKind = "freeform" | "function";
export type WebSearchMode = "cached" | "live" | "disabled";
export type NamedToolsetId = "gpt-5.3-codex-family" | "gpt-5.2";

export interface NamedToolset {
	id: NamedToolsetId;
	models: readonly string[];
	tools: CodexToolSpec[];
}

export interface CodexToolSpecsBuildOptions {
	platform?: "windows" | "posix";
	includePrefixRule?: boolean;
	shellToolKind?: ShellToolKind;
	applyPatchToolKind?: ApplyPatchToolKind | null;
	webSearchMode?: WebSearchMode;
	includeMcpResourceTools?: boolean;
	includeCollabTools?: boolean;
	includeCollaborationModesTools?: boolean;
	includeJsRepl?: boolean;
	includeJsReplToolsOnly?: boolean;
	includeSearchToolBm25?: boolean;
	searchToolBm25AppNames?: string[];
	includeExperimentalTools?: Array<"grep_files" | "read_file" | "list_dir" | "test_sync_tool">;
	dynamicTools?: FunctionToolSpec[];
}

const APPLY_PATCH_LARK_GRAMMAR = `start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF`;

const SEARCH_TOOL_BM25_DESCRIPTION_TEMPLATE = `# Apps tool discovery

Searches over apps tool metadata with BM25 and exposes matching tools for the next model call.

MCP tools of the apps ({{app_names}}) are hidden until you search for them with this tool (\`search_tool_bm25\`).

Follow this workflow:

1. Call \`search_tool_bm25\` with:
   - \`query\` (required): focused terms that describe the capability you need.
   - \`limit\` (optional): maximum number of tools to return (default \`8\`).
2. Use the returned \`tools\` list to decide which Apps tools are relevant.
3. Matching tools are added to available \`tools\` and available for the remainder of the current session/thread.
4. Repeated searches in the same session/thread are additive: new matches are unioned into \`tools\`.

Notes:
- Core tools remain available without searching.
- If you are unsure, start with \`limit\` between 5 and 10 to see a broader set of tools.
- \`query\` is matched against Apps tool metadata fields:
  - \`name\`
  - \`tool_name\`
  - \`server_name\`
  - \`title\`
  - \`description\`
  - \`connector_name\`
  - input schema property keys (\`input_keys\`)
- If the needed app is already explicit in the prompt (for example an \`apps://...\` mention) or already present in the current \`tools\` list, you can call that tool directly.
- Do not use \`search_tool_bm25\` for non-apps/local tasks (filesystem, repo search, or shell-only workflows) or anything not related to {{app_names}}.`;

const SHELL_DESCRIPTION_WINDOWS = `Runs a Powershell command (Windows) and returns its output. Arguments to \`shell\` will be passed to CreateProcessW(). Most commands should be prefixed with ["powershell.exe", "-Command"].

Examples of valid command strings:

- ls -a (show hidden): ["powershell.exe", "-Command", "Get-ChildItem -Force"]
- recursive find by name: ["powershell.exe", "-Command", "Get-ChildItem -Recurse -Filter *.py"]
- recursive grep: ["powershell.exe", "-Command", "Get-ChildItem -Path C:\\\\myrepo -Recurse | Select-String -Pattern 'TODO' -CaseSensitive"]
- ps aux | grep python: ["powershell.exe", "-Command", "Get-Process | Where-Object { $_.ProcessName -like '*python*' }"]
- setting an env var: ["powershell.exe", "-Command", "$env:FOO='bar'; echo $env:FOO"]
- running an inline Python script: ["powershell.exe", "-Command", "@'\\nprint('Hello, world!')\\n'@ | python -"]`;

const SHELL_DESCRIPTION_POSIX = `Runs a shell command and returns its output.
- The arguments to \`shell\` will be passed to execvp(). Most terminal commands should be prefixed with ["bash", "-lc"].
- Always set the \`workdir\` param when using the shell function. Do not use \`cd\` unless absolutely necessary.`;

const SHELL_COMMAND_DESCRIPTION_WINDOWS = `Runs a Powershell command (Windows) and returns its output.

Examples of valid command strings:

- ls -a (show hidden): "Get-ChildItem -Force"
- recursive find by name: "Get-ChildItem -Recurse -Filter *.py"
- recursive grep: "Get-ChildItem -Path C:\\\\myrepo -Recurse | Select-String -Pattern 'TODO' -CaseSensitive"
- ps aux | grep python: "Get-Process | Where-Object { $_.ProcessName -like '*python*' }"
- setting an env var: "$env:FOO='bar'; echo $env:FOO"
- running an inline Python script: "@'\\nprint('Hello, world!')\\n'@ | python -"`;

const SHELL_COMMAND_DESCRIPTION_POSIX = `Runs a shell command and returns its output.
- Always set the \`workdir\` param when using the shell_command function. Do not use \`cd\` unless absolutely necessary.`;

const UPDATE_PLAN_DESCRIPTION = `Updates the task plan.
Provide an optional explanation and a list of plan items, each with a step and status.
At most one step can be in_progress at a time.
`;

const APPLY_PATCH_JSON_DESCRIPTION = `Use the \`apply_patch\` tool to edit files.
Your patch language is a stripped‑down, file‑oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high‑level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single \`@@\` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple \`@@\` statements to jump to the right context. For instance:

@@ class BaseClass
@@ \t def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with \`+\` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
`;

function str(description?: string): JsonSchema {
	return description ? { type: "string", description } : { type: "string" };
}

function num(description?: string): JsonSchema {
	return description ? { type: "number", description } : { type: "number" };
}

function bool(description?: string): JsonSchema {
	return description ? { type: "boolean", description } : { type: "boolean" };
}

function arr(items: JsonSchema, description?: string): JsonSchema {
	return description ? { type: "array", items, description } : { type: "array", items };
}

function obj(
	properties: Record<string, JsonSchema>,
	required?: string[],
	additionalProperties?: boolean | JsonSchema,
): JsonSchema {
	const schema: JsonSchema = { type: "object", properties };
	if (required && required.length > 0) {
		schema.required = required;
	}
	if (additionalProperties !== undefined) {
		schema.additionalProperties = additionalProperties;
	}
	return schema;
}

function fnSpec(name: string, description: string, parameters: JsonSchema, required?: string[]): FunctionToolSpec {
	if (parameters.type !== "object") {
		throw new Error(`parameters for ${name} must be object schema`);
	}
	const normalized = obj(parameters.properties, required ?? parameters.required, false);
	return {
		type: "function",
		name,
		description,
		strict: false,
		parameters: normalized,
	};
}

function createApprovalParameters(includePrefixRule: boolean): Record<string, JsonSchema> {
	const properties: Record<string, JsonSchema> = {
		sandbox_permissions: str(
			'Sandbox permissions for the command. Set to "require_escalated" to request running without sandbox restrictions; defaults to "use_default".',
		),
		justification: str(`Only set if sandbox_permissions is \\"require_escalated\\".
                    Request approval from the user to run this command outside the sandbox.
                    Phrased as a simple question that summarizes the purpose of the
                    command as it relates to the task at hand - e.g. 'Do you want to
                    fetch and pull the latest version of this git branch?'`),
	};

	if (includePrefixRule) {
		properties.prefix_rule = arr(
			str(),
			'Only specify when sandbox_permissions is `require_escalated`.\n                    Suggest a prefix command pattern that will allow you to fulfill similar requests from the user in the future.\n                    Should be a short but reasonable prefix, e.g. [\\"git\\", \\"pull\\"] or [\\"uv\\", \\"run\\"] or [\\"pytest\\"].',
		);
	}

	return properties;
}

export function createExecCommandToolSpec(includePrefixRule = true): FunctionToolSpec {
	const properties: Record<string, JsonSchema> = {
		cmd: str("Shell command to execute."),
		workdir: str("Optional working directory to run the command in; defaults to the turn cwd."),
		shell: str("Shell binary to launch. Defaults to the user's default shell."),
		login: bool("Whether to run the shell with -l/-i semantics. Defaults to true."),
		tty: bool(
			"Whether to allocate a TTY for the command. Defaults to false (plain pipes); set to true to open a PTY and access TTY process.",
		),
		yield_time_ms: num("How long to wait (in milliseconds) for output before yielding."),
		max_output_tokens: num("Maximum number of tokens to return. Excess output will be truncated."),
		...createApprovalParameters(includePrefixRule),
	};

	return fnSpec(
		"exec_command",
		"Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
		obj(properties),
		["cmd"],
	);
}

export function createWriteStdinToolSpec(): FunctionToolSpec {
	const properties: Record<string, JsonSchema> = {
		session_id: num("Identifier of the running unified exec session."),
		chars: str("Bytes to write to stdin (may be empty to poll)."),
		yield_time_ms: num("How long to wait (in milliseconds) for output before yielding."),
		max_output_tokens: num("Maximum number of tokens to return. Excess output will be truncated."),
	};

	return fnSpec(
		"write_stdin",
		"Writes characters to an existing unified exec session and returns recent output.",
		obj(properties),
		["session_id"],
	);
}

export function createShellToolSpec(platform: "windows" | "posix", includePrefixRule = true): FunctionToolSpec {
	const properties: Record<string, JsonSchema> = {
		command: arr(str(), "The command to execute"),
		workdir: str("The working directory to execute the command in"),
		timeout_ms: num("The timeout for the command in milliseconds"),
		...createApprovalParameters(includePrefixRule),
	};

	return fnSpec(
		"shell",
		platform === "windows" ? SHELL_DESCRIPTION_WINDOWS : SHELL_DESCRIPTION_POSIX,
		obj(properties),
		["command"],
	);
}

export function createShellCommandToolSpec(platform: "windows" | "posix", includePrefixRule = true): FunctionToolSpec {
	const properties: Record<string, JsonSchema> = {
		command: str("The shell script to execute in the user's default shell"),
		workdir: str("The working directory to execute the command in"),
		login: bool("Whether to run the shell with login shell semantics. Defaults to true."),
		timeout_ms: num("The timeout for the command in milliseconds"),
		...createApprovalParameters(includePrefixRule),
	};

	return fnSpec(
		"shell_command",
		platform === "windows" ? SHELL_COMMAND_DESCRIPTION_WINDOWS : SHELL_COMMAND_DESCRIPTION_POSIX,
		obj(properties),
		["command"],
	);
}

export function createViewImageToolSpec(): FunctionToolSpec {
	return fnSpec(
		"view_image",
		"View a local image from the filesystem (only use if given a full filepath by the user, and the image isn't already attached to the thread context within <image ...> tags).",
		obj({
			path: str("Local filesystem path to an image file"),
		}),
		["path"],
	);
}

function createCollabInputItemsSchema(): JsonSchema {
	return arr(
		obj(
			{
				type: str("Input item type: text, image, local_image, skill, or mention."),
				text: str("Text content when type is text."),
				image_url: str("Image URL when type is image."),
				path: str(
					"Path when type is local_image/skill, or mention target such as app://<connector-id> when type is mention.",
				),
				name: str("Display name when type is skill or mention."),
			},
			undefined,
			false,
		),
		"Structured input items. Use this to pass explicit mentions (for example app:// connector paths).",
	);
}

export function createSpawnAgentToolSpec(): FunctionToolSpec {
	return fnSpec(
		"spawn_agent",
		"Spawn a sub-agent for a well-scoped task. Returns the agent id to use to communicate with this agent.",
		obj({
			message: str("Initial plain-text task for the new agent. Use either message or items."),
			items: createCollabInputItemsSchema(),
			agent_type: str(
				`Optional type name for the new agent. If omitted, \`default\` is used.
Available roles:
default: {
Default agent.
}
explorer: {
Use \`explorer\` for all codebase questions.
Explorers are fast and authoritative.
Always prefer them over manual search or file reading.
Rules:
- Ask explorers first and precisely.
- Do not re-read or re-search code they cover.
- Trust explorer results without verification.
- Run explorers in parallel when useful.
- Reuse existing explorers for related questions.
}
worker: {
Use for execution and production work.
Typical tasks:
- Implement part of a feature
- Fix tests or bugs
- Split large refactors into independent chunks
Rules:
- Explicitly assign **ownership** of the task (files / responsibility).
- Always tell workers they are **not alone in the codebase**, and they should ignore edits made by others without touching them.
}
            `,
			),
		}),
	);
}

export function createSendInputToolSpec(): FunctionToolSpec {
	return fnSpec(
		"send_input",
		"Send a message to an existing agent. Use interrupt=true to redirect work immediately.",
		obj({
			id: str("Agent id to message (from spawn_agent)."),
			message: str("Legacy plain-text message to send to the agent. Use either message or items."),
			items: createCollabInputItemsSchema(),
			interrupt: bool(
				"When true, stop the agent's current task and handle this immediately. When false (default), queue this message.",
			),
		}),
		["id"],
	);
}

export function createResumeAgentToolSpec(): FunctionToolSpec {
	return fnSpec(
		"resume_agent",
		"Resume a previously closed agent by id so it can receive send_input and wait calls.",
		obj({ id: str("Agent id to resume.") }),
		["id"],
	);
}

const DEFAULT_WAIT_TIMEOUT_MS = 30_000;
const MIN_WAIT_TIMEOUT_MS = 10_000;
const MAX_WAIT_TIMEOUT_MS = 300_000;

export function createWaitToolSpec(): FunctionToolSpec {
	return fnSpec(
		"wait",
		"Wait for agents to reach a final status. Completed statuses may include the agent's final message. Returns empty status when timed out.",
		obj({
			ids: arr(str(), "Agent ids to wait on. Pass multiple ids to wait for whichever finishes first."),
			timeout_ms: num(
				`Optional timeout in milliseconds. Defaults to ${DEFAULT_WAIT_TIMEOUT_MS}, min ${MIN_WAIT_TIMEOUT_MS}, max ${MAX_WAIT_TIMEOUT_MS}. Prefer longer waits (minutes) to avoid busy polling.`,
			),
		}),
		["ids"],
	);
}

export function requestUserInputToolDescription(allowedModes = "Plan mode"): string {
	return `Request user input for one to three short questions and wait for the response. This tool is only available in ${allowedModes}.`;
}

export function createRequestUserInputToolSpec(allowedModes = "Plan mode"): FunctionToolSpec {
	const optionSchema = obj(
		{
			label: str("User-facing label (1-5 words)."),
			description: str("One short sentence explaining impact/tradeoff if selected."),
		},
		["label", "description"],
		false,
	);

	const questionSchema = obj(
		{
			id: str("Stable identifier for mapping answers (snake_case)."),
			header: str("Short header label shown in the UI (12 or fewer chars)."),
			question: str("Single-sentence prompt shown to the user."),
			options: arr(
				optionSchema,
				'Provide 2-3 mutually exclusive choices. Put the recommended option first and suffix its label with "(Recommended)". Do not include an "Other" option in this list; the client will add a free-form "Other" option automatically.',
			),
		},
		["id", "header", "question", "options"],
		false,
	);

	return fnSpec(
		"request_user_input",
		requestUserInputToolDescription(allowedModes),
		obj(
			{
				questions: arr(questionSchema, "Questions to show the user. Prefer 1 and do not exceed 3"),
			},
			["questions"],
			false,
		),
		["questions"],
	);
}

export function createCloseAgentToolSpec(): FunctionToolSpec {
	return fnSpec(
		"close_agent",
		"Close an agent when it is no longer needed and return its last known status.",
		obj({ id: str("Agent id to close (from spawn_agent).") }),
		["id"],
	);
}

export function createTestSyncToolSpec(): FunctionToolSpec {
	return fnSpec(
		"test_sync_tool",
		"Internal synchronization helper used by Codex integration tests.",
		obj({
			sleep_before_ms: num("Optional delay in milliseconds before any other action"),
			sleep_after_ms: num("Optional delay in milliseconds after completing the barrier"),
			barrier: obj(
				{
					id: str("Identifier shared by concurrent calls that should rendezvous"),
					participants: num("Number of tool calls that must arrive before the barrier opens"),
					timeout_ms: num("Maximum time in milliseconds to wait at the barrier"),
				},
				["id", "participants"],
				false,
			),
		}),
	);
}

export function createGrepFilesToolSpec(): FunctionToolSpec {
	return fnSpec(
		"grep_files",
		"Finds files whose contents match the pattern and lists them by modification time.",
		obj({
			pattern: str("Regular expression pattern to search for."),
			include: str('Optional glob that limits which files are searched (e.g. "*.rs" or "*.{ts,tsx}").'),
			path: str("Directory or file path to search. Defaults to the session's working directory."),
			limit: num("Maximum number of file paths to return (defaults to 100)."),
		}),
		["pattern"],
	);
}

export function createSearchToolBm25ToolSpec(appNames: string[]): FunctionToolSpec {
	const names = [...new Set(appNames)].sort().join(", ");
	return fnSpec(
		"search_tool_bm25",
		SEARCH_TOOL_BM25_DESCRIPTION_TEMPLATE.replaceAll("{{app_names}}", names),
		obj({
			query: str("Search query for apps tools."),
			limit: num("Maximum number of tools to return (defaults to 8)."),
		}),
		["query"],
	);
}

export function createReadFileToolSpec(): FunctionToolSpec {
	return fnSpec(
		"read_file",
		"Reads a local file with 1-indexed line numbers, supporting slice and indentation-aware block modes.",
		obj({
			file_path: str("Absolute path to the file"),
			offset: num("The line number to start reading from. Must be 1 or greater."),
			limit: num("The maximum number of lines to return."),
			mode: str(
				'Optional mode selector: "slice" for simple ranges (default) or "indentation" to expand around an anchor line.',
			),
			indentation: obj(
				{
					anchor_line: num("Anchor line to center the indentation lookup on (defaults to offset)."),
					max_levels: num("How many parent indentation levels (smaller indents) to include."),
					include_siblings: bool("When true, include additional blocks that share the anchor indentation."),
					include_header: bool("Include doc comments or attributes directly above the selected block."),
					max_lines: num("Hard cap on the number of lines returned when using indentation mode."),
				},
				undefined,
				false,
			),
		}),
		["file_path"],
	);
}

export function createListDirToolSpec(): FunctionToolSpec {
	return fnSpec(
		"list_dir",
		"Lists entries in a local directory with 1-indexed entry numbers and simple type labels.",
		obj({
			dir_path: str("Absolute path to the directory to list."),
			offset: num("The entry number to start listing from. Must be 1 or greater."),
			limit: num("The maximum number of entries to return."),
			depth: num("The maximum directory depth to traverse. Must be 1 or greater."),
		}),
		["dir_path"],
	);
}

export function createJsReplToolSpec(): FreeformToolSpec {
	return {
		type: "custom",
		name: "js_repl",
		description:
			"Runs JavaScript in a persistent Node kernel with top-level await. This is a freeform tool: send raw JavaScript source text, optionally with a first-line pragma like `// codex-js-repl: timeout_ms=15000`; do not send JSON/quotes/markdown fences.",
		format: {
			type: "grammar",
			syntax: "lark",
			definition: "start: /[\\s\\S]*/",
		},
	};
}

export function createJsReplResetToolSpec(): FunctionToolSpec {
	return fnSpec(
		"js_repl_reset",
		"Restarts the js_repl kernel for this run and clears persisted top-level bindings.",
		obj({}),
	);
}

export function createListMcpResourcesToolSpec(): FunctionToolSpec {
	return fnSpec(
		"list_mcp_resources",
		"Lists resources provided by MCP servers. Resources allow servers to share data that provides context to language models, such as files, database schemas, or application-specific information. Prefer resources over web search when possible.",
		obj({
			server: str("Optional MCP server name. When omitted, lists resources from every configured server."),
			cursor: str("Opaque cursor returned by a previous list_mcp_resources call for the same server."),
		}),
	);
}

export function createListMcpResourceTemplatesToolSpec(): FunctionToolSpec {
	return fnSpec(
		"list_mcp_resource_templates",
		"Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to share data that takes parameters and provides context to language models, such as files, database schemas, or application-specific information. Prefer resource templates over web search when possible.",
		obj({
			server: str("Optional MCP server name. When omitted, lists resource templates from all configured servers."),
			cursor: str("Opaque cursor returned by a previous list_mcp_resource_templates call for the same server."),
		}),
	);
}

export function createReadMcpResourceToolSpec(): FunctionToolSpec {
	return fnSpec(
		"read_mcp_resource",
		"Read a specific resource from an MCP server given the server name and resource URI.",
		obj({
			server: str(
				"MCP server name exactly as configured. Must match the 'server' field returned by list_mcp_resources.",
			),
			uri: str("Resource URI to read. Must be one of the URIs returned by list_mcp_resources."),
		}),
		["server", "uri"],
	);
}

export function createApplyPatchFreeformToolSpec(): FreeformToolSpec {
	return {
		type: "custom",
		name: "apply_patch",
		description:
			"Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
		format: {
			type: "grammar",
			syntax: "lark",
			definition: `${APPLY_PATCH_LARK_GRAMMAR}\n`,
		},
	};
}

export function createApplyPatchJsonToolSpec(): FunctionToolSpec {
	return fnSpec(
		"apply_patch",
		APPLY_PATCH_JSON_DESCRIPTION,
		obj({
			input: str("The entire contents of the apply_patch command"),
		}),
		["input"],
	);
}

export function createUpdatePlanToolSpec(): FunctionToolSpec {
	return fnSpec(
		"update_plan",
		UPDATE_PLAN_DESCRIPTION,
		obj(
			{
				explanation: str(),
				plan: arr(
					obj(
						{
							step: str(),
							status: str("One of: pending, in_progress, completed"),
						},
						["step", "status"],
						false,
					),
					"The list of steps",
				),
			},
			["plan"],
			false,
		),
		["plan"],
	);
}

export function createWebSearchToolSpec(mode: Exclude<WebSearchMode, "disabled">): WebSearchToolSpec {
	return {
		type: "web_search",
		external_web_access: mode === "live",
	};
}

export function createLocalShellToolSpec(): LocalShellToolSpec {
	return { type: "local_shell" };
}

export function detectHostPlatform(): "windows" | "posix" {
	return process.platform === "win32" ? "windows" : "posix";
}

export function buildCodexToolSpecs(options: CodexToolSpecsBuildOptions = {}): CodexToolSpec[] {
	const platform = options.platform ?? detectHostPlatform();
	const includePrefixRule = options.includePrefixRule ?? true;
	const shellToolKind = options.shellToolKind ?? "shell_command";
	const applyPatchToolKind = options.applyPatchToolKind === undefined ? "freeform" : options.applyPatchToolKind;
	const webSearchMode = options.webSearchMode ?? "disabled";
	const includeMcpResourceTools = options.includeMcpResourceTools ?? false;
	const includeCollabTools = options.includeCollabTools ?? false;
	const includeCollaborationModesTools = options.includeCollaborationModesTools ?? false;
	const includeJsRepl = options.includeJsRepl ?? false;
	const includeJsReplToolsOnly = options.includeJsReplToolsOnly ?? false;
	const includeSearchToolBm25 = options.includeSearchToolBm25 ?? false;
	const includeExperimentalTools = new Set(options.includeExperimentalTools ?? []);
	const appNames = options.searchToolBm25AppNames ?? [];

	const tools: CodexToolSpec[] = [];

	if (shellToolKind === "shell") {
		tools.push(createShellToolSpec(platform, includePrefixRule));
	} else if (shellToolKind === "shell_command") {
		tools.push(createShellCommandToolSpec(platform, includePrefixRule));
	} else if (shellToolKind === "local_shell") {
		tools.push(createLocalShellToolSpec());
	} else if (shellToolKind === "unified_exec") {
		tools.push(createExecCommandToolSpec(includePrefixRule));
		tools.push(createWriteStdinToolSpec());
	}

	if (includeMcpResourceTools) {
		tools.push(createListMcpResourcesToolSpec());
		tools.push(createListMcpResourceTemplatesToolSpec());
		tools.push(createReadMcpResourceToolSpec());
	}

	tools.push(createUpdatePlanToolSpec());

	if (includeJsRepl) {
		tools.push(createJsReplToolSpec());
		tools.push(createJsReplResetToolSpec());
	}

	if (includeCollaborationModesTools) {
		tools.push(createRequestUserInputToolSpec("Plan mode"));
	}

	if (includeSearchToolBm25) {
		tools.push(createSearchToolBm25ToolSpec(appNames));
	}

	if (applyPatchToolKind === "freeform") {
		tools.push(createApplyPatchFreeformToolSpec());
	} else if (applyPatchToolKind === "function") {
		tools.push(createApplyPatchJsonToolSpec());
	}

	if (includeExperimentalTools.has("grep_files")) {
		tools.push(createGrepFilesToolSpec());
	}
	if (includeExperimentalTools.has("read_file")) {
		tools.push(createReadFileToolSpec());
	}
	if (includeExperimentalTools.has("list_dir")) {
		tools.push(createListDirToolSpec());
	}
	if (includeExperimentalTools.has("test_sync_tool")) {
		tools.push(createTestSyncToolSpec());
	}

	if (webSearchMode === "cached") {
		tools.push(createWebSearchToolSpec("cached"));
	} else if (webSearchMode === "live") {
		tools.push(createWebSearchToolSpec("live"));
	}

	tools.push(createViewImageToolSpec());

	if (includeCollabTools) {
		tools.push(createSpawnAgentToolSpec());
		tools.push(createSendInputToolSpec());
		tools.push(createResumeAgentToolSpec());
		tools.push(createWaitToolSpec());
		tools.push(createCloseAgentToolSpec());
	}

	if (options.dynamicTools && !includeJsReplToolsOnly) {
		const sorted = [...options.dynamicTools].sort((lhs, rhs) => lhs.name.localeCompare(rhs.name));
		tools.push(...sorted);
	}

	if (includeJsReplToolsOnly) {
		return tools.filter((tool) => {
			if (tool.type === "custom") {
				return tool.name === "js_repl";
			}
			if (tool.type === "function") {
				return tool.name === "js_repl_reset";
			}
			return false;
		});
	}

	return tools;
}

export function buildAllBuiltinToolVariants(platform: "windows" | "posix" = detectHostPlatform()): CodexToolSpec[] {
	const specs: CodexToolSpec[] = [
		createExecCommandToolSpec(true),
		createWriteStdinToolSpec(),
		createShellToolSpec(platform, true),
		createShellCommandToolSpec(platform, true),
		createLocalShellToolSpec(),
		createViewImageToolSpec(),
		createSpawnAgentToolSpec(),
		createSendInputToolSpec(),
		createResumeAgentToolSpec(),
		createWaitToolSpec(),
		createRequestUserInputToolSpec("Plan mode"),
		createCloseAgentToolSpec(),
		createTestSyncToolSpec(),
		createGrepFilesToolSpec(),
		createSearchToolBm25ToolSpec([]),
		createReadFileToolSpec(),
		createListDirToolSpec(),
		createJsReplToolSpec(),
		createJsReplResetToolSpec(),
		createListMcpResourcesToolSpec(),
		createListMcpResourceTemplatesToolSpec(),
		createReadMcpResourceToolSpec(),
		createApplyPatchFreeformToolSpec(),
		createApplyPatchJsonToolSpec(),
		createUpdatePlanToolSpec(),
		createWebSearchToolSpec("cached"),
		createWebSearchToolSpec("live"),
	];

	return specs;
}

export function findFunctionToolByName(specs: CodexToolSpec[], name: string): FunctionToolSpec | undefined {
	return specs.find((tool): tool is FunctionToolSpec => tool.type === "function" && tool.name === name);
}

export function resolveShellToolKindFromModelSlug(model: string): ShellToolKind {
	const normalized = path.basename(model).toLowerCase();
	if (normalized === "gpt-5") {
		return "shell";
	}
	return "shell_command";
}

export const GPT_5_3_CODEX_FAMILY_MODELS = ["gpt-5.3-codex", "gpt-5.3-codex-spark"] as const;
export const GPT_5_2_MODELS = ["gpt-5.2"] as const;

function buildGpt5CodexStyleToolset(platform: "windows" | "posix"): CodexToolSpec[] {
	return buildCodexToolSpecs({
		platform,
		shellToolKind: "unified_exec",
		includePrefixRule: true,
		applyPatchToolKind: "freeform",
		webSearchMode: "live",
		includeMcpResourceTools: false,
		includeCollabTools: false,
		includeCollaborationModesTools: true,
		includeJsRepl: false,
		includeJsReplToolsOnly: false,
		includeSearchToolBm25: false,
		includeExperimentalTools: [],
	});
}

export function buildGpt53CodexFamilyToolset(platform: "windows" | "posix" = detectHostPlatform()): CodexToolSpec[] {
	return buildGpt5CodexStyleToolset(platform);
}

export function buildGpt52Toolset(platform: "windows" | "posix" = detectHostPlatform()): CodexToolSpec[] {
	return buildGpt5CodexStyleToolset(platform);
}

export function buildNamedToolsets(
	platform: "windows" | "posix" = detectHostPlatform(),
): Record<NamedToolsetId, NamedToolset> {
	return {
		"gpt-5.3-codex-family": {
			id: "gpt-5.3-codex-family",
			models: GPT_5_3_CODEX_FAMILY_MODELS,
			tools: buildGpt53CodexFamilyToolset(platform),
		},
		"gpt-5.2": {
			id: "gpt-5.2",
			models: GPT_5_2_MODELS,
			tools: buildGpt52Toolset(platform),
		},
	};
}

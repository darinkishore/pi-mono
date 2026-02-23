import type { AgentTool } from "@mariozechner/pi-agent-core";
import type { TextContent } from "@mariozechner/pi-ai";
import { createApplyPatchTool } from "../tools/apply-patch.js";
import {
	createExecCommandToolSpec,
	createGrepFilesToolSpec,
	createListDirToolSpec,
	createReadFileToolSpec,
	createShellCommandToolSpec,
	createShellToolSpec,
	createTestSyncToolSpec,
	createViewImageToolSpec,
	createWriteStdinToolSpec,
	detectHostPlatform,
	type FunctionToolSpec,
} from "./tool-specs.js";
import { CodexToolbox, type CodexToolName } from "./toolbox.js";

export type CodexLocalToolName =
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
	| "test_sync_tool";

export const CODEX_LOCAL_TOOL_NAMES: ReadonlyArray<CodexLocalToolName> = [
	"apply_patch",
	"exec_command",
	"write_stdin",
	"shell",
	"local_shell",
	"container.exec",
	"shell_command",
	"grep_files",
	"read_file",
	"list_dir",
	"view_image",
	"test_sync_tool",
] as const;

function formatToolResult(result: unknown): string {
	if (typeof result === "string") return result;
	if (result === null || result === undefined) return "";
	if (result instanceof Error) return result.message;
	try {
		return JSON.stringify(result, null, 2);
	} catch {
		return String(result);
	}
}

function extractTextFromContent(content: Array<TextContent | { type: string; text?: string }>): string {
	return content
		.filter((block): block is TextContent => block.type === "text")
		.map((block) => block.text)
		.join("\n");
}

function createCodexFunctionTool(
	spec: FunctionToolSpec,
	toolbox: CodexToolbox,
	dispatchName: CodexToolName,
): AgentTool<any> {
	return {
		name: spec.name,
		label: spec.name,
		description: spec.description,
		parameters: spec.parameters as any,
		execute: async (_toolCallId, params) => {
			const result = await toolbox.dispatch(dispatchName, params);
			const text = formatToolResult(result);
			return {
				content: [{ type: "text", text: text || "(no output)" }],
				details: {},
			};
		},
	};
}

function createShellAliasTool(
	name: "local_shell" | "container.exec",
	shellSpec: FunctionToolSpec,
	toolbox: CodexToolbox,
): AgentTool<any> {
	const aliasedSpec: FunctionToolSpec = {
		...shellSpec,
		name,
		description: `${shellSpec.description}\n\nAlias that routes to the same shell executor.`,
	};
	return createCodexFunctionTool(aliasedSpec, toolbox, name as CodexToolName);
}

export function createCodexLocalTools(cwd: string): Record<string, AgentTool> {
	const platform = detectHostPlatform();
	const applyPatchTool = createApplyPatchTool(cwd);
	const toolbox = new CodexToolbox({
		cwd,
		applyPatch: async (patch: string) => {
			const result = await applyPatchTool.execute("codex_apply_patch", patch);
			return extractTextFromContent(result.content);
		},
	});

	const execCommandSpec = createExecCommandToolSpec(true);
	const writeStdinSpec = createWriteStdinToolSpec();
	const shellSpec = createShellToolSpec(platform, true);
	const shellCommandSpec = createShellCommandToolSpec(platform, true);
	const grepFilesSpec = createGrepFilesToolSpec();
	const readFileSpec = createReadFileToolSpec();
	const listDirSpec = createListDirToolSpec();
	const viewImageSpec = createViewImageToolSpec();
	const testSyncSpec = createTestSyncToolSpec();

	return {
		apply_patch: applyPatchTool as unknown as AgentTool,
		exec_command: createCodexFunctionTool(execCommandSpec, toolbox, "exec_command"),
		write_stdin: createCodexFunctionTool(writeStdinSpec, toolbox, "write_stdin"),
		shell: createCodexFunctionTool(shellSpec, toolbox, "shell"),
		local_shell: createShellAliasTool("local_shell", shellSpec, toolbox),
		"container.exec": createShellAliasTool("container.exec", shellSpec, toolbox),
		shell_command: createCodexFunctionTool(shellCommandSpec, toolbox, "shell_command"),
		grep_files: createCodexFunctionTool(grepFilesSpec, toolbox, "grep_files"),
		read_file: createCodexFunctionTool(readFileSpec, toolbox, "read_file"),
		list_dir: createCodexFunctionTool(listDirSpec, toolbox, "list_dir"),
		view_image: createCodexFunctionTool(viewImageSpec, toolbox, "view_image"),
		test_sync_tool: createCodexFunctionTool(testSyncSpec, toolbox, "test_sync_tool"),
	};
}

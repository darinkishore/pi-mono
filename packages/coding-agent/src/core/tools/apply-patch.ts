import { constants } from "node:fs";
import {
	access as fsAccess,
	mkdir as fsMkdir,
	readFile as fsReadFile,
	rm as fsRm,
	writeFile as fsWriteFile,
} from "node:fs/promises";
import { dirname } from "node:path";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { type Static, Type } from "@sinclair/typebox";
import * as Diff from "diff";
import { detectLineEnding, normalizeToLF, restoreLineEndings, stripBom } from "./edit-diff.js";
import { resolveToCwd } from "./path-utils.js";

const APPLY_PATCH_GRAMMAR = `start: begin_patch hunk+ end_patch
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

const applyPatchSchema = Type.Union([
	Type.String({ description: "Patch text in apply_patch grammar format." }),
	Type.Object({
		patch: Type.String({ description: "Patch text in apply_patch grammar format." }),
	}),
]);

export type ApplyPatchToolInput = Static<typeof applyPatchSchema>;

export interface ApplyPatchToolDetails {
	/** Combined unified diff for all file changes. */
	diff: string;
	/** Files touched by the patch. */
	changedFiles: string[];
	/** First changed line in the first modified file (best effort). */
	firstChangedLine?: number;
}

interface PatchLine {
	type: "context" | "add" | "remove";
	text: string;
}

interface PatchHunk {
	lines: PatchLine[];
}

interface AddFileOperation {
	type: "add";
	path: string;
	lines: string[];
}

interface DeleteFileOperation {
	type: "delete";
	path: string;
}

interface UpdateFileOperation {
	type: "update";
	path: string;
	moveTo?: string;
	hunks: PatchHunk[];
}

type PatchOperation = AddFileOperation | DeleteFileOperation | UpdateFileOperation;

interface ParsedPatch {
	operations: PatchOperation[];
}

const BEGIN_PATCH = "*** Begin Patch";
const END_PATCH = "*** End Patch";
const ADD_FILE_PREFIX = "*** Add File: ";
const DELETE_FILE_PREFIX = "*** Delete File: ";
const UPDATE_FILE_PREFIX = "*** Update File: ";
const MOVE_TO_PREFIX = "*** Move to: ";
const END_OF_FILE_MARKER = "*** End of File";

export interface ApplyPatchOperations {
	readFile: (absolutePath: string) => Promise<string>;
	writeFile: (absolutePath: string, content: string) => Promise<void>;
	deleteFile: (absolutePath: string) => Promise<void>;
	fileExists: (absolutePath: string) => Promise<boolean>;
	ensureParentDir: (absolutePath: string) => Promise<void>;
}

const defaultApplyPatchOperations: ApplyPatchOperations = {
	readFile: (path) => fsReadFile(path, "utf-8"),
	writeFile: (path, content) => fsWriteFile(path, content, "utf-8"),
	deleteFile: (path) => fsRm(path),
	fileExists: async (path) => {
		try {
			await fsAccess(path, constants.F_OK);
			return true;
		} catch {
			return false;
		}
	},
	ensureParentDir: async (path) => {
		await fsMkdir(dirname(path), { recursive: true });
	},
};

export interface ApplyPatchToolOptions {
	operations?: ApplyPatchOperations;
}

function parseError(message: string, lineIndex: number): never {
	throw new Error(`${message} (line ${lineIndex + 1})`);
}

function parseRequiredPath(line: string, prefix: string, lineIndex: number): string {
	const path = line.slice(prefix.length).trim();
	if (!path) {
		parseError(`Missing path for "${prefix.trim()}"`, lineIndex);
	}
	return path;
}

function isOperationStart(line: string): boolean {
	return (
		line.startsWith(ADD_FILE_PREFIX) || line.startsWith(DELETE_FILE_PREFIX) || line.startsWith(UPDATE_FILE_PREFIX)
	);
}

function parsePatch(patchText: string): ParsedPatch {
	const lines = normalizeToLF(patchText).split("\n");
	if (lines.length === 0 || lines[0] !== BEGIN_PATCH) {
		throw new Error('Patch must start with "*** Begin Patch"');
	}

	const operations: PatchOperation[] = [];
	let i = 1;
	let sawEndPatch = false;

	while (i < lines.length) {
		const line = lines[i] ?? "";

		if (line === END_PATCH) {
			sawEndPatch = true;
			i++;
			break;
		}
		if (!line.trim()) {
			i++;
			continue;
		}

		if (line.startsWith(ADD_FILE_PREFIX)) {
			const path = parseRequiredPath(line, ADD_FILE_PREFIX, i);
			i++;
			const addLines: string[] = [];
			while (i < lines.length) {
				const next = lines[i] ?? "";
				if (next === END_PATCH || isOperationStart(next)) break;
				if (!next.startsWith("+")) {
					parseError('Expected "+" line in add file section', i);
				}
				addLines.push(next.slice(1));
				i++;
			}
			if (addLines.length === 0) {
				parseError("Add file section must contain at least one + line", i - 1);
			}
			operations.push({ type: "add", path, lines: addLines });
			continue;
		}

		if (line.startsWith(DELETE_FILE_PREFIX)) {
			const path = parseRequiredPath(line, DELETE_FILE_PREFIX, i);
			operations.push({ type: "delete", path });
			i++;
			continue;
		}

		if (line.startsWith(UPDATE_FILE_PREFIX)) {
			const path = parseRequiredPath(line, UPDATE_FILE_PREFIX, i);
			i++;

			let moveTo: string | undefined;
			if (i < lines.length && lines[i]?.startsWith(MOVE_TO_PREFIX)) {
				moveTo = parseRequiredPath(lines[i]!, MOVE_TO_PREFIX, i);
				i++;
			}

			const hunks: PatchHunk[] = [];
			let currentHunk: PatchHunk | null = null;

			while (i < lines.length) {
				const next = lines[i] ?? "";
				if (next === END_PATCH || isOperationStart(next)) break;

				if (next.startsWith("@@")) {
					if (currentHunk) {
						hunks.push(currentHunk);
					}
					currentHunk = { lines: [] };
					i++;
					continue;
				}

				if (next === END_OF_FILE_MARKER) {
					i++;
					continue;
				}

				if (!currentHunk) {
					parseError('Expected "@@" before patch lines in update section', i);
				}

				const prefix = next[0];
				if (prefix !== " " && prefix !== "+" && prefix !== "-") {
					parseError('Expected line prefix " ", "+", or "-" in update section', i);
				}

				currentHunk.lines.push({
					type: prefix === " " ? "context" : prefix === "+" ? "add" : "remove",
					text: next.slice(1),
				});
				i++;
			}

			if (currentHunk) {
				hunks.push(currentHunk);
			}
			if (hunks.length === 0) {
				parseError("Update file section must include at least one hunk", i - 1);
			}

			operations.push({ type: "update", path, moveTo, hunks });
			continue;
		}

		parseError("Unexpected line in patch", i);
	}

	if (!sawEndPatch) {
		throw new Error('Patch must end with "*** End Patch"');
	}

	if (operations.length === 0) {
		throw new Error("Patch did not include any operations");
	}

	return { operations };
}

function splitLines(content: string): string[] {
	if (content.length === 0) {
		return [];
	}
	return content.split("\n");
}

function findSequence(haystack: string[], needle: string[], start: number): number {
	if (needle.length === 0) {
		return start;
	}
	for (let i = Math.max(0, start); i <= haystack.length - needle.length; i++) {
		let match = true;
		for (let j = 0; j < needle.length; j++) {
			if (haystack[i + j] !== needle[j]) {
				match = false;
				break;
			}
		}
		if (match) return i;
	}
	return -1;
}

function applyUpdateHunks(
	lines: string[],
	hunks: PatchHunk[],
	path: string,
): { lines: string[]; firstChangedLine?: number } {
	const working = [...lines];
	let searchStart = 0;
	let firstChangedLine: number | undefined;

	for (const hunk of hunks) {
		const unchangedOrRemoved = hunk.lines.filter((line) => line.type !== "add").map((line) => line.text);
		const withAdds = hunk.lines.filter((line) => line.type !== "remove").map((line) => line.text);
		const hasChanges = hunk.lines.some((line) => line.type !== "context");

		let matchIndex = findSequence(working, unchangedOrRemoved, searchStart);
		if (matchIndex === -1) {
			matchIndex = findSequence(working, unchangedOrRemoved, 0);
		}
		if (matchIndex === -1) {
			throw new Error(`Failed to apply hunk to ${path}: context did not match`);
		}

		if (hasChanges && firstChangedLine === undefined) {
			firstChangedLine = matchIndex + 1;
		}

		working.splice(matchIndex, unchangedOrRemoved.length, ...withAdds);
		searchStart = matchIndex + withAdds.length;
	}

	return { lines: working, firstChangedLine };
}

function toUnifiedDiff(oldPath: string, newPath: string, oldContent: string, newContent: string): string {
	return Diff.createTwoFilesPatch(oldPath, newPath, oldContent, newContent, "before", "after", {
		context: 3,
	});
}

function normalizePatchInput(input: ApplyPatchToolInput): string {
	if (typeof input === "string") {
		return input;
	}
	if (input && typeof input === "object" && "patch" in input && typeof input.patch === "string") {
		return input.patch;
	}
	throw new Error("Invalid apply_patch input. Expected patch text string.");
}

export function createApplyPatchTool(cwd: string, options?: ApplyPatchToolOptions): AgentTool<typeof applyPatchSchema> {
	const ops = options?.operations ?? defaultApplyPatchOperations;

	return {
		name: "apply_patch",
		label: "apply_patch",
		description:
			"Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
		type: "custom",
		format: {
			type: "grammar",
			syntax: "lark",
			definition: APPLY_PATCH_GRAMMAR,
		},
		parameters: applyPatchSchema,
		execute: async (_toolCallId, input) => {
			const patchText = normalizePatchInput(input);
			const parsed = parsePatch(patchText);

			const diffParts: string[] = [];
			const changedFiles: string[] = [];
			let firstChangedLine: number | undefined;

			for (const operation of parsed.operations) {
				if (operation.type === "add") {
					const absolutePath = resolveToCwd(operation.path, cwd);
					if (await ops.fileExists(absolutePath)) {
						throw new Error(`File already exists: ${operation.path}`);
					}

					const normalizedNewContent = operation.lines.join("\n");
					await ops.ensureParentDir(absolutePath);
					await ops.writeFile(absolutePath, normalizedNewContent);

					diffParts.push(toUnifiedDiff(operation.path, operation.path, "", normalizedNewContent));
					changedFiles.push(operation.path);
					if (firstChangedLine === undefined) firstChangedLine = 1;
					continue;
				}

				if (operation.type === "delete") {
					const absolutePath = resolveToCwd(operation.path, cwd);
					if (!(await ops.fileExists(absolutePath))) {
						throw new Error(`File not found: ${operation.path}`);
					}

					const rawContent = await ops.readFile(absolutePath);
					const { text: withoutBom } = stripBom(rawContent);
					const normalizedOldContent = normalizeToLF(withoutBom);

					await ops.deleteFile(absolutePath);

					diffParts.push(toUnifiedDiff(operation.path, operation.path, normalizedOldContent, ""));
					changedFiles.push(operation.path);
					if (firstChangedLine === undefined) firstChangedLine = 1;
					continue;
				}

				const sourceAbsolutePath = resolveToCwd(operation.path, cwd);
				if (!(await ops.fileExists(sourceAbsolutePath))) {
					throw new Error(`File not found: ${operation.path}`);
				}

				const rawContent = await ops.readFile(sourceAbsolutePath);
				const { bom, text: withoutBom } = stripBom(rawContent);
				const originalEnding = detectLineEnding(withoutBom);
				const normalizedOldContent = normalizeToLF(withoutBom);
				const oldLines = splitLines(normalizedOldContent);
				const applied = applyUpdateHunks(oldLines, operation.hunks, operation.path);
				const normalizedNewContent = applied.lines.join("\n");
				const targetPath = operation.moveTo ?? operation.path;
				const targetAbsolutePath = resolveToCwd(targetPath, cwd);

				if (
					operation.moveTo &&
					targetAbsolutePath !== sourceAbsolutePath &&
					(await ops.fileExists(targetAbsolutePath))
				) {
					throw new Error(`Cannot move to existing path: ${operation.moveTo}`);
				}

				const finalContent = bom + restoreLineEndings(normalizedNewContent, originalEnding);
				await ops.ensureParentDir(targetAbsolutePath);
				await ops.writeFile(targetAbsolutePath, finalContent);
				if (operation.moveTo && targetAbsolutePath !== sourceAbsolutePath) {
					await ops.deleteFile(sourceAbsolutePath);
				}

				diffParts.push(toUnifiedDiff(operation.path, targetPath, normalizedOldContent, normalizedNewContent));
				changedFiles.push(operation.moveTo ? `${operation.path} -> ${operation.moveTo}` : operation.path);
				if (firstChangedLine === undefined && applied.firstChangedLine !== undefined) {
					firstChangedLine = applied.firstChangedLine;
				}
			}

			const changedCount = changedFiles.length;
			const changedSummary = changedCount === 1 ? `1 file (${changedFiles[0]})` : `${changedCount} files`;

			return {
				content: [{ type: "text", text: `Applied patch successfully to ${changedSummary}.` }],
				details: {
					diff: diffParts.join("\n"),
					changedFiles,
					firstChangedLine,
				} satisfies ApplyPatchToolDetails,
			};
		},
	};
}

/** Default apply_patch tool using process.cwd() - for backwards compatibility */
export const applyPatchTool = createApplyPatchTool(process.cwd());

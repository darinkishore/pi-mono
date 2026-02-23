import * as Diff from "diff";
import { theme } from "../theme/theme.js";

function replaceTabs(text: string): string {
	return text.replace(/\t/g, "   ");
}

function renderIntraLineDiff(oldContent: string, newContent: string): { removedLine: string; addedLine: string } {
	const wordDiff = Diff.diffWordsWithSpace(oldContent, newContent);

	let removedLine = "";
	let addedLine = "";
	let isFirstRemoved = true;
	let isFirstAdded = true;

	for (const part of wordDiff) {
		if (part.removed) {
			let value = part.value;
			// Strip leading whitespace from the first removed part
			if (isFirstRemoved) {
				const leadingWs = value.match(/^(\s*)/)?.[1] || "";
				value = value.slice(leadingWs.length);
				removedLine += leadingWs;
				isFirstRemoved = false;
			}
			if (value) {
				removedLine += theme.inverse(value);
			}
		} else if (part.added) {
			let value = part.value;
			// Strip leading whitespace from the first added part
			if (isFirstAdded) {
				const leadingWs = value.match(/^(\s*)/)?.[1] || "";
				value = value.slice(leadingWs.length);
				addedLine += leadingWs;
				isFirstAdded = false;
			}
			if (value) {
				addedLine += theme.inverse(value);
			}
		} else {
			removedLine += part.value;
			addedLine += part.value;
		}
	}

	return { removedLine, addedLine };
}

function parseLegacyDiffLine(line: string): { prefix: string; lineNum: string; content: string } | null {
	const match = line.match(/^([+-\s])(\s*\d*)\s(.*)$/);
	if (!match) return null;
	return { prefix: match[1], lineNum: match[2], content: match[3] };
}

function isUnifiedDiff(diffText: string): boolean {
	return diffText.includes("@@") || (diffText.includes("--- ") && diffText.includes("+++ "));
}

function renderUnifiedDiff(diffText: string): string {
	const lines = diffText.split("\n");
	const result: string[] = [];

	let i = 0;
	while (i < lines.length) {
		const line = lines[i] ?? "";

		if (
			line.startsWith("diff --git ") ||
			line.startsWith("index ") ||
			line.startsWith("Index: ") ||
			line.startsWith("===") ||
			line.startsWith("@@") ||
			line.startsWith("--- ") ||
			line.startsWith("+++ ")
		) {
			result.push(theme.fg("toolDiffContext", line));
			i++;
			continue;
		}

		if (line.startsWith("-") && !line.startsWith("--- ")) {
			const removedLines: string[] = [];
			while (i < lines.length) {
				const current = lines[i] ?? "";
				if (!current.startsWith("-") || current.startsWith("--- ")) break;
				removedLines.push(current.slice(1));
				i++;
			}

			const addedLines: string[] = [];
			while (i < lines.length) {
				const current = lines[i] ?? "";
				if (!current.startsWith("+") || current.startsWith("+++ ")) break;
				addedLines.push(current.slice(1));
				i++;
			}

			if (removedLines.length === 1 && addedLines.length === 1) {
				const { removedLine, addedLine } = renderIntraLineDiff(
					replaceTabs(removedLines[0]),
					replaceTabs(addedLines[0]),
				);
				result.push(theme.fg("toolDiffRemoved", `-${removedLine}`));
				result.push(theme.fg("toolDiffAdded", `+${addedLine}`));
			} else {
				for (const removed of removedLines) {
					result.push(theme.fg("toolDiffRemoved", `-${replaceTabs(removed)}`));
				}
				for (const added of addedLines) {
					result.push(theme.fg("toolDiffAdded", `+${replaceTabs(added)}`));
				}
			}
			continue;
		}

		if (line.startsWith("+") && !line.startsWith("+++ ")) {
			result.push(theme.fg("toolDiffAdded", `+${replaceTabs(line.slice(1))}`));
			i++;
			continue;
		}

		if (line.startsWith(" ")) {
			result.push(theme.fg("toolDiffContext", ` ${replaceTabs(line.slice(1))}`));
			i++;
			continue;
		}

		result.push(theme.fg("toolDiffContext", line));
		i++;
	}

	return result.join("\n");
}

function renderLegacyDiff(diffText: string): string {
	const lines = diffText.split("\n");
	const result: string[] = [];

	let i = 0;
	while (i < lines.length) {
		const line = lines[i] ?? "";
		const parsed = parseLegacyDiffLine(line);

		if (!parsed) {
			result.push(theme.fg("toolDiffContext", line));
			i++;
			continue;
		}

		if (parsed.prefix === "-") {
			const removedLines: { lineNum: string; content: string }[] = [];
			while (i < lines.length) {
				const parsedLine = parseLegacyDiffLine(lines[i] ?? "");
				if (!parsedLine || parsedLine.prefix !== "-") break;
				removedLines.push({ lineNum: parsedLine.lineNum, content: parsedLine.content });
				i++;
			}

			const addedLines: { lineNum: string; content: string }[] = [];
			while (i < lines.length) {
				const parsedLine = parseLegacyDiffLine(lines[i] ?? "");
				if (!parsedLine || parsedLine.prefix !== "+") break;
				addedLines.push({ lineNum: parsedLine.lineNum, content: parsedLine.content });
				i++;
			}

			if (removedLines.length === 1 && addedLines.length === 1) {
				const removed = removedLines[0];
				const added = addedLines[0];
				const { removedLine, addedLine } = renderIntraLineDiff(
					replaceTabs(removed.content),
					replaceTabs(added.content),
				);
				result.push(theme.fg("toolDiffRemoved", `-${removed.lineNum} ${removedLine}`));
				result.push(theme.fg("toolDiffAdded", `+${added.lineNum} ${addedLine}`));
			} else {
				for (const removed of removedLines) {
					result.push(theme.fg("toolDiffRemoved", `-${removed.lineNum} ${replaceTabs(removed.content)}`));
				}
				for (const added of addedLines) {
					result.push(theme.fg("toolDiffAdded", `+${added.lineNum} ${replaceTabs(added.content)}`));
				}
			}
			continue;
		}

		if (parsed.prefix === "+") {
			result.push(theme.fg("toolDiffAdded", `+${parsed.lineNum} ${replaceTabs(parsed.content)}`));
			i++;
			continue;
		}

		result.push(theme.fg("toolDiffContext", ` ${parsed.lineNum} ${replaceTabs(parsed.content)}`));
		i++;
	}

	return result.join("\n");
}

export interface RenderDiffOptions {
	/** File path (unused, kept for API compatibility) */
	filePath?: string;
}

export function renderDiff(diffText: string, _options: RenderDiffOptions = {}): string {
	if (!diffText.trim()) {
		return "";
	}
	return isUnifiedDiff(diffText) ? renderUnifiedDiff(diffText) : renderLegacyDiff(diffText);
}

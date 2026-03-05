import type { Model } from "@mariozechner/pi-ai";
import { buildGpt52Toolset, buildGpt53CodexFamilyToolset, type CodexToolSpec } from "./tool-specs.js";

export const DEFAULT_PI_BASE_TOOL_NAMES = ["read", "bash", "edit", "write"] as const;

const ADAPTER_BACKED_TOOL_NAMES = new Set<string>([
	"list_mcp_resources",
	"list_mcp_resource_templates",
	"read_mcp_resource",
	"search_tool_bm25",
	"js_repl",
	"js_repl_reset",
	"web_search",
	"update_plan",
	"request_user_input",
	"spawn_agent",
	"send_input",
	"resume_agent",
	"wait",
	"close_agent",
]);

function normalizeModelId(modelId: string): string {
	const normalized = modelId.toLowerCase();
	const slashIndex = normalized.lastIndexOf("/");
	if (slashIndex === -1) {
		return normalized;
	}
	return normalized.slice(slashIndex + 1);
}

function isGpt52Model(modelId: string): boolean {
	return normalizeModelId(modelId) === "gpt-5.2";
}

function isGpt53CodexFamilyModel(modelId: string): boolean {
	return /^gpt-5\.(?:3|4)(?:\.\d+)?-codex/.test(normalizeModelId(modelId));
}

function isGpt54Model(modelId: string): boolean {
	return normalizeModelId(modelId) === "gpt-5.4";
}

function getToolName(spec: CodexToolSpec): string | undefined {
	if (spec.type === "function" || spec.type === "custom") {
		return spec.name;
	}
	return undefined;
}

function dedupeInOrder(items: string[]): string[] {
	const seen = new Set<string>();
	const out: string[] = [];
	for (const item of items) {
		if (seen.has(item)) continue;
		seen.add(item);
		out.push(item);
	}
	return out;
}

function filterSupportedToolNames(toolNames: string[], availableBaseToolNames: Iterable<string>): string[] {
	const available = new Set(availableBaseToolNames);
	return toolNames.filter((name) => available.has(name));
}

export function isCodexPresetModel(model: Model<any> | undefined): boolean {
	if (!model) return false;
	if (model.provider === "openai-codex") {
		return isGpt52Model(model.id) || isGpt53CodexFamilyModel(model.id);
	}
	if (model.provider === "openai") {
		return isGpt54Model(model.id);
	}
	return false;
}

export function getCodexPresetSpecsForModel(model: Model<any> | undefined): CodexToolSpec[] | undefined {
	if (!model) return undefined;

	if (model.provider === "openai" && isGpt54Model(model.id)) {
		return buildGpt53CodexFamilyToolset();
	}
	if (model.provider !== "openai-codex") return undefined;

	if (isGpt52Model(model.id)) {
		return buildGpt52Toolset();
	}

	if (isGpt53CodexFamilyModel(model.id)) {
		return buildGpt53CodexFamilyToolset();
	}

	return undefined;
}

export function getCodexPresetToolNamesForModel(
	model: Model<any> | undefined,
	availableBaseToolNames: Iterable<string>,
): string[] | undefined {
	const specs = getCodexPresetSpecsForModel(model);
	if (!specs) return undefined;

	const nonAdapterNames = specs
		.map(getToolName)
		.filter((name): name is string => Boolean(name))
		.filter((name) => !ADAPTER_BACKED_TOOL_NAMES.has(name));

	const deduped = dedupeInOrder(nonAdapterNames);
	const supported = filterSupportedToolNames(deduped, availableBaseToolNames);
	return supported.length > 0 ? supported : undefined;
}

export function getDefaultBaseToolNamesForModel(
	model: Model<any> | undefined,
	availableBaseToolNames: Iterable<string>,
): string[] {
	const codexPresetNames = getCodexPresetToolNamesForModel(model, availableBaseToolNames);
	if (codexPresetNames && codexPresetNames.length > 0) {
		return codexPresetNames;
	}
	return filterSupportedToolNames([...DEFAULT_PI_BASE_TOOL_NAMES], availableBaseToolNames);
}

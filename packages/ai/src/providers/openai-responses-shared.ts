import type OpenAI from "openai";
import type {
	ApplyPatchTool as OpenAIApplyPatchTool,
	CustomTool as OpenAICustomTool,
	Tool as OpenAITool,
	ResponseApplyPatchToolCall,
	ResponseApplyPatchToolCallOutput,
	ResponseCreateParamsStreaming,
	ResponseCustomToolCall,
	ResponseCustomToolCallOutput,
	ResponseFunctionToolCall,
	ResponseInput,
	ResponseInputContent,
	ResponseInputImage,
	ResponseInputText,
	ResponseOutputMessage,
	ResponseReasoningItem,
	ResponseStreamEvent,
} from "openai/resources/responses/responses.js";
import { calculateCost } from "../models.js";
import type {
	Api,
	AssistantMessage,
	Context,
	ImageContent,
	Message,
	Model,
	StopReason,
	TextContent,
	ThinkingContent,
	Tool,
	ToolCall,
	Usage,
} from "../types.js";
import type { AssistantMessageEventStream } from "../utils/event-stream.js";
import { parseStreamingJson } from "../utils/json-parse.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import { transformMessages } from "./transform-messages.js";

// =============================================================================
// Utilities
// =============================================================================

/** Fast deterministic hash to shorten long strings */
function shortHash(str: string): string {
	let h1 = 0xdeadbeef;
	let h2 = 0x41c6ce57;
	for (let i = 0; i < str.length; i++) {
		const ch = str.charCodeAt(i);
		h1 = Math.imul(h1 ^ ch, 2654435761);
		h2 = Math.imul(h2 ^ ch, 1597334677);
	}
	h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
	h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
	return (h2 >>> 0).toString(36) + (h1 >>> 0).toString(36);
}

export interface OpenAIResponsesStreamOptions {
	serviceTier?: ResponseCreateParamsStreaming["service_tier"];
	applyServiceTierPricing?: (
		usage: Usage,
		serviceTier: ResponseCreateParamsStreaming["service_tier"] | undefined,
	) => void;
}

export interface ConvertResponsesMessagesOptions {
	includeSystemPrompt?: boolean;
}

export interface ConvertResponsesToolsOptions {
	strict?: boolean | null;
}

type ResolvedToolType = "function" | "custom" | "apply_patch";

function resolveToolType(tool: Tool | undefined): ResolvedToolType {
	if (tool?.type === "custom" || tool?.type === "apply_patch") {
		return tool.type;
	}
	return "function";
}

function serializeToolCallArguments(args: unknown): string {
	if (typeof args === "string") return args;
	try {
		return JSON.stringify(args);
	} catch {
		return String(args);
	}
}

// =============================================================================
// Message conversion
// =============================================================================

export function convertResponsesMessages<TApi extends Api>(
	model: Model<TApi>,
	context: Context,
	allowedToolCallProviders: ReadonlySet<string>,
	options?: ConvertResponsesMessagesOptions,
): ResponseInput {
	const messages: ResponseInput = [];

	const normalizeToolCallId = (id: string): string => {
		if (!allowedToolCallProviders.has(model.provider)) return id;
		if (!id.includes("|")) return id;
		const [callId, itemId] = id.split("|");
		const sanitizedCallId = callId.replace(/[^a-zA-Z0-9_-]/g, "_");
		let sanitizedItemId = itemId.replace(/[^a-zA-Z0-9_-]/g, "_");
		// OpenAI Responses API item IDs typically start with:
		// - fc_  (function_call)
		// - ctc_ (custom_tool_call)
		// - apc_ (apply_patch_call)
		if (
			!sanitizedItemId.startsWith("fc") &&
			!sanitizedItemId.startsWith("ctc") &&
			!sanitizedItemId.startsWith("apc")
		) {
			sanitizedItemId = `fc_${sanitizedItemId}`;
		}
		// Truncate to 64 chars and strip trailing underscores (OpenAI Codex rejects them)
		let normalizedCallId = sanitizedCallId.length > 64 ? sanitizedCallId.slice(0, 64) : sanitizedCallId;
		let normalizedItemId = sanitizedItemId.length > 64 ? sanitizedItemId.slice(0, 64) : sanitizedItemId;
		normalizedCallId = normalizedCallId.replace(/_+$/, "");
		normalizedItemId = normalizedItemId.replace(/_+$/, "");
		return `${normalizedCallId}|${normalizedItemId}`;
	};

	const transformedMessages = transformMessages(context.messages, model, normalizeToolCallId);
	const toolTypeByName = new Map<string, ResolvedToolType>(
		(context.tools || []).map((tool) => [tool.name, resolveToolType(tool)]),
	);

	const includeSystemPrompt = options?.includeSystemPrompt ?? true;
	if (includeSystemPrompt && context.systemPrompt) {
		const role = model.reasoning ? "developer" : "system";
		messages.push({
			role,
			content: sanitizeSurrogates(context.systemPrompt),
		});
	}

	let msgIndex = 0;
	for (const msg of transformedMessages) {
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				messages.push({
					role: "user",
					content: [{ type: "input_text", text: sanitizeSurrogates(msg.content) }],
				});
			} else {
				const content: ResponseInputContent[] = msg.content.map((item): ResponseInputContent => {
					if (item.type === "text") {
						return {
							type: "input_text",
							text: sanitizeSurrogates(item.text),
						} satisfies ResponseInputText;
					}
					return {
						type: "input_image",
						detail: "auto",
						image_url: `data:${item.mimeType};base64,${item.data}`,
					} satisfies ResponseInputImage;
				});
				const filteredContent = !model.input.includes("image")
					? content.filter((c) => c.type !== "input_image")
					: content;
				if (filteredContent.length === 0) continue;
				messages.push({
					role: "user",
					content: filteredContent,
				});
			}
		} else if (msg.role === "assistant") {
			const output: ResponseInput = [];
			const assistantMsg = msg as AssistantMessage;
			const isDifferentModel =
				assistantMsg.model !== model.id &&
				assistantMsg.provider === model.provider &&
				assistantMsg.api === model.api;

			for (const block of msg.content) {
				if (block.type === "thinking") {
					if (block.thinkingSignature) {
						const reasoningItem = JSON.parse(block.thinkingSignature) as ResponseReasoningItem;
						output.push(reasoningItem);
					}
				} else if (block.type === "text") {
					const textBlock = block as TextContent;
					// OpenAI requires id to be max 64 characters
					let msgId = textBlock.textSignature;
					if (!msgId) {
						msgId = `msg_${msgIndex}`;
					} else if (msgId.length > 64) {
						msgId = `msg_${shortHash(msgId)}`;
					}
					output.push({
						type: "message",
						role: "assistant",
						content: [{ type: "output_text", text: sanitizeSurrogates(textBlock.text), annotations: [] }],
						status: "completed",
						id: msgId,
					} satisfies ResponseOutputMessage);
				} else if (block.type === "toolCall") {
					const toolCall = block as ToolCall;
					const [callId, itemIdRaw] = toolCall.id.split("|");
					let itemId: string | undefined = itemIdRaw;
					const toolType = toolCall.toolType ?? toolTypeByName.get(toolCall.name) ?? "function";

					// For different-model messages, set id to undefined to avoid pairing validation.
					// OpenAI tracks which fc_xxx IDs were paired with rs_xxx reasoning items.
					// By omitting the id, we avoid triggering that validation (like cross-provider does).
					if (isDifferentModel && itemId?.startsWith("fc_")) {
						itemId = undefined;
					}

					if (toolType === "custom") {
						output.push({
							type: "custom_tool_call",
							id: itemId,
							call_id: callId,
							name: toolCall.name,
							input: serializeToolCallArguments(toolCall.arguments),
						} satisfies ResponseCustomToolCall);
					} else if (toolType === "apply_patch" && toolCall.arguments && typeof toolCall.arguments === "object") {
						output.push({
							type: "apply_patch_call",
							id: itemId,
							call_id: callId,
							operation: toolCall.arguments as ResponseApplyPatchToolCall["operation"],
						} as ResponseApplyPatchToolCall);
					} else {
						output.push({
							type: "function_call",
							id: itemId,
							call_id: callId,
							name: toolCall.name,
							arguments: serializeToolCallArguments(toolCall.arguments),
						});
					}
				}
			}
			if (output.length === 0) continue;
			messages.push(...output);
		} else if (msg.role === "toolResult") {
			// Extract text and image content
			const textResult = msg.content
				.filter((c): c is TextContent => c.type === "text")
				.map((c) => c.text)
				.join("\n");
			const hasImages = msg.content.some((c): c is ImageContent => c.type === "image");

			// Always send function_call_output with text (or placeholder if only images)
			const hasText = textResult.length > 0;
			const [callId] = msg.toolCallId.split("|");
			const outputText = sanitizeSurrogates(hasText ? textResult : "(see attached image)");
			const toolType = toolTypeByName.get(msg.toolName) ?? "function";
			if (toolType === "custom") {
				messages.push({
					type: "custom_tool_call_output",
					call_id: callId,
					output: outputText,
				} satisfies ResponseCustomToolCallOutput);
			} else if (toolType === "apply_patch") {
				messages.push({
					type: "apply_patch_call_output",
					call_id: callId,
					id: callId,
					status: msg.isError ? "failed" : "completed",
					output: outputText,
				} satisfies ResponseApplyPatchToolCallOutput);
			} else {
				messages.push({
					type: "function_call_output",
					call_id: callId,
					output: outputText,
				});
			}

			// If there are images and model supports them, send a follow-up user message with images
			if (hasImages && model.input.includes("image")) {
				const contentParts: ResponseInputContent[] = [];

				// Add text prefix
				contentParts.push({
					type: "input_text",
					text: "Attached image(s) from tool result:",
				} satisfies ResponseInputText);

				// Add images
				for (const block of msg.content) {
					if (block.type === "image") {
						contentParts.push({
							type: "input_image",
							detail: "auto",
							image_url: `data:${block.mimeType};base64,${block.data}`,
						} satisfies ResponseInputImage);
					}
				}

				messages.push({
					role: "user",
					content: contentParts,
				});
			}
		}
		msgIndex++;
	}

	return messages;
}

function buildEmptyAssistantUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			total: 0,
		},
	};
}

function parseDataUrlToImageContent(imageUrl: string): ImageContent | undefined {
	const match = /^data:([^;,]+);base64,(.+)$/.exec(imageUrl);
	if (!match) return undefined;
	return {
		type: "image",
		mimeType: match[1],
		data: match[2],
	};
}

function stringifyToolOutput(output: unknown): string {
	if (typeof output === "string") return sanitizeSurrogates(output);
	try {
		return sanitizeSurrogates(JSON.stringify(output));
	} catch {
		return sanitizeSurrogates(String(output));
	}
}

function extractTextFromUserInputContent(content: unknown): string {
	if (!Array.isArray(content)) return "";
	const textParts: string[] = [];
	for (const part of content) {
		if (
			part &&
			typeof part === "object" &&
			(part as { type?: string }).type === "input_text" &&
			typeof (part as { text?: unknown }).text === "string"
		) {
			textParts.push((part as { text: string }).text);
		}
	}
	return textParts.join("\n");
}

function extractThinkingTextFromItem(item: unknown): string {
	if (!item || typeof item !== "object") return "";
	const typed = item as {
		summary?: unknown;
		content?: unknown;
	};
	const chunks: string[] = [];
	if (Array.isArray(typed.summary)) {
		for (const part of typed.summary) {
			if (part && typeof part === "object" && typeof (part as { text?: unknown }).text === "string") {
				chunks.push((part as { text: string }).text);
			}
		}
	}
	if (Array.isArray(typed.content)) {
		for (const part of typed.content) {
			if (part && typeof part === "object" && typeof (part as { text?: unknown }).text === "string") {
				chunks.push((part as { text: string }).text);
			}
		}
	}
	return chunks.join("\n");
}

export function convertResponsesInputToMessages<TApi extends Api>(model: Model<TApi>, input: ResponseInput): Message[] {
	const messages: Message[] = [];
	const toolNameByCallId = new Map<string, string>();
	let timestamp = Date.now();
	const nextTimestamp = () => {
		timestamp += 1;
		return timestamp;
	};

	const makeAssistantMessage = (
		content: AssistantMessage["content"],
		stopReason: StopReason = "stop",
	): AssistantMessage => ({
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: buildEmptyAssistantUsage(),
		stopReason,
		timestamp: nextTimestamp(),
	});

	for (const item of input) {
		const itemType = (item as { type?: string }).type;
		if (!itemType) continue;

		if (itemType === "message") {
			const role = (item as { role?: string }).role;
			const content = (item as { content?: unknown }).content;
			if (role === "user") {
				const contentBlocks: (TextContent | ImageContent)[] = [];
				if (typeof content === "string") {
					contentBlocks.push({ type: "text", text: sanitizeSurrogates(content) });
				} else if (Array.isArray(content)) {
					for (const part of content) {
						if (
							part &&
							typeof part === "object" &&
							(part as { type?: string }).type === "input_text" &&
							typeof (part as { text?: unknown }).text === "string"
						) {
							contentBlocks.push({
								type: "text",
								text: sanitizeSurrogates((part as { text: string }).text),
							});
						} else if (
							part &&
							typeof part === "object" &&
							(part as { type?: string }).type === "input_image" &&
							typeof (part as { image_url?: unknown }).image_url === "string"
						) {
							const image = parseDataUrlToImageContent((part as { image_url: string }).image_url);
							if (image) {
								contentBlocks.push(image);
							}
						}
					}
				}
				if (contentBlocks.length > 0) {
					messages.push({
						role: "user",
						content: contentBlocks,
						timestamp: nextTimestamp(),
					});
				}
			} else if (role === "assistant" && Array.isArray(content)) {
				const assistantContent: AssistantMessage["content"] = [];
				for (const part of content) {
					if (
						part &&
						typeof part === "object" &&
						(part as { type?: string }).type === "output_text" &&
						typeof (part as { text?: unknown }).text === "string"
					) {
						assistantContent.push({
							type: "text",
							text: sanitizeSurrogates((part as { text: string }).text),
						});
					}
				}
				if (assistantContent.length > 0) {
					messages.push(makeAssistantMessage(assistantContent));
				}
			}
			continue;
		}

		if (itemType === "reasoning" || itemType === "compaction") {
			let signature: string | undefined;
			try {
				signature = JSON.stringify(item);
			} catch {
				signature = undefined;
			}
			const thinking = extractThinkingTextFromItem(item);
			messages.push(
				makeAssistantMessage(
					[
						{
							type: "thinking",
							thinking,
							thinkingSignature: signature,
						},
					],
					"stop",
				),
			);
			continue;
		}

		if (itemType === "function_call") {
			const call = item as {
				call_id?: string;
				id?: string;
				name?: string;
				arguments?: unknown;
			};
			if (!call.call_id || !call.name) continue;
			toolNameByCallId.set(call.call_id, call.name);
			const parsedArgs = typeof call.arguments === "string" ? parseStreamingJson(call.arguments) : call.arguments;
			messages.push(
				makeAssistantMessage([
					{
						type: "toolCall",
						id: `${call.call_id}|${call.id ?? call.call_id}`,
						name: call.name,
						arguments: parsedArgs,
						toolType: "function",
					},
				]),
			);
			continue;
		}

		if (itemType === "custom_tool_call") {
			const call = item as {
				call_id?: string;
				id?: string;
				name?: string;
				input?: unknown;
			};
			if (!call.call_id || !call.name) continue;
			toolNameByCallId.set(call.call_id, call.name);
			messages.push(
				makeAssistantMessage([
					{
						type: "toolCall",
						id: `${call.call_id}|${call.id ?? call.call_id}`,
						name: call.name,
						arguments: call.input ?? "",
						toolType: "custom",
					},
				]),
			);
			continue;
		}

		if (itemType === "apply_patch_call") {
			const call = item as {
				call_id?: string;
				id?: string;
				operation?: unknown;
			};
			if (!call.call_id) continue;
			toolNameByCallId.set(call.call_id, "apply_patch");
			messages.push(
				makeAssistantMessage([
					{
						type: "toolCall",
						id: `${call.call_id}|${call.id ?? call.call_id}`,
						name: "apply_patch",
						arguments: call.operation ?? {},
						toolType: "apply_patch",
					},
				]),
			);
			continue;
		}

		if (itemType === "function_call_output") {
			const output = item as {
				call_id?: string;
				output?: unknown;
			};
			if (!output.call_id) continue;
			const toolName = toolNameByCallId.get(output.call_id) ?? "tool";
			messages.push({
				role: "toolResult",
				toolCallId: output.call_id,
				toolName,
				content: [{ type: "text", text: stringifyToolOutput(output.output) }],
				isError: false,
				timestamp: nextTimestamp(),
			});
			continue;
		}

		if (itemType === "custom_tool_call_output") {
			const output = item as {
				call_id?: string;
				output?: unknown;
			};
			if (!output.call_id) continue;
			const toolName = toolNameByCallId.get(output.call_id) ?? "tool";
			messages.push({
				role: "toolResult",
				toolCallId: output.call_id,
				toolName,
				content: [{ type: "text", text: stringifyToolOutput(output.output) }],
				isError: false,
				timestamp: nextTimestamp(),
			});
			continue;
		}

		if (itemType === "apply_patch_call_output") {
			const output = item as {
				call_id?: string;
				output?: unknown;
				status?: string;
			};
			if (!output.call_id) continue;
			messages.push({
				role: "toolResult",
				toolCallId: output.call_id,
				toolName: "apply_patch",
				content: [{ type: "text", text: stringifyToolOutput(output.output) }],
				isError: output.status === "failed",
				timestamp: nextTimestamp(),
			});
			continue;
		}

		if (itemType === "user_message") {
			// Compat shim for alternate envelopes, if present.
			const text = extractTextFromUserInputContent((item as { content?: unknown }).content);
			if (text.length > 0) {
				messages.push({
					role: "user",
					content: [{ type: "text", text }],
					timestamp: nextTimestamp(),
				});
			}
		}
	}

	return messages;
}

// =============================================================================
// Tool conversion
// =============================================================================

export function convertResponsesTools(tools: Tool[], options?: ConvertResponsesToolsOptions): OpenAITool[] {
	const strict = options?.strict === undefined ? false : options.strict;
	return tools.map((tool): OpenAITool => {
		const toolType = resolveToolType(tool);
		if (toolType === "custom") {
			return {
				type: "custom",
				name: tool.name,
				description: tool.description,
				format: tool.format ?? { type: "text" },
			} satisfies OpenAICustomTool;
		}
		if (toolType === "apply_patch") {
			return {
				type: "apply_patch",
			} satisfies OpenAIApplyPatchTool;
		}
		return {
			type: "function",
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters as any, // TypeBox already generates JSON Schema
			strict,
		};
	});
}

// =============================================================================
// Stream processing
// =============================================================================

export async function processResponsesStream<TApi extends Api>(
	openaiStream: AsyncIterable<ResponseStreamEvent>,
	output: AssistantMessage,
	stream: AssistantMessageEventStream,
	model: Model<TApi>,
	options?: OpenAIResponsesStreamOptions,
): Promise<void> {
	let currentItem:
		| ResponseReasoningItem
		| ResponseOutputMessage
		| ResponseFunctionToolCall
		| ResponseCustomToolCall
		| ResponseApplyPatchToolCall
		| null = null;
	let currentBlock:
		| ThinkingContent
		| TextContent
		| (ToolCall & { partialJson?: string; partialInput?: string })
		| null = null;
	let reasoningDeltaMode: "none" | "summary" | "content" = "none";
	const blocks = output.content;
	const blockIndex = () => blocks.length - 1;

	for await (const event of openaiStream) {
		if (event.type === "response.output_item.added") {
			const item = event.item;
			if (item.type === "reasoning") {
				currentItem = item;
				currentBlock = { type: "thinking", thinking: "" };
				reasoningDeltaMode = "none";
				output.content.push(currentBlock);
				stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
			} else if (item.type === "message") {
				currentItem = item;
				currentBlock = { type: "text", text: "" };
				output.content.push(currentBlock);
				stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
			} else if (item.type === "function_call") {
				currentItem = item;
				currentBlock = {
					type: "toolCall",
					id: `${item.call_id}|${item.id ?? item.call_id}`,
					name: item.name,
					arguments: {},
					toolType: "function",
					partialJson: item.arguments || "",
				};
				output.content.push(currentBlock);
				stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
			} else if (item.type === "custom_tool_call") {
				currentItem = item;
				currentBlock = {
					type: "toolCall",
					id: `${item.call_id}|${item.id ?? item.call_id}`,
					name: item.name,
					arguments: item.input || "",
					toolType: "custom",
					partialInput: item.input || "",
				};
				output.content.push(currentBlock);
				stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
			} else if (item.type === "apply_patch_call") {
				currentItem = item;
				currentBlock = {
					type: "toolCall",
					id: `${item.call_id}|${item.id ?? item.call_id}`,
					name: "apply_patch",
					arguments: item.operation,
					toolType: "apply_patch",
				};
				output.content.push(currentBlock);
				stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
			}
		} else if (event.type === "response.reasoning_summary_part.added") {
			if (currentItem && currentItem.type === "reasoning") {
				currentItem.summary = currentItem.summary || [];
				currentItem.summary.push(event.part);
			}
		} else if (event.type === "response.reasoning_summary_text.delta") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				if (reasoningDeltaMode === "content") {
					continue;
				}
				reasoningDeltaMode = "summary";
				currentItem.summary = currentItem.summary || [];
				const lastPart = currentItem.summary[currentItem.summary.length - 1];
				if (lastPart) {
					currentBlock.thinking += event.delta;
					lastPart.text += event.delta;
					stream.push({
						type: "thinking_delta",
						contentIndex: blockIndex(),
						delta: event.delta,
						partial: output,
					});
				}
			}
		} else if (event.type === "response.reasoning_summary_part.done") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				if (reasoningDeltaMode === "content") {
					continue;
				}
				currentItem.summary = currentItem.summary || [];
				const lastPart = currentItem.summary[currentItem.summary.length - 1];
				if (lastPart) {
					currentBlock.thinking += "\n\n";
					lastPart.text += "\n\n";
					stream.push({
						type: "thinking_delta",
						contentIndex: blockIndex(),
						delta: "\n\n",
						partial: output,
					});
				}
			}
		} else if (event.type === "response.reasoning_text.delta") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				if (reasoningDeltaMode === "summary") {
					continue;
				}
				reasoningDeltaMode = "content";
				const contentIndex = event.content_index;
				currentItem.content = currentItem.content || [];
				while (currentItem.content.length <= contentIndex) {
					currentItem.content.push({ type: "reasoning_text", text: "" });
				}
				const contentPart = currentItem.content[contentIndex];
				if (contentPart?.type === "reasoning_text") {
					contentPart.text += event.delta;
				}
				currentBlock.thinking += event.delta;
				stream.push({
					type: "thinking_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.reasoning_text.done") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				if (reasoningDeltaMode === "summary") {
					continue;
				}
				reasoningDeltaMode = "content";
				const contentIndex = event.content_index;
				currentItem.content = currentItem.content || [];
				while (currentItem.content.length <= contentIndex) {
					currentItem.content.push({ type: "reasoning_text", text: "" });
				}
				const contentPart = currentItem.content[contentIndex];
				if (contentPart?.type === "reasoning_text") {
					contentPart.text = event.text;
				}
			}
		} else if (event.type === "response.content_part.added") {
			if (currentItem?.type === "message") {
				currentItem.content = currentItem.content || [];
				// Filter out ReasoningText, only accept output_text and refusal
				if (event.part.type === "output_text" || event.part.type === "refusal") {
					currentItem.content.push(event.part);
				}
			}
		} else if (event.type === "response.output_text.delta") {
			if (currentItem?.type === "message" && currentBlock?.type === "text") {
				if (!currentItem.content || currentItem.content.length === 0) {
					continue;
				}
				const lastPart = currentItem.content[currentItem.content.length - 1];
				if (lastPart?.type === "output_text") {
					currentBlock.text += event.delta;
					lastPart.text += event.delta;
					stream.push({
						type: "text_delta",
						contentIndex: blockIndex(),
						delta: event.delta,
						partial: output,
					});
				}
			}
		} else if (event.type === "response.refusal.delta") {
			if (currentItem?.type === "message" && currentBlock?.type === "text") {
				if (!currentItem.content || currentItem.content.length === 0) {
					continue;
				}
				const lastPart = currentItem.content[currentItem.content.length - 1];
				if (lastPart?.type === "refusal") {
					currentBlock.text += event.delta;
					lastPart.refusal += event.delta;
					stream.push({
						type: "text_delta",
						contentIndex: blockIndex(),
						delta: event.delta,
						partial: output,
					});
				}
			}
		} else if (event.type === "response.function_call_arguments.delta") {
			if (currentItem?.type === "function_call" && currentBlock?.type === "toolCall") {
				currentBlock.partialJson = (currentBlock.partialJson || "") + event.delta;
				currentBlock.arguments = parseStreamingJson(currentBlock.partialJson);
				stream.push({
					type: "toolcall_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.function_call_arguments.done") {
			if (currentItem?.type === "function_call" && currentBlock?.type === "toolCall") {
				currentBlock.partialJson = event.arguments;
				currentBlock.arguments = parseStreamingJson(currentBlock.partialJson);
			}
		} else if (event.type === "response.custom_tool_call_input.delta") {
			if (currentItem?.type === "custom_tool_call" && currentBlock?.type === "toolCall") {
				currentBlock.partialInput = (currentBlock.partialInput || "") + event.delta;
				currentBlock.arguments = currentBlock.partialInput;
				stream.push({
					type: "toolcall_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.custom_tool_call_input.done") {
			if (currentItem?.type === "custom_tool_call" && currentBlock?.type === "toolCall") {
				currentBlock.partialInput = event.input;
				currentBlock.arguments = event.input;
			}
		} else if (event.type === "response.output_item.done") {
			const item = event.item;

			if (item.type === "reasoning" && currentBlock?.type === "thinking") {
				const summaryText = item.summary?.map((s) => s.text).join("\n\n") || "";
				const contentText =
					item.content
						?.filter((part): part is { type: "reasoning_text"; text: string } => part.type === "reasoning_text")
						.map((part) => part.text)
						.join("") || "";
				currentBlock.thinking =
					reasoningDeltaMode === "content"
						? contentText || currentBlock.thinking
						: summaryText || contentText || currentBlock.thinking;
				currentBlock.thinkingSignature = JSON.stringify(item);
				stream.push({
					type: "thinking_end",
					contentIndex: blockIndex(),
					content: currentBlock.thinking,
					partial: output,
				});
				currentBlock = null;
				reasoningDeltaMode = "none";
			} else if (item.type === "message" && currentBlock?.type === "text") {
				currentBlock.text = item.content.map((c) => (c.type === "output_text" ? c.text : c.refusal)).join("");
				currentBlock.textSignature = item.id;
				stream.push({
					type: "text_end",
					contentIndex: blockIndex(),
					content: currentBlock.text,
					partial: output,
				});
				currentBlock = null;
				reasoningDeltaMode = "none";
			} else if (item.type === "function_call") {
				const args =
					currentBlock?.type === "toolCall" && currentBlock.partialJson
						? parseStreamingJson(currentBlock.partialJson)
						: parseStreamingJson(item.arguments || "{}");
				const toolCall: ToolCall = {
					type: "toolCall",
					id: `${item.call_id}|${item.id ?? item.call_id}`,
					name: item.name,
					arguments: args,
					toolType: "function",
				};

				currentBlock = null;
				reasoningDeltaMode = "none";
				stream.push({ type: "toolcall_end", contentIndex: blockIndex(), toolCall, partial: output });
			} else if (item.type === "custom_tool_call") {
				const input =
					currentBlock?.type === "toolCall" && typeof currentBlock.partialInput === "string"
						? currentBlock.partialInput
						: item.input || "";
				const toolCall: ToolCall = {
					type: "toolCall",
					id: `${item.call_id}|${item.id ?? item.call_id}`,
					name: item.name,
					arguments: input,
					toolType: "custom",
				};

				currentBlock = null;
				reasoningDeltaMode = "none";
				stream.push({ type: "toolcall_end", contentIndex: blockIndex(), toolCall, partial: output });
			} else if (item.type === "apply_patch_call") {
				const toolCall: ToolCall = {
					type: "toolCall",
					id: `${item.call_id}|${item.id ?? item.call_id}`,
					name: "apply_patch",
					arguments: item.operation,
					toolType: "apply_patch",
				};

				currentBlock = null;
				reasoningDeltaMode = "none";
				stream.push({ type: "toolcall_end", contentIndex: blockIndex(), toolCall, partial: output });
			}
		} else if (event.type === "response.completed") {
			const response = event.response;
			if (response?.usage) {
				const cachedTokens = response.usage.input_tokens_details?.cached_tokens || 0;
				output.usage = {
					// OpenAI includes cached tokens in input_tokens, so subtract to get non-cached input
					input: (response.usage.input_tokens || 0) - cachedTokens,
					output: response.usage.output_tokens || 0,
					cacheRead: cachedTokens,
					cacheWrite: 0,
					totalTokens: response.usage.total_tokens || 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				};
			}
			calculateCost(model, output.usage);
			if (options?.applyServiceTierPricing) {
				const serviceTier = response?.service_tier ?? options.serviceTier;
				options.applyServiceTierPricing(output.usage, serviceTier);
			}
			// Map status to stop reason
			output.stopReason = mapStopReason(response?.status);
			if (output.content.some((b) => b.type === "toolCall") && output.stopReason === "stop") {
				output.stopReason = "toolUse";
			}
		} else if (event.type === "error") {
			throw new Error(`Error Code ${event.code}: ${event.message}` || "Unknown error");
		} else if (event.type === "response.failed") {
			throw new Error("Unknown error");
		}
	}
}

function mapStopReason(status: OpenAI.Responses.ResponseStatus | undefined): StopReason {
	if (!status) return "stop";
	switch (status) {
		case "completed":
			return "stop";
		case "incomplete":
			return "length";
		case "failed":
		case "cancelled":
			return "error";
		// These two are wonky ...
		case "in_progress":
		case "queued":
			return "stop";
		default: {
			const _exhaustive: never = status;
			throw new Error(`Unhandled stop reason: ${_exhaustive}`);
		}
	}
}

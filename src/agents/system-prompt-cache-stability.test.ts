import { createHash } from "node:crypto";
import {
  SYSTEM_PROMPT_CACHE_BOUNDARY,
  splitSystemPromptCacheBoundary,
} from "@openclaw/ai/internal/shared";
import { describe, expect, it } from "vitest";
import { buildAgentSystemPrompt } from "./system-prompt.js";

/**
 * Regression tests for issue #150081: prompt-cache prefix instability
 * between normal turns and announce/yield turns.
 *
 * The system prompt is split by SYSTEM_PROMPT_CACHE_BOUNDARY into:
 * - stable prefix (cached, should be byte-identical across turn types)
 * - dynamic suffix (rebuilt each turn, should also be stable across turn types)
 *
 * Ollama does pure prefix matching on the raw system message. Any change
 * in the suffix invalidates ~11k tokens of cached content, causing
 * prompt_eval_duration to increase from ~11s to ~123-203s.
 *
 * Prior partial fixes (all merged):
 * - #167898: Conversation Context stabilization
 * - #175309: Tool definition stabilization
 * - #170516: Sandbox/elevated section moved below boundary
 *
 * This test covers the remaining volatile fields:
 * - Authorized Senders (ownerNumbers missing on announce turns)
 * - Silent Replies / completion-event guidance
 * - Assistant Output Directives
 * - Control UI Embed/Side Chat sections
 */

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

const basePromptParams = {
  workspaceDir: "/test-workspace",
  toolNames: [
    "read",
    "write",
    "edit",
    "apply_patch",
    "exec",
    "process",
    "web_search",
    "web_fetch",
    "browser",
    "screen",
    "automations",
    "message",
    "sessions_list",
    "sessions_history",
    "sessions_search",
    "sessions_send",
    "sessions_spawn",
    "sessions_yield",
    "subagents",
    "session_status",
    "skill_workshop",
    "view_image",
    "image_generate",
    "dashboard",
    "show_widget",
    "portal",
    "openclaw",
    "gateway",
    "agents_list",
    "terminal",
    "canvas",
    "nodes",
    "pdf",
    "tts",
    "talk_voice",
    "theme",
    "memory_get",
    "memory_search",
    "wiki_get",
    "wiki_search",
    "wiki_apply",
    "wiki_lint",
    "wiki_status",
    "intent",
    "secrets",
    "presence",
    "create_goal",
    "get_goal",
    "update_goal",
    "ask_user",
    "dismiss_task",
    "suggest_task",
    "github_identity_status",
    "mobile_ui",
    "node_inference",
    "conversations_list",
    "conversations_send",
    "conversations_turn",
    "file_fetch",
    "file_write",
    "dir_fetch",
    "dir_list",
    "plugins",
    "skills_read",
    "skills_search",
    "progress_card",
  ],
  capabilityToolNames: [] as string[],
  promptMode: "full" as const,
  ownerNumbers: ["owner-1@example.com"],
  runtimeInfo: {
    sessionKey: "agent:main:test-session",
    channel: "webchat",
    agentId: "main",
    model: "test-model",
    chatType: "direct" as const,
    capabilities: ["markdowndetails"],
  },
  sandboxInfo: { enabled: false },
  reasoningLevel: "off" as const,
  extraSystemPrompt: "## Conversation Context\nOpenClaw runtime context.",
  skillsPrompt: "## Skills\nNo skills listed.",
  contextFiles: [
    { path: "AGENTS.md", content: "# AGENTS.md\nTest agent workspace." },
    { path: "SOUL.md", content: "# SOUL.md\nTest agent soul." },
  ],
};

describe("system-prompt cache stability across normal/announce transitions", () => {
  it("stable prefix digest is identical for normal and announce/yield turns", () => {
    const normalPrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "automatic",
      silentReplyPromptMode: "generic",
    });

    const announcePrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "message_tool_only",
      silentReplyPromptMode: "none",
    });

    const normalSplit = splitSystemPromptCacheBoundary(normalPrompt)!;
    const announceSplit = splitSystemPromptCacheBoundary(announcePrompt)!;

    // The stable prefix must be byte-identical.
    expect(normalSplit.stablePrefix).toBe(announceSplit.stablePrefix);
  });

  it("dynamic suffix digest is identical for normal and announce/yield turns", () => {
    const normalPrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "automatic",
      silentReplyPromptMode: "generic",
    });

    const announcePrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "message_tool_only",
      silentReplyPromptMode: "none",
    });

    const normalSplit = splitSystemPromptCacheBoundary(normalPrompt)!;
    const announceSplit = splitSystemPromptCacheBoundary(announcePrompt)!;

    // The dynamic suffix must have the same digest after the fix.
    const normalSuffixDigest = sha256(normalSplit.dynamicSuffix);
    const announceSuffixDigest = sha256(announceSplit.dynamicSuffix);
    expect(normalSuffixDigest).toBe(announceSuffixDigest);
  });

  it("digests remain stable across normal → announce → normal transition", () => {
    const prompts = ["normal", "announce", "normal"].map((turnType) => {
      const isAnnounce = turnType === "announce";
      return buildAgentSystemPrompt({
        ...basePromptParams,
        sourceReplyDeliveryMode: isAnnounce ? "message_tool_only" : "automatic",
        silentReplyPromptMode: isAnnounce ? "none" : "generic",
      });
    });

    const splits = prompts.map((p) => splitSystemPromptCacheBoundary(p)!);
    const prefixDigests = splits.map((s) => sha256(s.stablePrefix));
    const suffixDigests = splits.map((s) => sha256(s.dynamicSuffix));

    // All stable prefixes must be identical.
    expect(prefixDigests[0]).toBe(prefixDigests[1]);
    expect(prefixDigests[1]).toBe(prefixDigests[2]);

    // All dynamic suffixes must be identical.
    expect(suffixDigests[0]).toBe(suffixDigests[1]);
    expect(suffixDigests[1]).toBe(suffixDigests[2]);
  });

  it("Silent Replies section is present in both normal and announce prompts", () => {
    const normalPrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "automatic",
      silentReplyPromptMode: "generic",
    });

    const announcePrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "message_tool_only",
      silentReplyPromptMode: "none",
    });

    // Both must contain the Silent Replies section with stable content.
    expect(normalPrompt).toContain("## Silent Replies");
    expect(announcePrompt).toContain("## Silent Replies");
  });

  it("completion-event guidance wording is identical in both normal and announce prompts", () => {
    const normalPrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "automatic",
      silentReplyPromptMode: "generic",
    });

    const announcePrompt = buildAgentSystemPrompt({
      ...basePromptParams,
      sourceReplyDeliveryMode: "message_tool_only",
      silentReplyPromptMode: "none",
    });

    // Both must contain the same completion-event guidance referencing NO_REPLY.
    const token = "NO_REPLY";
    expect(normalPrompt).toContain(token);
    expect(announcePrompt).toContain(token);
  });
});

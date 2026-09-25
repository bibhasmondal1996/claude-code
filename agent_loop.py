from __future__ import annotations

from dataclasses import dataclass, replace, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, Protocol
from uuid import uuid4

Message = dict[str, Any]
ToolFn = Callable[[dict[str, Any]], Awaitable[Any] | Any]


@dataclass
class ToolUseContextOptions:
    main_loop_model: str = "default"
    tools: dict[str, ToolFn] = field(default_factory=dict)


@dataclass
class ToolUseContext:
    options: ToolUseContextOptions = field(default_factory=ToolUseContextOptions)


@dataclass
class QueryParams:
    messages: list[Message]
    system_prompt: str
    user_context: dict[str, str]
    system_context: dict[str, str]
    can_use_tool: Callable[[str, dict[str, Any], Any], bool] | None
    tool_use_context: Any
    query_source: str
    fallback_model: Optional[str] = None
    max_output_tokens_override: Optional[int] = None
    max_turns: Optional[int] = None
    skip_cache_write: bool = False
    task_budget: Optional[dict[str, int]] = None


@dataclass
class State:
    messages: list[Message]
    tool_use_context: Any
    auto_compact_tracking: Optional[dict[str, Any]]
    max_output_tokens_recovery_count: int
    has_attempted_reactive_compact: bool
    max_output_tokens_override: Optional[int]
    pending_tool_use_summary: Optional[Any]
    stop_hook_active: Optional[bool]
    turn_count: int
    transition: Optional[dict[str, Any]]


class Deps(Protocol):
    async def microcompact(
        self, messages, tool_use_context, query_source
    ) -> dict[str, Any]: ...

    async def autocompact(
        self,
        messages,
        tool_use_context,
        cache_safe_params,
        query_source,
        tracking,
        snip_tokens_freed=0,
    ) -> dict[str, Any]: ...

    async def call_model(self, request: dict[str, Any]) -> AsyncGenerator[Message, None]: ...

    async def run_tools(
        self, tool_use_blocks, assistant_messages, can_use_tool, tool_use_context
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    async def get_attachment_messages(
        self, updated_context, queued_commands, all_messages, query_source
    ) -> AsyncGenerator[Message, None]: ...

    async def handle_stop_hooks(
        self,
        messages_for_query,
        assistant_messages,
        system_prompt,
        user_context,
        system_context,
        tool_use_context,
        query_source,
        stop_hook_active,
    ) -> dict[str, Any]: ...

    async def try_reactive_compact(
        self, payload: dict[str, Any]
    ) -> Optional[dict[str, Any]]: ...

    def uuid(self) -> str: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def _token_estimate_text(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _tool_use_context_options(tool_use_context: Any) -> ToolUseContextOptions:
    options = getattr(tool_use_context, "options", None)
    if isinstance(options, ToolUseContextOptions):
        return options
    if options is None:
        options = SimpleNamespace(main_loop_model="default", tools={})
        setattr(tool_use_context, "options", options)
    if not hasattr(options, "main_loop_model"):
        setattr(options, "main_loop_model", "default")
    if not hasattr(options, "tools") or not isinstance(options.tools, dict):
        setattr(options, "tools", {})
    return options  # type: ignore[return-value]


def get_messages_after_compact_boundary(messages: list[Message]) -> list[Message]:
    boundary_index = -1
    for i, message in enumerate(messages):
        if message.get("type") == "system" and message.get("subtype") == "compact_boundary":
            boundary_index = i
    return messages[boundary_index + 1 :] if boundary_index >= 0 else list(messages)


def apply_tool_result_budget(messages: list[Message], context: Any, max_chars: int = 4000) -> list[Message]:
    _ = context
    rewritten: list[Message] = []
    for message in messages:
        if message.get("type") != "user" or not isinstance(message.get("content"), list):
            rewritten.append(message)
            continue
        changed = False
        new_blocks: list[dict[str, Any]] = []
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                value = str(block.get("content", ""))
                if len(value) > max_chars:
                    changed = True
                    block = {
                        **block,
                        "content": value[:max_chars] + "\n...[truncated]",
                    }
            new_blocks.append(block)
        rewritten.append({**message, "content": new_blocks} if changed else message)
    return rewritten


def apply_snip_if_needed(
    messages: list[Message], *, keep_last: int = 40
) -> tuple[list[Message], int, Optional[Message]]:
    if len(messages) <= keep_last:
        return messages, 0, None
    dropped = messages[:-keep_last]
    kept = messages[-keep_last:]
    dropped_tokens = token_count_with_estimation(dropped)
    boundary = {
        "type": "system",
        "subtype": "snip_boundary",
        "timestamp": _now(),
        "content": f"Snipped {len(dropped)} older messages.",
        "tokens_freed": dropped_tokens,
    }
    return kept, dropped_tokens, boundary


def calculate_token_warning_state(
    token_usage: int,
    model: str,
    *,
    context_window: int = 200_000,
    blocking_buffer: int = 3_000,
) -> dict[str, bool]:
    _ = model
    blocking_limit = context_window - blocking_buffer
    return {"is_at_blocking_limit": token_usage >= blocking_limit}


def token_count_with_estimation(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += _token_estimate_text(_content_text(message.get("content", "")))
    return total


def is_prompt_too_long(msg: Optional[Message]) -> bool:
    return bool(
        msg and msg.get("type") == "assistant" and msg.get("api_error") == "prompt_too_long"
    )


def is_max_output_tokens(msg: Optional[Message]) -> bool:
    return bool(
        msg and msg.get("type") == "assistant" and msg.get("api_error") == "max_output_tokens"
    )


def create_stream_request_start() -> Message:
    return {"type": "stream_request_start", "timestamp": _now()}


def create_api_error_message(content: str, error: str = "invalid_request") -> Message:
    return {
        "type": "assistant",
        "is_api_error_message": True,
        "content": content,
        "api_error": error,
        "timestamp": _now(),
    }


def create_user_message(content: str, *, is_meta: bool = False) -> Message:
    return {
        "type": "user",
        "content": content,
        "is_meta": is_meta,
        "timestamp": _now(),
    }


def build_post_compact_messages(compaction_result: dict[str, Any]) -> list[Message]:
    return [
        compaction_result["boundary_marker"],
        *compaction_result.get("summary_messages", []),
        *compaction_result.get("messages_to_keep", []),
        *compaction_result.get("attachments", []),
        *compaction_result.get("hook_results", []),
    ]


class DefaultDeps:
    def __init__(self, *, auto_compact_threshold: int = 120_000, keep_recent_after_compact: int = 10):
        self.auto_compact_threshold = auto_compact_threshold
        self.keep_recent_after_compact = keep_recent_after_compact

    async def microcompact(self, messages, tool_use_context, query_source) -> dict[str, Any]:
        _ = tool_use_context, query_source
        compacted = apply_tool_result_budget(messages, tool_use_context)
        return {"messages": compacted}

    async def autocompact(
        self,
        messages,
        tool_use_context,
        cache_safe_params,
        query_source,
        tracking,
        snip_tokens_freed=0,
    ) -> dict[str, Any]:
        _ = tool_use_context, cache_safe_params, query_source, snip_tokens_freed
        total_tokens = token_count_with_estimation(messages)
        if total_tokens < self.auto_compact_threshold:
            return {"compaction_result": None, "consecutive_failures": None}
        try:
            compaction = self._compact_messages(messages)
            return {"compaction_result": compaction, "consecutive_failures": 0}
        except Exception:
            previous_failures = int((tracking or {}).get("consecutive_failures", 0))
            return {"compaction_result": None, "consecutive_failures": previous_failures + 1}

    async def call_model(self, request: dict[str, Any]) -> AsyncGenerator[Message, None]:
        messages = request.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.get("type") == "user"), None)
        last_text = _content_text(last_user.get("content", "")) if last_user else ""

        if "PROMPT_TOO_LONG" in last_text:
            yield create_api_error_message("prompt too long", "prompt_too_long")
            return
        if "MAX_OUTPUT" in last_text:
            yield create_api_error_message("output token limit hit", "max_output_tokens")
            return

        if "USE_TOOL:" in last_text:
            tool_name = last_text.split("USE_TOOL:", 1)[1].strip().split()[0]
            tool_input: dict[str, Any] = {"raw": last_text}
            yield {
                "type": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"toolu_{uuid4().hex[:12]}",
                        "name": tool_name,
                        "input": tool_input,
                    }
                ],
                "timestamp": _now(),
            }
            return

        yield {
            "type": "assistant",
            "content": f"Processed: {last_text}" if last_text else "Processed.",
            "timestamp": _now(),
        }

    async def run_tools(
        self, tool_use_blocks, assistant_messages, can_use_tool, tool_use_context
    ) -> AsyncGenerator[dict[str, Any], None]:
        _ = assistant_messages
        options = _tool_use_context_options(tool_use_context)
        tools = options.tools
        for block in tool_use_blocks:
            name = block.get("name", "")
            tool_input = block.get("input", {})
            allow = True
            if can_use_tool is not None:
                allow = bool(can_use_tool(name, tool_input, tool_use_context))
            if not allow:
                result_content = f"Tool denied: {name}"
            else:
                tool = tools.get(name)
                if tool is None:
                    result_content = f"Tool not found: {name}"
                else:
                    output = tool(tool_input)
                    if hasattr(output, "__await__"):
                        output = await output
                    result_content = str(output)
            message = {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("id", ""),
                        "content": result_content,
                        "is_error": False,
                    }
                ],
                "timestamp": _now(),
            }
            yield {"message": message, "new_context": tool_use_context}

    async def get_attachment_messages(
        self, updated_context, queued_commands, all_messages, query_source
    ) -> AsyncGenerator[Message, None]:
        _ = updated_context, all_messages, query_source
        for command in queued_commands:
            yield {
                "type": "attachment",
                "attachment": {
                    "type": "queued_command",
                    "command": command,
                },
                "timestamp": _now(),
            }

    async def handle_stop_hooks(
        self,
        messages_for_query,
        assistant_messages,
        system_prompt,
        user_context,
        system_context,
        tool_use_context,
        query_source,
        stop_hook_active,
    ) -> dict[str, Any]:
        _ = (
            messages_for_query,
            system_prompt,
            user_context,
            system_context,
            tool_use_context,
            query_source,
            stop_hook_active,
        )
        text = " ".join(_content_text(msg.get("content", "")) for msg in assistant_messages)
        if "[STOP]" in text:
            return {"prevent_continuation": True, "blocking_errors": []}
        if "[RETRY]" in text:
            return {
                "prevent_continuation": False,
                "blocking_errors": [create_user_message("Retry requested by stop-hook", is_meta=True)],
            }
        return {"prevent_continuation": False, "blocking_errors": []}

    async def try_reactive_compact(
        self, payload: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        if payload.get("has_attempted"):
            return None
        messages = payload.get("messages", [])
        if not messages:
            return None
        return self._compact_messages(messages)

    def uuid(self) -> str:
        return str(uuid4())

    def _compact_messages(self, messages: list[Message]) -> dict[str, Any]:
        keep_count = min(self.keep_recent_after_compact, max(1, len(messages) // 4))
        keep = messages[-keep_count:]
        summarize = messages[:-keep_count]
        summary_text = self._summarize_messages(summarize)
        boundary = {
            "type": "system",
            "subtype": "compact_boundary",
            "content": f"Compacted {len(summarize)} messages",
            "timestamp": _now(),
        }
        summary_message = create_user_message(f"Conversation summary: {summary_text}", is_meta=True)
        return {
            "boundary_marker": boundary,
            "summary_messages": [summary_message],
            "messages_to_keep": keep,
            "attachments": [],
            "hook_results": [],
        }

    def _summarize_messages(self, messages: list[Message], limit: int = 8) -> str:
        if not messages:
            return "(no prior messages)"
        sampled = messages[-limit:]
        fragments = []
        for message in sampled:
            role = message.get("type", "message")
            text = _content_text(message.get("content", "")).strip().replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            fragments.append(f"{role}: {text}")
        return " | ".join(fragments)


async def query_loop(params: QueryParams, deps: Deps) -> AsyncGenerator[Message, None]:
    state = State(
        messages=params.messages,
        tool_use_context=params.tool_use_context,
        auto_compact_tracking=None,
        max_output_tokens_recovery_count=0,
        has_attempted_reactive_compact=False,
        max_output_tokens_override=params.max_output_tokens_override,
        pending_tool_use_summary=None,
        stop_hook_active=None,
        turn_count=1,
        transition=None,
    )
    task_budget_remaining: Optional[int] = None

    while True:
        tool_use_context = state.tool_use_context
        messages = state.messages
        tracking = state.auto_compact_tracking
        turn_count = state.turn_count

        yield create_stream_request_start()

        messages_for_query = get_messages_after_compact_boundary(messages)
        messages_for_query = apply_tool_result_budget(messages_for_query, tool_use_context)
        messages_for_query, snip_tokens_freed, snip_boundary = apply_snip_if_needed(messages_for_query)
        if snip_boundary:
            yield snip_boundary

        microcompact_result = await deps.microcompact(
            messages_for_query, tool_use_context, params.query_source
        )
        messages_for_query = microcompact_result["messages"]

        autocompact_result = await deps.autocompact(
            messages_for_query,
            tool_use_context,
            {
                "system_prompt": params.system_prompt,
                "user_context": params.user_context,
                "system_context": params.system_context,
                "tool_use_context": tool_use_context,
                "fork_context_messages": messages_for_query,
            },
            params.query_source,
            tracking,
            snip_tokens_freed,
        )

        compaction_result = autocompact_result.get("compaction_result")
        if compaction_result:
            if params.task_budget:
                pre_compact_context = token_count_with_estimation(messages_for_query)
                task_budget_remaining = max(
                    0,
                    (task_budget_remaining or params.task_budget["total"]) - pre_compact_context,
                )

            tracking = {
                "compacted": True,
                "turn_id": deps.uuid(),
                "turn_counter": 0,
                "consecutive_failures": 0,
            }

            post_compact_messages = build_post_compact_messages(compaction_result)
            for message in post_compact_messages:
                yield message
            messages_for_query = post_compact_messages
        elif autocompact_result.get("consecutive_failures") is not None:
            tracking = {
                **(tracking or {"compacted": False, "turn_id": "", "turn_counter": 0}),
                "consecutive_failures": autocompact_result["consecutive_failures"],
            }

        options = _tool_use_context_options(tool_use_context)
        token_state = calculate_token_warning_state(
            token_count_with_estimation(messages_for_query) - snip_tokens_freed,
            options.main_loop_model,
        )
        if token_state["is_at_blocking_limit"] and not compaction_result:
            yield create_api_error_message("prompt too long", "invalid_request")
            return

        assistant_messages: list[Message] = []
        tool_results: list[Message] = []
        tool_use_blocks: list[Message] = []
        needs_follow_up = False
        attempt_with_fallback = True
        current_model = options.main_loop_model

        while attempt_with_fallback:
            attempt_with_fallback = False
            try:
                async for message in deps.call_model(
                    {
                        "messages": messages_for_query,
                        "system_prompt": params.system_prompt,
                        "user_context": params.user_context,
                        "system_context": params.system_context,
                        "tool_use_context": tool_use_context,
                        "model": current_model,
                        "fallback_model": params.fallback_model,
                        "max_output_tokens_override": state.max_output_tokens_override,
                        "task_budget_remaining": task_budget_remaining,
                    }
                ):
                    withheld = is_prompt_too_long(message) or is_max_output_tokens(message)
                    if not withheld:
                        yield message

                    if message.get("type") == "assistant":
                        assistant_messages.append(message)
                        content = message.get("content", [])
                        if isinstance(content, list):
                            blocks = [
                                block
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "tool_use"
                            ]
                            if blocks:
                                tool_use_blocks.extend(blocks)
                                needs_follow_up = True
            except Exception as error:
                if params.fallback_model and "fallback_triggered" in str(error):
                    current_model = params.fallback_model
                    attempt_with_fallback = True
                    assistant_messages.clear()
                    tool_results.clear()
                    tool_use_blocks.clear()
                    needs_follow_up = False
                    continue

                yield create_api_error_message(str(error), "model_error")
                return

        if not needs_follow_up:
            last_message = assistant_messages[-1] if assistant_messages else None

            if is_prompt_too_long(last_message):
                compacted = await deps.try_reactive_compact(
                    {
                        "has_attempted": state.has_attempted_reactive_compact,
                        "query_source": params.query_source,
                        "messages": messages_for_query,
                        "cache_safe_params": {
                            "system_prompt": params.system_prompt,
                            "user_context": params.user_context,
                            "system_context": params.system_context,
                            "tool_use_context": tool_use_context,
                            "fork_context_messages": messages_for_query,
                        },
                    }
                )
                if compacted:
                    post_compact_messages = build_post_compact_messages(compacted)
                    for msg in post_compact_messages:
                        yield msg

                    state = replace(
                        state,
                        messages=post_compact_messages,
                        auto_compact_tracking=None,
                        has_attempted_reactive_compact=True,
                        pending_tool_use_summary=None,
                        stop_hook_active=None,
                        transition={"reason": "reactive_compact_retry"},
                    )
                    continue

                if last_message:
                    yield last_message
                return

            if is_max_output_tokens(last_message):
                if state.max_output_tokens_recovery_count < 3:
                    recovery_message = create_user_message(
                        "Continue directly, no recap.", is_meta=True
                    )
                    state = replace(
                        state,
                        messages=[*messages_for_query, *assistant_messages, recovery_message],
                        auto_compact_tracking=tracking,
                        max_output_tokens_recovery_count=state.max_output_tokens_recovery_count
                        + 1,
                        pending_tool_use_summary=None,
                        transition={"reason": "max_output_tokens_recovery"},
                    )
                    continue

                if last_message:
                    yield last_message
                return

            stop_result = await deps.handle_stop_hooks(
                messages_for_query,
                assistant_messages,
                params.system_prompt,
                params.user_context,
                params.system_context,
                tool_use_context,
                params.query_source,
                state.stop_hook_active,
            )
            if stop_result.get("prevent_continuation"):
                return
            if stop_result.get("blocking_errors"):
                state = replace(
                    state,
                    messages=[
                        *messages_for_query,
                        *assistant_messages,
                        *stop_result["blocking_errors"],
                    ],
                    auto_compact_tracking=tracking,
                    pending_tool_use_summary=None,
                    stop_hook_active=True,
                    transition={"reason": "stop_hook_blocking"},
                )
                continue
            return

        async for update in deps.run_tools(
            tool_use_blocks,
            assistant_messages,
            params.can_use_tool,
            tool_use_context,
        ):
            if update.get("message"):
                msg = update["message"]
                yield msg
                tool_results.append(msg)
            if update.get("new_context"):
                tool_use_context = update["new_context"]

        queued_commands: list[dict[str, Any]] = []
        async for attachment in deps.get_attachment_messages(
            tool_use_context,
            queued_commands,
            [*messages_for_query, *assistant_messages, *tool_results],
            params.query_source,
        ):
            yield attachment
            tool_results.append(attachment)

        next_turn_count = turn_count + 1
        if params.max_turns and next_turn_count > params.max_turns:
            yield {
                "type": "attachment",
                "attachment": {
                    "type": "max_turns_reached",
                    "max_turns": params.max_turns,
                    "turn_count": next_turn_count,
                },
            }
            return

        state = replace(
            state,
            messages=[*messages_for_query, *assistant_messages, *tool_results],
            tool_use_context=tool_use_context,
            auto_compact_tracking=tracking,
            turn_count=next_turn_count,
            max_output_tokens_recovery_count=0,
            has_attempted_reactive_compact=False,
            pending_tool_use_summary=None,
            max_output_tokens_override=None,
            transition={"reason": "next_turn"},
        )

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, AsyncGenerator, Optional, Protocol


Message = dict[str, Any]


@dataclass
class QueryParams:
    messages: list[Message]
    system_prompt: str
    user_context: dict[str, str]
    system_context: dict[str, str]
    can_use_tool: Any
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

    async def try_reactive_compact(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]: ...

    def uuid(self) -> str: ...


def get_messages_after_compact_boundary(messages: list[Message]) -> list[Message]:
    return messages


def apply_tool_result_budget(messages, context) -> list[Message]:
    return messages


def apply_snip_if_needed(messages) -> tuple[list[Message], int, Optional[Message]]:
    return messages, 0, None


def calculate_token_warning_state(token_usage: int, model: str) -> dict[str, bool]:
    return {"is_at_blocking_limit": False}


def token_count_with_estimation(messages) -> int:
    return 0


def is_prompt_too_long(msg: Optional[Message]) -> bool:
    return bool(
        msg and msg.get("type") == "assistant" and msg.get("api_error") == "prompt_too_long"
    )


def is_max_output_tokens(msg: Optional[Message]) -> bool:
    return bool(
        msg and msg.get("type") == "assistant" and msg.get("api_error") == "max_output_tokens"
    )


def create_stream_request_start() -> Message:
    return {"type": "stream_request_start"}


def create_api_error_message(content: str, error: str = "invalid_request") -> Message:
    return {"type": "assistant", "is_api_error_message": True, "content": content, "api_error": error}


def build_post_compact_messages(compaction_result: dict[str, Any]) -> list[Message]:
    return [
        compaction_result["boundary_marker"],
        *compaction_result["summary_messages"],
        *compaction_result.get("messages_to_keep", []),
        *compaction_result.get("attachments", []),
        *compaction_result.get("hook_results", []),
    ]


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

        token_state = calculate_token_warning_state(
            token_count_with_estimation(messages_for_query) - snip_tokens_freed,
            getattr(tool_use_context.options, "main_loop_model", "default"),
        )
        if token_state["is_at_blocking_limit"] and not compaction_result:
            yield create_api_error_message("prompt too long", "invalid_request")
            return

        assistant_messages: list[Message] = []
        tool_results: list[Message] = []
        tool_use_blocks: list[Message] = []
        needs_follow_up = False
        attempt_with_fallback = True
        current_model = getattr(tool_use_context.options, "main_loop_model", None)

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
                        blocks = [
                            block
                            for block in message.get("content", [])
                            if block.get("type") == "tool_use"
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
                    recovery_message: Message = {
                        "type": "user",
                        "is_meta": True,
                        "content": "Continue directly, no recap.",
                    }
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

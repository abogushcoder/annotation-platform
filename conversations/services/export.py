import json
import random
import uuid
from datetime import datetime

from conversations.models import Conversation, Turn, ToolCall, SystemPrompt


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create a pickup order for the customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customerName": {"type": "string", "description": "Customer's name"},
                    "customerPhone": {"type": "string", "description": "Customer's phone number"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "itemName": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "modifiers": {"type": "array", "items": {"type": "string"}},
                                "specialInstructions": {"type": "string"},
                            },
                            "required": ["itemName", "quantity"],
                        },
                    },
                    "specialInstructions": {"type": "string", "description": "Special instructions for the whole order"},
                },
                "required": ["customerName", "customerPhone", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an existing order",
            "parameters": {
                "type": "object",
                "properties": {
                    "orderId": {"type": "string", "description": "The order ID to cancel"},
                    "reason": {"type": "string", "description": "Reason for cancellation"},
                },
                "required": ["orderId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove an item from an existing order",
            "parameters": {
                "type": "object",
                "properties": {
                    "orderId": {"type": "string", "description": "The order ID"},
                    "itemName": {"type": "string", "description": "Name of the item to remove"},
                },
                "required": ["orderId", "itemName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_item",
            "description": "Modify an item in an existing order",
            "parameters": {
                "type": "object",
                "properties": {
                    "orderId": {"type": "string", "description": "The order ID"},
                    "itemName": {"type": "string", "description": "Name of the item to modify"},
                    "modifications": {"type": "string", "description": "Description of modifications"},
                },
                "required": ["orderId", "itemName", "modifications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check table availability for a reservation",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format"},
                    "partySize": {"type": "integer", "description": "Number of guests"},
                },
                "required": ["date", "time", "partySize"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reservation",
            "description": "Create a table reservation",
            "parameters": {
                "type": "object",
                "properties": {
                    "customerName": {"type": "string", "description": "Customer's name"},
                    "customerPhone": {"type": "string", "description": "Customer's phone number"},
                    "partySize": {"type": "integer", "description": "Number of guests"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format"},
                    "specialRequests": {"type": "string", "description": "Any special requests"},
                },
                "required": ["customerName", "customerPhone", "partySize", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_specials",
            "description": "Get today's specials and promotions",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_past_orders",
            "description": "Look up a customer's past orders",
            "parameters": {
                "type": "object",
                "properties": {
                    "customerPhone": {"type": "string", "description": "Customer's phone number"},
                },
                "required": ["customerPhone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the current phone call",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_menu_link",
            "description": "Send a link to the online menu via SMS",
            "parameters": {
                "type": "object",
                "properties": {
                    "customerPhone": {"type": "string", "description": "Customer's phone number"},
                },
                "required": ["customerPhone"],
            },
        },
    },
]


def _get_tools_used(turns):
    """Get set of tool names used in the conversation."""
    tools_used = set()
    for turn in turns:
        for tc in turn.tool_calls.all():
            tools_used.add(tc.tool_name)
    return tools_used


def conversation_to_messages(conversation, include_system_prompt=True, include_tools=True):
    """Convert a Conversation to OpenAI fine-tuning message format.

    Returns dict with 'messages', 'tools', 'parallel_tool_calls' keys.
    """
    messages = []
    call_counter = 0
    seen_user = False

    # System prompt
    if include_system_prompt:
        active_prompt = SystemPrompt.objects.filter(is_active=True).first()
        if active_prompt:
            messages.append({
                "role": "system",
                "content": active_prompt.content,
            })

    turns = list(conversation.turns.prefetch_related('tool_calls').all())

    for turn in turns:
        if turn.role == "user":
            seen_user = True

        tool_calls_for_turn = list(turn.tool_calls.all())

        # If this turn has tool calls, we need special handling
        if tool_calls_for_turn:
            tc_entries = []
            tc_responses = []
            for tc in tool_calls_for_turn:
                call_counter += 1
                call_id = f"call_{call_counter:03d}"
                args = tc.display_args

                tc_entries.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(args),
                    },
                })

                # Build tool response
                response_content = tc.response_body
                if response_content is None:
                    response_content = json.dumps({"status": "ok"})
                elif isinstance(response_content, (dict, list)):
                    if isinstance(response_content, dict) and not response_content:
                        response_content = json.dumps({"status": "ok"})
                    else:
                        response_content = json.dumps(response_content)
                elif not isinstance(response_content, str):
                    response_content = str(response_content)

                tc_responses.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": response_content,
                })

            # Combine text content and tool_calls into a single assistant message
            assistant_msg = {
                "role": "assistant",
                "tool_calls": tc_entries,
            }
            text = turn.display_text.strip()
            if text:
                assistant_msg["content"] = text
            if not seen_user:
                assistant_msg["weight"] = 0
            messages.append(assistant_msg)

            # Add tool responses
            messages.extend(tc_responses)

        else:
            # Regular message (no tool calls)
            text = turn.display_text.strip()
            if not text:
                continue

            role = "user" if turn.role == "user" else "assistant"
            msg = {
                "role": role,
                "content": text,
            }
            if role == "assistant" and not seen_user:
                msg["weight"] = 0
            messages.append(msg)

    result = {
        "messages": messages,
        "parallel_tool_calls": False,
    }

    if include_tools:
        tools_used = _get_tools_used(turns)
        if tools_used:
            result["tools"] = [
                t for t in TOOL_DEFINITIONS
                if t["function"]["name"] in tools_used
            ]

    return result


MAX_EXAMPLE_TOKENS = 65536


def validate_example(example):
    """Validate a single training example. Returns list of error strings."""
    errors = []
    msgs = example.get("messages", [])

    if not msgs:
        errors.append("No messages")
        return errors

    has_user = any(m.get("role") == "user" for m in msgs)
    has_assistant = any(m.get("role") == "assistant" for m in msgs)

    if not has_user:
        errors.append("Missing user message")
    if not has_assistant:
        errors.append("Missing assistant message")

    # Last message must be assistant for the model to learn a final response
    last_msg = msgs[-1]
    if last_msg.get("role") != "assistant":
        errors.append(f"Last message must be assistant (got {last_msg.get('role')})")

    # Token limit check (rough estimate — 3 chars/token for JSON-heavy content)
    example_chars = len(json.dumps(example))
    estimated_tokens = example_chars // 3
    if estimated_tokens > MAX_EXAMPLE_TOKENS:
        errors.append(
            f"Example exceeds token limit (~{estimated_tokens} tokens, max {MAX_EXAMPLE_TOKENS})"
        )

    tool_call_ids = set()
    pending_tool_call_ids = set()

    for i, msg in enumerate(msgs):
        role = msg.get("role")

        # First message must be system or user
        if i == 0 and role not in ("system", "user"):
            errors.append("First message must be system or user")

        # Check empty content
        if role in ("user", "system") and not msg.get("content", "").strip():
            errors.append(f"Empty content in {role} message")

        # Validate tool calls
        if "tool_calls" in msg:
            # If there are pending unmatched tool_call_ids from a previous block, error
            if pending_tool_call_ids:
                errors.append(f"Unmatched tool_call_ids: {pending_tool_call_ids}")
            pending_tool_call_ids = set()

            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id in tool_call_ids:
                    errors.append(f"Duplicate tool_call_id: {tc_id}")
                tool_call_ids.add(tc_id)
                pending_tool_call_ids.add(tc_id)

                args_str = tc.get("function", {}).get("arguments", "")
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"Invalid JSON args in {tc.get('function', {}).get('name')}")

        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if not pending_tool_call_ids:
                errors.append("Orphaned tool response (no preceding tool_calls)")
            elif tc_id not in pending_tool_call_ids:
                errors.append(f"tool_call_id '{tc_id}' not in preceding tool_calls")
            else:
                pending_tool_call_ids.discard(tc_id)
            tool_content = msg.get("content", "")
            if not tool_content or not tool_content.strip():
                errors.append("Empty content in tool response")

        else:
            # Non-tool message encountered — any pending IDs are unmatched
            if pending_tool_call_ids:
                errors.append(f"Unmatched tool_call_ids: {pending_tool_call_ids}")
                pending_tool_call_ids = set()

    # After loop — check for trailing unmatched
    if pending_tool_call_ids:
        errors.append(f"Unmatched tool_call_ids at end: {pending_tool_call_ids}")

    return errors


def validate_dataset(examples):
    """Validate the full dataset. Returns list of warning strings."""
    warnings = []
    if len(examples) < 10:
        warnings.append(
            f"OpenAI requires at least 10 training examples. "
            f"You have {len(examples)}. The fine-tuning job will be rejected."
        )
    return warnings


def generate_jsonl_examples(limit=None, agent_id=None, tool_calls_only=False,
                             include_system_prompt=True, include_tools=True):
    """Generate JSONL examples from approved conversations.

    Returns list of dicts (each dict is one training example).
    """
    qs = Conversation.objects.filter(status='approved')
    if agent_id:
        qs = qs.filter(agent_id=agent_id)
    if tool_calls_only:
        qs = qs.filter(turns__tool_calls__isnull=False).distinct()

    conversations = qs.prefetch_related('turns__tool_calls')
    if limit:
        conversations = conversations[:limit]

    examples = []
    for conv in conversations:
        example = conversation_to_messages(conv, include_system_prompt, include_tools)
        validation_errors = validate_example(example)
        if not validation_errors:
            examples.append(example)

    return examples


def count_tokens(examples):
    """Estimate token count for training examples using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        total = 0
        for ex in examples:
            text = json.dumps(ex)
            total += len(enc.encode(text))
        return total
    except Exception:
        # Rough estimate: 1 token per 4 chars
        total_chars = sum(len(json.dumps(ex)) for ex in examples)
        return total_chars // 4


def estimate_training_cost(token_count, epochs=3):
    """Estimate training cost for gpt-4o fine-tuning.

    Current pricing: $25 per 1M training tokens.
    """
    cost_per_million = 25.0
    total_training_tokens = token_count * epochs
    cost = (total_training_tokens / 1_000_000) * cost_per_million
    return round(cost, 2)


def export_jsonl(examples):
    """Convert examples list to JSONL string."""
    lines = [json.dumps(ex, ensure_ascii=False) for ex in examples]
    return "\n".join(lines) + "\n"


def split_train_validation(examples, train_ratio=0.8):
    """Split examples into train and validation sets."""
    shuffled = examples.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]

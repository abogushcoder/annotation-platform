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
        tool_calls_for_turn = list(turn.tool_calls.all())

        # If this turn has tool calls, we need special handling
        if tool_calls_for_turn:
            # First, if there's text content, emit it as an assistant message
            text = turn.display_text.strip()
            if text:
                messages.append({
                    "role": "assistant",
                    "content": text,
                })

            # Then emit the tool call(s) as an assistant message with tool_calls array
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
                if isinstance(response_content, dict):
                    response_content = json.dumps(response_content)
                elif not isinstance(response_content, str):
                    response_content = str(response_content)

                tc_responses.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": response_content,
                })

            messages.append({
                "role": "assistant",
                "tool_calls": tc_entries,
            })

            # Add tool responses
            messages.extend(tc_responses)

        else:
            # Regular message (no tool calls)
            text = turn.display_text.strip()
            if not text:
                continue

            role = "user" if turn.role == "user" else "assistant"
            messages.append({
                "role": role,
                "content": text,
            })

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

    tool_call_ids = set()
    for msg in msgs:
        # Check empty content
        if msg.get("role") in ("user", "system") and not msg.get("content", "").strip():
            errors.append(f"Empty content in {msg['role']} message")

        # Validate tool calls
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id in tool_call_ids:
                    errors.append(f"Duplicate tool_call_id: {tc_id}")
                tool_call_ids.add(tc_id)

                args_str = tc.get("function", {}).get("arguments", "")
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"Invalid JSON args in {tc.get('function', {}).get('name')}")

    return errors


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

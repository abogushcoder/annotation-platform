import json
import logging
from datetime import datetime, timezone

from django.utils import timezone as django_tz

from conversations.models import Agent, Conversation, Turn, ToolCall
from .elevenlabs import ElevenLabsClient

logger = logging.getLogger(__name__)


def sync_agent_conversations(agent: Agent) -> dict:
    """Sync conversations from ElevenLabs for a given agent.

    Returns dict with counts: {'imported': N, 'skipped': N, 'errors': N}
    """
    client = ElevenLabsClient(agent.elevenlabs_api_key)
    stats = {'imported': 0, 'skipped': 0, 'errors': 0}
    cursor = None

    while True:
        try:
            data = client.list_conversations(agent.agent_id, cursor=cursor)
        except Exception as e:
            logger.error(f"Failed to list conversations for agent {agent.label}: {e}")
            stats['errors'] += 1
            break

        conversations = data.get('conversations', [])
        if not conversations:
            break

        for conv_summary in conversations:
            conv_id = conv_summary.get('conversation_id', '')
            if not conv_id:
                continue

            if Conversation.objects.filter(elevenlabs_id=conv_id).exists():
                stats['skipped'] += 1
                continue

            try:
                conv_detail = client.get_conversation(conv_id)
                _import_conversation(agent, conv_id, conv_detail)
                stats['imported'] += 1
            except Exception as e:
                logger.error(f"Failed to import conversation {conv_id}: {e}")
                stats['errors'] += 1

        cursor = data.get('cursor')
        if not cursor:
            break

    agent.last_synced_at = django_tz.now()
    agent.save()

    return stats


def _import_conversation(agent: Agent, conv_id: str, data: dict):
    """Create Conversation, Turn, and ToolCall records from ElevenLabs data."""
    metadata = data.get('metadata', {})

    call_timestamp = None
    start_time = metadata.get('start_time_unix_secs')
    if start_time:
        call_timestamp = datetime.fromtimestamp(start_time, tz=timezone.utc)

    conversation = Conversation.objects.create(
        elevenlabs_id=conv_id,
        agent=agent,
        status='unassigned',
        call_duration_secs=metadata.get('call_duration_secs'),
        call_timestamp=call_timestamp,
        has_audio=data.get('has_audio', False),
        raw_data=data,
    )

    transcript = data.get('transcript', [])
    for position, turn_data in enumerate(transcript):
        role = turn_data.get('role', 'user')
        if role not in ('user', 'agent'):
            role = 'user'

        turn = Turn.objects.create(
            conversation=conversation,
            position=position,
            role=role,
            original_text=turn_data.get('message', ''),
            time_in_call_secs=turn_data.get('time_in_call_secs'),
        )

        tool_calls = turn_data.get('tool_calls', [])
        for tc_data in tool_calls:
            # Parse args from params or request_headers_body
            original_args = tc_data.get('params', {})
            if not original_args:
                raw_body = tc_data.get('request_headers_body', '')
                if raw_body:
                    try:
                        original_args = json.loads(raw_body)
                    except (json.JSONDecodeError, TypeError):
                        original_args = {}

            # Parse response body
            response_body = {}
            raw_response = tc_data.get('response_body', '')
            if isinstance(raw_response, str) and raw_response:
                try:
                    response_body = json.loads(raw_response)
                except (json.JSONDecodeError, TypeError):
                    response_body = {'raw': raw_response}
            elif isinstance(raw_response, dict):
                response_body = raw_response

            ToolCall.objects.create(
                turn=turn,
                tool_name=tc_data.get('tool_name', 'unknown'),
                original_args=original_args,
                status_code=tc_data.get('status_code'),
                response_body=response_body,
                error_message=tc_data.get('error_message', '') or '',
            )

    return conversation

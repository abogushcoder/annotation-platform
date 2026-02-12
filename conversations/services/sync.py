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
                _import_conversation(agent, conv_id, conv_detail, client)
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


def _import_conversation(agent: Agent, conv_id: str, data: dict, client: ElevenLabsClient):
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

    # Build a map of tool results by request_id from all turns.
    # ElevenLabs puts tool_results in a separate turn after the tool_calls turn.
    tool_results_map = {}
    for turn_data in transcript:
        for tr in turn_data.get('tool_results', []):
            req_id = tr.get('request_id', '')
            if req_id:
                tool_results_map[req_id] = tr

    for position, turn_data in enumerate(transcript):
        role = turn_data.get('role', 'user')
        if role not in ('user', 'agent'):
            role = 'user'

        turn = Turn.objects.create(
            conversation=conversation,
            position=position,
            role=role,
            original_text=turn_data.get('message') or '',
            time_in_call_secs=turn_data.get('time_in_call_secs'),
        )

        # Extract RAG context if present
        rag_info = turn_data.get('rag_retrieval_info')
        if rag_info:
            chunks_meta = rag_info.get('chunks', [])
            rag_chunks = []
            for chunk_meta in chunks_meta:
                doc_id = chunk_meta.get('document_id', '')
                chunk_id = chunk_meta.get('chunk_id', '')
                distance = chunk_meta.get('vector_distance')
                if doc_id and chunk_id:
                    try:
                        chunk_data = client.get_kb_chunk(doc_id, chunk_id)
                        rag_chunks.append({
                            'document_id': doc_id,
                            'chunk_id': chunk_id,
                            'content': chunk_data.get('content', ''),
                            'vector_distance': distance,
                        })
                    except Exception as e:
                        logger.warning(f"Failed to fetch KB chunk {doc_id}/{chunk_id}: {e}")
                        rag_chunks.append({
                            'document_id': doc_id,
                            'chunk_id': chunk_id,
                            'content': '',
                            'vector_distance': distance,
                            'fetch_error': str(e),
                        })
            if rag_chunks:
                turn.rag_context = rag_chunks
                turn.save(update_fields=['rag_context'])

        tool_calls = turn_data.get('tool_calls', [])
        for tc_data in tool_calls:
            # Parse args: try params_as_json first, then tool_details.body,
            # then legacy params/request_headers_body
            original_args = {}
            params_json = tc_data.get('params_as_json', '')
            if params_json:
                try:
                    original_args = json.loads(params_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            if not original_args:
                tool_details = tc_data.get('tool_details', {})
                if tool_details and tool_details.get('body'):
                    try:
                        original_args = json.loads(tool_details['body'])
                    except (json.JSONDecodeError, TypeError):
                        pass

            if not original_args:
                original_args = tc_data.get('params', {})
                if not original_args:
                    raw_body = tc_data.get('request_headers_body', '')
                    if raw_body:
                        try:
                            original_args = json.loads(raw_body)
                        except (json.JSONDecodeError, TypeError):
                            original_args = {}

            # Strip internal system__ keys from args
            original_args = {
                k: v for k, v in original_args.items()
                if not k.startswith('system__')
            }

            # Match tool result by request_id
            request_id = tc_data.get('request_id', '')
            result = tool_results_map.get(request_id, {})

            response_body = {}
            raw_result = result.get('result_value', '')
            if isinstance(raw_result, str) and raw_result:
                try:
                    response_body = json.loads(raw_result)
                except (json.JSONDecodeError, TypeError):
                    response_body = {'raw': raw_result}
            elif isinstance(raw_result, dict):
                response_body = raw_result

            # Fall back to legacy response_body on the tool call itself
            if not response_body:
                raw_response = tc_data.get('response_body', '')
                if isinstance(raw_response, str) and raw_response:
                    try:
                        response_body = json.loads(raw_response)
                    except (json.JSONDecodeError, TypeError):
                        response_body = {'raw': raw_response}
                elif isinstance(raw_response, dict):
                    response_body = raw_response

            error_msg = result.get('error_type', '') or tc_data.get('error_message', '') or ''

            ToolCall.objects.create(
                turn=turn,
                tool_name=tc_data.get('tool_name', 'unknown'),
                original_args=original_args,
                status_code=tc_data.get('status_code'),
                response_body=response_body,
                error_message=error_msg,
            )

    return conversation

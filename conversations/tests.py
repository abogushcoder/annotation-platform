import json
import requests
from unittest.mock import patch, MagicMock
from django.db import models, IntegrityError
from django.test import TestCase, Client
from django.utils import timezone

from accounts.models import User
from conversations.models import Agent, Conversation, Turn, ToolCall, SystemPrompt, ExportLog
from conversations.models import Tag
from conversations.services.export import (
    conversation_to_messages, validate_example, validate_dataset,
    generate_jsonl_examples, export_jsonl, split_train_validation,
    count_tokens, estimate_training_cost, TOOL_DEFINITIONS,
    MAX_EXAMPLE_TOKENS,
)
from conversations.services.sync import sync_agent_conversations, _import_conversation


# =============================================================================
# MODEL TESTS
# =============================================================================

class ModelTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user(
            username='admin', password='admin', role='admin'
        )
        self.annotator = User.objects.create_user(
            username='annotator', password='annotator', role='annotator'
        )
        self.agent = Agent.objects.create(
            agent_id='agent_test', label='Test Agent', elevenlabs_api_key='test-key'
        )

    def test_user_roles(self):
        self.assertTrue(self.admin.is_admin())
        self.assertFalse(self.admin.is_annotator())
        self.assertTrue(self.annotator.is_annotator())
        self.assertFalse(self.annotator.is_admin())

    def test_user_default_role_is_annotator(self):
        user = User.objects.create_user(username='default_user', password='pass')
        self.assertEqual(user.role, 'annotator')
        self.assertTrue(user.is_annotator())

    def test_conversation_creation(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_001', agent=self.agent, status='unassigned'
        )
        self.assertEqual(conv.status, 'unassigned')
        self.assertIsNone(conv.assigned_to)

    def test_conversation_unique_elevenlabs_id(self):
        Conversation.objects.create(elevenlabs_id='conv_dup', agent=self.agent)
        with self.assertRaises(IntegrityError):
            Conversation.objects.create(elevenlabs_id='conv_dup', agent=self.agent)

    def test_conversation_all_statuses(self):
        statuses = ['unassigned', 'assigned', 'in_progress', 'completed', 'approved', 'rejected', 'flagged']
        for i, status in enumerate(statuses):
            conv = Conversation.objects.create(
                elevenlabs_id=f'conv_status_{i}', agent=self.agent, status=status
            )
            self.assertEqual(conv.status, status)

    def test_conversation_str(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_str', agent=self.agent, status='unassigned'
        )
        self.assertIn('conv_str', str(conv))
        self.assertIn('Unassigned', str(conv))

    def test_conversation_assigned_to_set_null_on_delete(self):
        temp_user = User.objects.create_user(username='temp', password='temp')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_null', agent=self.agent, assigned_to=temp_user
        )
        temp_user.delete()
        conv.refresh_from_db()
        self.assertIsNone(conv.assigned_to)

    def test_turn_display_text(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_002', agent=self.agent
        )
        turn = Turn.objects.create(
            conversation=conv, position=0, role='user',
            original_text='hello'
        )
        self.assertEqual(turn.display_text, 'hello')
        turn.edited_text = 'Hello!'
        turn.is_edited = True
        turn.save()
        self.assertEqual(turn.display_text, 'Hello!')

    def test_turn_display_text_empty_edited(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_empty_edit', agent=self.agent
        )
        turn = Turn.objects.create(
            conversation=conv, position=0, role='user',
            original_text='hello', edited_text='', is_edited=False
        )
        self.assertEqual(turn.display_text, 'hello')

    def test_turn_unique_position_per_conversation(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_pos', agent=self.agent
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='first')
        with self.assertRaises(IntegrityError):
            Turn.objects.create(conversation=conv, position=0, role='agent', original_text='dup')

    def test_turn_ordering(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_order', agent=self.agent
        )
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='third')
        Turn.objects.create(conversation=conv, position=0, role='agent', original_text='first')
        Turn.objects.create(conversation=conv, position=1, role='user', original_text='second')
        turns = list(conv.turns.all())
        self.assertEqual([t.position for t in turns], [0, 1, 2])

    def test_turn_str(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_turnstr', agent=self.agent
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='user', original_text='hi')
        self.assertIn('Turn 0', str(turn))

    def test_tool_call_display_args(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_003', agent=self.agent
        )
        turn = Turn.objects.create(
            conversation=conv, position=0, role='agent',
            original_text='ordering...'
        )
        tc = ToolCall.objects.create(
            turn=turn, tool_name='create_order',
            original_args={'customerName': 'John'},
        )
        self.assertEqual(tc.display_args, {'customerName': 'John'})
        tc.edited_args = {'customerName': 'Jane'}
        tc.is_edited = True
        tc.save()
        self.assertEqual(tc.display_args, {'customerName': 'Jane'})

    def test_tool_call_empty_args(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_tc_empty', agent=self.agent
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='...')
        tc = ToolCall.objects.create(turn=turn, tool_name='get_specials', original_args={})
        self.assertEqual(tc.display_args, {})

    def test_tool_call_nested_json_args(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_tc_nested', agent=self.agent
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='...')
        nested_args = {
            'customerName': 'John',
            'items': [{'itemName': 'Pizza', 'quantity': 1, 'modifiers': ['cheese', 'pepperoni']}]
        }
        tc = ToolCall.objects.create(turn=turn, tool_name='create_order', original_args=nested_args)
        self.assertEqual(tc.display_args['items'][0]['itemName'], 'Pizza')

    def test_tool_call_str(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_tcstr', agent=self.agent
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='...')
        tc = ToolCall.objects.create(turn=turn, tool_name='create_order', original_args={})
        self.assertIn('create_order', str(tc))

    def test_system_prompt_only_one_active(self):
        p1 = SystemPrompt.objects.create(name='v1', content='prompt 1', is_active=True)
        p2 = SystemPrompt.objects.create(name='v2', content='prompt 2', is_active=True)
        p1.refresh_from_db()
        self.assertFalse(p1.is_active)
        self.assertTrue(p2.is_active)

    def test_system_prompt_three_active_cascading(self):
        p1 = SystemPrompt.objects.create(name='v1', content='p1', is_active=True)
        p2 = SystemPrompt.objects.create(name='v2', content='p2', is_active=True)
        p3 = SystemPrompt.objects.create(name='v3', content='p3', is_active=True)
        p1.refresh_from_db()
        p2.refresh_from_db()
        self.assertFalse(p1.is_active)
        self.assertFalse(p2.is_active)
        self.assertTrue(p3.is_active)

    def test_system_prompt_inactive_doesnt_deactivate_others(self):
        p1 = SystemPrompt.objects.create(name='v1', content='p1', is_active=True)
        p2 = SystemPrompt.objects.create(name='v2', content='p2', is_active=False)
        p1.refresh_from_db()
        self.assertTrue(p1.is_active)
        self.assertFalse(p2.is_active)

    def test_system_prompt_str(self):
        p = SystemPrompt.objects.create(name='test', content='c', is_active=True)
        self.assertIn('active', str(p))

    def test_agent_unique_agent_id(self):
        with self.assertRaises(IntegrityError):
            Agent.objects.create(
                agent_id='agent_test', label='Dup Agent', elevenlabs_api_key='key2'
            )

    def test_agent_str(self):
        self.assertEqual(str(self.agent), 'Test Agent')

    def test_export_log_creation(self):
        log = ExportLog.objects.create(
            exported_by=self.admin, conversation_count=10, token_count=5000
        )
        self.assertIn('10 convs', str(log))

    def test_conversation_cascade_delete(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_cascade', agent=self.agent
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='hi')
        ToolCall.objects.create(turn=turn, tool_name='get_specials', original_args={})
        conv.delete()
        self.assertEqual(Turn.objects.filter(conversation__elevenlabs_id='conv_cascade').count(), 0)
        self.assertEqual(ToolCall.objects.filter(turn=turn).count(), 0)


# =============================================================================
# EXPORT PIPELINE TESTS
# =============================================================================

class ExportTests(TestCase):
    def setUp(self):
        self.agent = Agent.objects.create(
            agent_id='agent_export', label='Export Agent', elevenlabs_api_key='key'
        )
        self.prompt = SystemPrompt.objects.create(
            name='Test Prompt', content='You are a test assistant.', is_active=True
        )
        self.conv = Conversation.objects.create(
            elevenlabs_id='conv_export_001', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        t0 = Turn.objects.create(
            conversation=self.conv, position=0, role='agent',
            original_text='Welcome! How can I help?'
        )
        t1 = Turn.objects.create(
            conversation=self.conv, position=1, role='user',
            original_text='I want a pizza.'
        )
        t2 = Turn.objects.create(
            conversation=self.conv, position=2, role='agent',
            original_text='Let me place that order.'
        )
        ToolCall.objects.create(
            turn=t2, tool_name='create_order',
            original_args={'customerName': 'Test', 'customerPhone': '555', 'items': []},
            status_code=200,
            response_body={'success': True, 'orderId': 'ORD-1'},
        )
        Turn.objects.create(
            conversation=self.conv, position=3, role='agent',
            original_text='Order placed!'
        )

    def test_conversation_to_messages(self):
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        self.assertEqual(msgs[0]['role'], 'system')
        self.assertEqual(msgs[0]['content'], 'You are a test assistant.')
        roles = [m['role'] for m in msgs]
        self.assertIn('user', roles)
        self.assertIn('assistant', roles)

    def test_tool_call_format(self):
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        tc_msgs = [m for m in msgs if 'tool_calls' in m]
        self.assertEqual(len(tc_msgs), 1)
        tc = tc_msgs[0]['tool_calls'][0]
        self.assertEqual(tc['function']['name'], 'create_order')
        args = json.loads(tc['function']['arguments'])
        self.assertEqual(args['customerName'], 'Test')
        tool_msgs = [m for m in msgs if m['role'] == 'tool']
        self.assertEqual(len(tool_msgs), 1)

    def test_tool_call_id_format(self):
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        tc_msg = [m for m in msgs if 'tool_calls' in m][0]
        tc_id = tc_msg['tool_calls'][0]['id']
        self.assertTrue(tc_id.startswith('call_'))
        tool_response = [m for m in msgs if m['role'] == 'tool'][0]
        self.assertEqual(tool_response['tool_call_id'], tc_id)

    def test_validate_valid_example(self):
        result = conversation_to_messages(self.conv)
        errors = validate_example(result)
        self.assertEqual(errors, [])

    def test_validate_invalid_example(self):
        errors = validate_example({'messages': []})
        self.assertIn('No messages', errors)

    def test_validate_missing_user_message(self):
        example = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'assistant', 'content': 'hello'},
        ]}
        errors = validate_example(example)
        self.assertIn('Missing user message', errors)

    def test_validate_missing_assistant_message(self):
        example = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'hello'},
        ]}
        errors = validate_example(example)
        self.assertIn('Missing assistant message', errors)

    def test_validate_empty_system_content(self):
        example = {'messages': [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]}
        errors = validate_example(example)
        self.assertIn('Empty content in system message', errors)

    def test_validate_empty_user_content(self):
        example = {'messages': [
            {'role': 'user', 'content': '  '},
            {'role': 'assistant', 'content': 'hello'},
        ]}
        errors = validate_example(example)
        self.assertIn('Empty content in user message', errors)

    def test_generate_jsonl(self):
        examples = generate_jsonl_examples()
        self.assertEqual(len(examples), 1)

    def test_generate_jsonl_excludes_non_approved(self):
        Conversation.objects.create(
            elevenlabs_id='conv_not_approved', agent=self.agent, status='in_progress',
            call_timestamp=timezone.now()
        )
        examples = generate_jsonl_examples()
        self.assertEqual(len(examples), 1)

    def test_generate_jsonl_agent_filter(self):
        other_agent = Agent.objects.create(
            agent_id='other_agent', label='Other', elevenlabs_api_key='k'
        )
        conv2 = Conversation.objects.create(
            elevenlabs_id='conv_other', agent=other_agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv2, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv2, position=1, role='agent', original_text='Hello')

        all_examples = generate_jsonl_examples()
        self.assertEqual(len(all_examples), 2)
        filtered = generate_jsonl_examples(agent_id=self.agent.pk)
        self.assertEqual(len(filtered), 1)

    def test_generate_jsonl_tool_calls_only(self):
        conv_no_tc = Conversation.objects.create(
            elevenlabs_id='conv_no_tc', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv_no_tc, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv_no_tc, position=1, role='agent', original_text='Hello')

        all_examples = generate_jsonl_examples()
        self.assertEqual(len(all_examples), 2)
        tc_only = generate_jsonl_examples(tool_calls_only=True)
        self.assertEqual(len(tc_only), 1)

    def test_export_jsonl_format(self):
        examples = generate_jsonl_examples()
        jsonl = export_jsonl(examples)
        lines = jsonl.strip().split('\n')
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertIn('messages', parsed)
        self.assertIn('tools', parsed)

    def test_export_jsonl_each_line_valid_json(self):
        for i in range(3):
            conv = Conversation.objects.create(
                elevenlabs_id=f'conv_jsonl_{i}', agent=self.agent, status='approved',
                call_timestamp=timezone.now()
            )
            Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
            Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello')

        examples = generate_jsonl_examples()
        jsonl = export_jsonl(examples)
        for line in jsonl.strip().split('\n'):
            parsed = json.loads(line)
            self.assertIn('messages', parsed)

    def test_split_train_validation(self):
        for i in range(10):
            conv = Conversation.objects.create(
                elevenlabs_id=f'conv_split_{i}', agent=self.agent, status='approved',
                call_timestamp=timezone.now()
            )
            Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
            Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello')

        examples = generate_jsonl_examples()
        train, val = split_train_validation(examples)
        self.assertEqual(len(train) + len(val), len(examples))

    def test_split_zero_examples(self):
        train, val = split_train_validation([])
        self.assertEqual(len(train), 0)
        self.assertEqual(len(val), 0)

    def test_split_one_example(self):
        examples = [{'messages': [{'role': 'user', 'content': 'hi'}]}]
        train, val = split_train_validation(examples)
        self.assertEqual(len(train) + len(val), 1)

    def test_split_two_examples(self):
        examples = [
            {'messages': [{'role': 'user', 'content': 'a'}]},
            {'messages': [{'role': 'user', 'content': 'b'}]},
        ]
        train, val = split_train_validation(examples)
        self.assertEqual(len(train) + len(val), 2)

    def test_count_tokens(self):
        examples = generate_jsonl_examples()
        token_count = count_tokens(examples)
        self.assertGreater(token_count, 0)

    def test_count_tokens_consistent(self):
        examples = generate_jsonl_examples()
        count1 = count_tokens(examples)
        count2 = count_tokens(examples)
        self.assertEqual(count1, count2)

    def test_estimate_training_cost(self):
        cost = estimate_training_cost(1_000_000, epochs=3)
        self.assertEqual(cost, 75.0)

    def test_estimate_training_cost_small(self):
        cost = estimate_training_cost(1000, epochs=1)
        expected = round((1000 / 1_000_000) * 25.0, 2)
        self.assertEqual(cost, expected)

    def test_tools_included(self):
        result = conversation_to_messages(self.conv, include_tools=True)
        self.assertIn('tools', result)
        tool_names = [t['function']['name'] for t in result['tools']]
        self.assertIn('create_order', tool_names)

    def test_no_tools_when_disabled(self):
        result = conversation_to_messages(self.conv, include_tools=False)
        self.assertNotIn('tools', result)

    def test_tool_definitions_count(self):
        self.assertEqual(len(TOOL_DEFINITIONS), 10)

    def test_tool_definitions_structure(self):
        expected_tools = [
            'create_order', 'cancel_order', 'remove_item', 'modify_item',
            'check_availability', 'create_reservation', 'get_specials', 'get_past_orders',
            'end_call', 'send_menu_link'
        ]
        actual_tools = [t['function']['name'] for t in TOOL_DEFINITIONS]
        for tool in expected_tools:
            self.assertIn(tool, actual_tools)

    def test_tool_definitions_have_parameters(self):
        for tool_def in TOOL_DEFINITIONS:
            self.assertEqual(tool_def['type'], 'function')
            self.assertIn('name', tool_def['function'])
            self.assertIn('description', tool_def['function'])
            self.assertIn('parameters', tool_def['function'])
            self.assertEqual(tool_def['function']['parameters']['type'], 'object')

    def test_no_system_prompt_when_disabled(self):
        result = conversation_to_messages(self.conv, include_system_prompt=False)
        msgs = result['messages']
        self.assertNotEqual(msgs[0]['role'], 'system')

    def test_no_system_prompt_when_none_active(self):
        self.prompt.is_active = False
        self.prompt.save()
        result = conversation_to_messages(self.conv, include_system_prompt=True)
        msgs = result['messages']
        self.assertNotEqual(msgs[0].get('role'), 'system')

    def test_empty_conversation(self):
        empty_conv = Conversation.objects.create(
            elevenlabs_id='conv_empty', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        result = conversation_to_messages(empty_conv)
        msgs = result['messages']
        # Should only have system prompt
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]['role'], 'system')

    def test_user_only_conversation(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_user_only', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hello')
        result = conversation_to_messages(conv)
        msgs = result['messages']
        roles = [m['role'] for m in msgs]
        self.assertIn('user', roles)
        self.assertNotIn('assistant', roles)

    def test_agent_only_conversation(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_agent_only', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='agent', original_text='Welcome!')
        result = conversation_to_messages(conv)
        msgs = result['messages']
        roles = [m['role'] for m in msgs]
        self.assertIn('assistant', roles)

    def test_edited_text_used_in_export(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_edited', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(
            conversation=conv, position=0, role='user',
            original_text='I wanna pizza', edited_text='I want a pizza.', is_edited=True
        )
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='OK!')
        result = conversation_to_messages(conv)
        user_msgs = [m for m in result['messages'] if m.get('role') == 'user']
        self.assertEqual(user_msgs[0]['content'], 'I want a pizza.')

    def test_edited_tool_call_args_used(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_edited_tc', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='Placing...')
        ToolCall.objects.create(
            turn=turn, tool_name='create_order',
            original_args={'customerName': 'Jon'},
            edited_args={'customerName': 'John'},
            is_edited=True,
            status_code=200, response_body={'success': True},
        )
        Turn.objects.create(conversation=conv, position=1, role='user', original_text='OK')
        result = conversation_to_messages(conv)
        tc_msg = [m for m in result['messages'] if 'tool_calls' in m][0]
        args = json.loads(tc_msg['tool_calls'][0]['function']['arguments'])
        self.assertEqual(args['customerName'], 'John')

    def test_multiple_tool_calls_on_same_turn(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_multi_tc', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Help me')
        turn = Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Checking...')
        ToolCall.objects.create(
            turn=turn, tool_name='check_availability',
            original_args={'date': '2026-02-15', 'time': '19:00', 'partySize': 4},
            status_code=200, response_body={'available': True},
        )
        ToolCall.objects.create(
            turn=turn, tool_name='create_reservation',
            original_args={'customerName': 'Test', 'customerPhone': '555',
                           'partySize': 4, 'date': '2026-02-15', 'time': '19:00'},
            status_code=200, response_body={'success': True, 'reservationId': 'RES-1'},
        )
        result = conversation_to_messages(conv)
        tc_msgs = [m for m in result['messages'] if 'tool_calls' in m]
        self.assertEqual(len(tc_msgs), 1)
        self.assertEqual(len(tc_msgs[0]['tool_calls']), 2)
        tool_responses = [m for m in result['messages'] if m['role'] == 'tool']
        self.assertEqual(len(tool_responses), 2)
        # IDs should be unique
        ids = {r['tool_call_id'] for r in tool_responses}
        self.assertEqual(len(ids), 2)

    def test_tool_call_null_response_body(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_null_resp', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        turn = Turn.objects.create(conversation=conv, position=1, role='agent', original_text='...')
        ToolCall.objects.create(
            turn=turn, tool_name='get_specials', original_args={},
            status_code=200, response_body={},
        )
        result = conversation_to_messages(conv)
        tool_msgs = [m for m in result['messages'] if m['role'] == 'tool']
        self.assertEqual(len(tool_msgs), 1)
        # Empty dict becomes {"status": "ok"}
        self.assertEqual(tool_msgs[0]['content'], '{"status": "ok"}')

    def test_tool_call_error_status_code(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_error_tc', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='order')
        turn = Turn.objects.create(conversation=conv, position=1, role='agent', original_text='placing...')
        ToolCall.objects.create(
            turn=turn, tool_name='create_order',
            original_args={'customerName': 'Test', 'customerPhone': '555', 'items': []},
            status_code=500,
            response_body={'error': 'Internal server error'},
        )
        result = conversation_to_messages(conv)
        tool_msgs = [m for m in result['messages'] if m['role'] == 'tool']
        self.assertEqual(len(tool_msgs), 1)
        content = json.loads(tool_msgs[0]['content'])
        self.assertIn('error', content)

    def test_tools_only_include_used(self):
        result = conversation_to_messages(self.conv, include_tools=True)
        tool_names = [t['function']['name'] for t in result['tools']]
        self.assertEqual(tool_names, ['create_order'])

    def test_no_tools_when_no_tool_calls_in_conversation(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_no_tc_tools', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello')
        result = conversation_to_messages(conv, include_tools=True)
        self.assertNotIn('tools', result)

    def test_parallel_tool_calls_false(self):
        result = conversation_to_messages(self.conv)
        self.assertFalse(result['parallel_tool_calls'])

    def test_empty_turn_text_skipped(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_empty_turn', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='')
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='Hello!')
        result = conversation_to_messages(conv)
        assistant_msgs = [m for m in result['messages'] if m.get('role') == 'assistant']
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]['content'], 'Hello!')

    # ---- Fix 1: Empty response body ----

    def test_empty_response_body_gets_status_ok(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_empty_resp', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Any specials?')
        turn = Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Let me check.')
        ToolCall.objects.create(
            turn=turn, tool_name='get_specials', original_args={},
            status_code=200, response_body={},
        )
        result = conversation_to_messages(conv)
        tool_msgs = [m for m in result['messages'] if m['role'] == 'tool']
        content = json.loads(tool_msgs[0]['content'])
        self.assertEqual(content, {"status": "ok"})

    # ---- Fix 2: Validator ordering checks ----

    def test_validate_first_message_must_be_system_or_user(self):
        example = {'messages': [
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'hi'},
        ]}
        errors = validate_example(example)
        self.assertTrue(any('First message must be system or user' in e for e in errors))

    def test_validate_orphaned_tool_response(self):
        example = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'tool', 'tool_call_id': 'call_001', 'content': '{}'},
            {'role': 'assistant', 'content': 'done'},
        ]}
        errors = validate_example(example)
        self.assertTrue(any('Orphaned tool response' in e for e in errors))

    def test_validate_unmatched_tool_call_ids(self):
        example = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_001', 'type': 'function', 'function': {'name': 'get_specials', 'arguments': '{}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_999', 'content': '{}'},
        ]}
        errors = validate_example(example)
        self.assertTrue(any("call_999" in e and "not in preceding" in e for e in errors))

    def test_validate_missing_tool_response(self):
        example = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_001', 'type': 'function', 'function': {'name': 'get_specials', 'arguments': '{}'}},
            ]},
            {'role': 'assistant', 'content': 'Here you go!'},
        ]}
        errors = validate_example(example)
        self.assertTrue(any('Unmatched tool_call_ids' in e for e in errors))

    def test_validate_tool_ordering_valid(self):
        example = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_001', 'type': 'function', 'function': {'name': 'get_specials', 'arguments': '{}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_001', 'content': '{"status": "ok"}'},
            {'role': 'assistant', 'content': 'Here are the specials!'},
        ]}
        errors = validate_example(example)
        self.assertEqual(errors, [])

    # ---- Fix 3: Dataset minimum warning ----

    def test_dataset_warning_below_10(self):
        examples = [{'messages': [{'role': 'user', 'content': f'msg {i}'}]} for i in range(5)]
        warnings = validate_dataset(examples)
        self.assertEqual(len(warnings), 1)
        self.assertIn('at least 10', warnings[0])
        self.assertIn('5', warnings[0])

    def test_dataset_warning_at_10(self):
        examples = [{'messages': [{'role': 'user', 'content': f'msg {i}'}]} for i in range(10)]
        warnings = validate_dataset(examples)
        self.assertEqual(warnings, [])

    # ---- Fix 4: Weight on greeting ----

    def test_weight_zero_on_greeting(self):
        """First assistant message before any user message gets weight: 0."""
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        # First assistant message is the greeting (position 0, before user at position 1)
        assistant_msgs = [m for m in msgs if m.get('role') == 'assistant']
        self.assertEqual(assistant_msgs[0].get('weight'), 0)

    def test_weight_not_set_after_user(self):
        """Assistant messages after the first user message have no weight key (default 1)."""
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        # The last assistant message "Order placed!" comes after user input
        last_assistant = [m for m in msgs if m.get('role') == 'assistant' and 'tool_calls' not in m]
        # Last one should be "Order placed!" which is after user
        self.assertNotIn('weight', last_assistant[-1])

    def test_weight_on_tool_call_before_user(self):
        """Tool-call assistant message before first user gets weight: 0."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_weight_tc', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='Checking menu...')
        ToolCall.objects.create(
            turn=turn, tool_name='get_specials', original_args={},
            status_code=200, response_body={'specials': []},
        )
        Turn.objects.create(conversation=conv, position=1, role='user', original_text='What do you have?')
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='Here are our specials!')
        result = conversation_to_messages(conv)
        msgs = result['messages']
        # The tool_calls assistant msg (before user) should have weight: 0
        tc_msg = [m for m in msgs if 'tool_calls' in m][0]
        self.assertEqual(tc_msg.get('weight'), 0)
        # The final assistant msg (after user) should NOT have weight
        final_assistant = [m for m in msgs if m.get('role') == 'assistant' and 'tool_calls' not in m][-1]
        self.assertNotIn('weight', final_assistant)

    # ---- Fix 5: None response_body ----

    def test_none_response_body_gets_status_ok(self):
        """None response_body becomes {"status": "ok"} in export."""
        # Use mocks to simulate None response_body (DB has NOT NULL constraint)
        mock_tc = MagicMock()
        mock_tc.tool_name = 'get_specials'
        mock_tc.display_args = {}
        mock_tc.response_body = None

        mock_turn_user = MagicMock()
        mock_turn_user.role = 'user'
        mock_turn_user.display_text = 'Specials?'
        mock_turn_user.is_deleted = False
        mock_turn_user.weight = None
        mock_turn_user.tool_calls.all.return_value = []

        mock_tc.is_deleted = False

        mock_turn_tc = MagicMock()
        mock_turn_tc.role = 'agent'
        mock_turn_tc.display_text = 'Checking...'
        mock_turn_tc.is_deleted = False
        mock_turn_tc.weight = None
        mock_turn_tc.tool_calls.all.return_value = [mock_tc]

        mock_turn_final = MagicMock()
        mock_turn_final.role = 'agent'
        mock_turn_final.display_text = 'Here!'
        mock_turn_final.is_deleted = False
        mock_turn_final.weight = None
        mock_turn_final.tool_calls.all.return_value = []

        mock_conv = MagicMock()
        mock_conv.turns.prefetch_related.return_value.all.return_value = [
            mock_turn_user, mock_turn_tc, mock_turn_final,
        ]

        result = conversation_to_messages(mock_conv, include_system_prompt=False, include_tools=False)
        tool_msgs = [m for m in result['messages'] if m['role'] == 'tool']
        self.assertEqual(tool_msgs[0]['content'], '{"status": "ok"}')

    # ---- Fix 6: Last message must be assistant ----

    def test_validate_last_message_must_be_assistant(self):
        """Example ending on user or tool message should fail validation."""
        # Ends on user
        example_user = {'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'bye'},
        ]}
        errors = validate_example(example_user)
        self.assertTrue(any('Last message must be assistant' in e for e in errors))

        # Ends on tool
        example_tool = {'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_001', 'type': 'function', 'function': {'name': 'get_specials', 'arguments': '{}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_001', 'content': '{"status": "ok"}'},
        ]}
        errors = validate_example(example_tool)
        self.assertTrue(any('Last message must be assistant' in e for e in errors))

    def test_validate_last_message_assistant_passes(self):
        """Example ending on assistant message passes the last-message check."""
        example = {'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]}
        errors = validate_example(example)
        self.assertFalse(any('Last message must be assistant' in e for e in errors))

    # ---- Fix 7: Token limit check ----

    def test_validate_example_token_limit(self):
        """Huge example triggers token limit error."""
        # Create a message with content larger than MAX_EXAMPLE_TOKENS * 3 chars
        huge_content = "x" * (MAX_EXAMPLE_TOKENS * 4)
        example = {'messages': [
            {'role': 'user', 'content': huge_content},
            {'role': 'assistant', 'content': 'ok'},
        ]}
        errors = validate_example(example)
        self.assertTrue(any('exceeds token limit' in e for e in errors))

    # ---- Fix 8: List response_body ----

    def test_list_response_body_serialized(self):
        """List response_body is properly JSON-serialized (not Python repr)."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_list_resp', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Menu?')
        turn = Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Checking...')
        ToolCall.objects.create(
            turn=turn, tool_name='get_specials', original_args={},
            status_code=200, response_body=[{"item": "Pizza"}, {"item": "Pasta"}],
        )
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='Here!')
        result = conversation_to_messages(conv)
        tool_msgs = [m for m in result['messages'] if m['role'] == 'tool']
        content = tool_msgs[0]['content']
        # Must be valid JSON
        parsed = json.loads(content)
        self.assertEqual(parsed, [{"item": "Pizza"}, {"item": "Pasta"}])
        # Must NOT contain single quotes (Python repr)
        self.assertNotIn("'", content)

    # ---- Fix 9: Empty tool content ----

    def test_validate_empty_tool_content(self):
        """Empty tool response content fails validation."""
        example = {'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'call_001', 'type': 'function', 'function': {'name': 'get_specials', 'arguments': '{}'}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call_001', 'content': ''},
            {'role': 'assistant', 'content': 'done'},
        ]}
        errors = validate_example(example)
        self.assertTrue(any('Empty content in tool response' in e for e in errors))


# =============================================================================
# SYNC SERVICE TESTS
# =============================================================================

class SyncServiceTests(TestCase):
    def setUp(self):
        self.agent = Agent.objects.create(
            agent_id='agent_sync', label='Sync Agent', elevenlabs_api_key='test-key'
        )

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_empty_list(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {'conversations': []}
        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['imported'], 0)
        self.assertEqual(stats['skipped'], 0)

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_imports_new_conversation(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {
            'conversations': [{'conversation_id': 'conv_new_001'}],
        }
        mock_instance.get_conversation.return_value = {
            'transcript': [
                {'role': 'agent', 'message': 'Hello!', 'time_in_call_secs': 0.5},
                {'role': 'user', 'message': 'Hi!', 'time_in_call_secs': 2.0},
            ],
            'metadata': {'start_time_unix_secs': 1714423232, 'call_duration_secs': 45},
            'has_audio': False,
        }
        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['imported'], 1)
        self.assertTrue(Conversation.objects.filter(elevenlabs_id='conv_new_001').exists())
        conv = Conversation.objects.get(elevenlabs_id='conv_new_001')
        self.assertEqual(conv.turns.count(), 2)

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_deduplication(self, MockClient):
        Conversation.objects.create(elevenlabs_id='conv_existing', agent=self.agent)
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {
            'conversations': [{'conversation_id': 'conv_existing'}],
        }
        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['imported'], 0)
        self.assertEqual(stats['skipped'], 1)

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_pagination(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.side_effect = [
            {
                'conversations': [{'conversation_id': 'page1_conv'}],
                'cursor': 'next_page_cursor',
            },
            {
                'conversations': [{'conversation_id': 'page2_conv'}],
            },
        ]
        mock_instance.get_conversation.return_value = {
            'transcript': [{'role': 'agent', 'message': 'Hello'}],
            'metadata': {},
            'has_audio': False,
        }
        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['imported'], 2)

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_handles_api_error(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.side_effect = Exception("API error")
        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['errors'], 1)

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_handles_individual_conv_error(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {
            'conversations': [
                {'conversation_id': 'conv_good'},
                {'conversation_id': 'conv_bad'},
            ],
        }
        mock_instance.get_conversation.side_effect = [
            {'transcript': [{'role': 'agent', 'message': 'Hi'}], 'metadata': {}, 'has_audio': False},
            Exception("Failed to fetch"),
        ]
        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['imported'], 1)
        self.assertEqual(stats['errors'], 1)

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_sync_updates_last_synced_at(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {'conversations': []}
        self.assertIsNone(self.agent.last_synced_at)
        sync_agent_conversations(self.agent)
        self.agent.refresh_from_db()
        self.assertIsNotNone(self.agent.last_synced_at)

    def test_import_conversation_with_tool_calls(self):
        data = {
            'transcript': [
                {
                    'role': 'agent',
                    'message': 'Ordering...',
                    'tool_calls': [{
                        'tool_name': 'create_order',
                        'params': {'customerName': 'John', 'items': []},
                        'status_code': 200,
                        'response_body': '{"success": true}',
                    }],
                },
            ],
            'metadata': {'start_time_unix_secs': 1714423232, 'call_duration_secs': 30},
            'has_audio': True,
        }
        mock_client = MagicMock()
        conv = _import_conversation(self.agent, 'conv_with_tc', data, mock_client)
        self.assertEqual(conv.turns.count(), 1)
        turn = conv.turns.first()
        self.assertEqual(turn.tool_calls.count(), 1)
        tc = turn.tool_calls.first()
        self.assertEqual(tc.tool_name, 'create_order')
        self.assertEqual(tc.original_args['customerName'], 'John')

    def test_import_conversation_tool_call_from_request_body(self):
        data = {
            'transcript': [
                {
                    'role': 'agent',
                    'message': 'Checking...',
                    'tool_calls': [{
                        'tool_name': 'check_availability',
                        'request_headers_body': '{"date":"2026-02-15","time":"19:00","partySize":4}',
                        'status_code': 200,
                        'response_body': '{"available": true}',
                    }],
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        mock_client = MagicMock()
        conv = _import_conversation(self.agent, 'conv_req_body', data, mock_client)
        tc = conv.turns.first().tool_calls.first()
        self.assertEqual(tc.original_args['date'], '2026-02-15')

    def test_import_conversation_no_transcript(self):
        data = {'transcript': [], 'metadata': {}, 'has_audio': False}
        mock_client = MagicMock()
        conv = _import_conversation(self.agent, 'conv_no_transcript', data, mock_client)
        self.assertEqual(conv.turns.count(), 0)

    def test_import_conversation_invalid_role(self):
        data = {
            'transcript': [{'role': 'system', 'message': 'internal'}],
            'metadata': {},
            'has_audio': False,
        }
        mock_client = MagicMock()
        conv = _import_conversation(self.agent, 'conv_bad_role', data, mock_client)
        turn = conv.turns.first()
        self.assertEqual(turn.role, 'user')

    def test_import_conversation_malformed_response_body(self):
        data = {
            'transcript': [
                {
                    'role': 'agent',
                    'message': '...',
                    'tool_calls': [{
                        'tool_name': 'get_specials',
                        'params': {},
                        'status_code': 200,
                        'response_body': 'not json',
                    }],
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        mock_client = MagicMock()
        conv = _import_conversation(self.agent, 'conv_bad_resp', data, mock_client)
        tc = conv.turns.first().tool_calls.first()
        self.assertEqual(tc.response_body, {'raw': 'not json'})


# =============================================================================
# VIEW TESTS
# =============================================================================

class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='admin', password='admin', role='admin'
        )
        self.annotator = User.objects.create_user(
            username='annotator', password='annotator', role='annotator'
        )
        self.agent = Agent.objects.create(
            agent_id='agent_view', label='View Agent', elevenlabs_api_key='key'
        )

    # ---- Authentication ----

    def test_login_page(self):
        response = self.client.get('/login/')
        self.assertEqual(response.status_code, 200)

    def test_login_success(self):
        response = self.client.post('/login/', {'username': 'admin', 'password': 'admin'})
        self.assertEqual(response.status_code, 302)

    def test_login_failure(self):
        response = self.client.post('/login/', {'username': 'admin', 'password': 'wrong'})
        self.assertEqual(response.status_code, 200)

    def test_logout(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/logout/')
        self.assertEqual(response.status_code, 302)

    def test_unauthenticated_conversations(self):
        response = self.client.get('/conversations/')
        self.assertEqual(response.status_code, 302)  # redirect to login

    def test_unauthenticated_admin_panel(self):
        response = self.client.get('/admin-panel/')
        self.assertEqual(response.status_code, 403)

    # ---- Dashboard ----

    def test_dashboard_redirect_admin(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/dashboard/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('admin-panel', response.url)

    def test_dashboard_redirect_annotator(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/dashboard/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('conversations', response.url)

    # ---- Annotator Views ----

    def test_annotator_list(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/')
        self.assertEqual(response.status_code, 200)

    def test_annotator_list_with_status_filter(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/?status=in_progress')
        self.assertEqual(response.status_code, 200)

    def test_annotator_list_all_filter(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/?status=all')
        self.assertEqual(response.status_code, 200)

    def test_conversation_editor(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_view_001', agent=self.agent,
            assigned_to=self.annotator, status='assigned'
        )
        Turn.objects.create(
            conversation=conv, position=0, role='agent',
            original_text='Hello!'
        )
        response = self.client.get(f'/conversations/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'in_progress')

    def test_conversation_editor_no_auto_transition_for_admin(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_admin_view', agent=self.agent,
            assigned_to=self.annotator, status='assigned'
        )
        Turn.objects.create(conversation=conv, position=0, role='agent', original_text='Hi')
        response = self.client.get(f'/conversations/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        conv.refresh_from_db()
        # Admin viewing doesn't auto-transition since they're not the assignee
        self.assertEqual(conv.status, 'assigned')

    def test_conversation_editor_permission_denied(self):
        self.client.login(username='annotator', password='annotator')
        other_annotator = User.objects.create_user(
            username='other', password='other', role='annotator'
        )
        conv = Conversation.objects.create(
            elevenlabs_id='conv_other_user', agent=self.agent,
            assigned_to=other_annotator, status='assigned'
        )
        response = self.client.get(f'/conversations/{conv.pk}/')
        self.assertEqual(response.status_code, 403)

    def test_conversation_editor_admin_can_view_any(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_admin_access', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        Turn.objects.create(conversation=conv, position=0, role='agent', original_text='Hi')
        response = self.client.get(f'/conversations/{conv.pk}/')
        self.assertEqual(response.status_code, 200)

    # ---- Turn Editing ----

    def test_turn_edit_htmx(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_edit_001', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        turn = Turn.objects.create(
            conversation=conv, position=0, role='user',
            original_text='I wanna pizza'
        )
        response = self.client.get(f'/conversations/{conv.pk}/turn/{turn.pk}/edit/')
        self.assertEqual(response.status_code, 200)
        response = self.client.post(
            f'/conversations/{conv.pk}/turn/{turn.pk}/edit/',
            {'edited_text': 'I want a pizza'}
        )
        self.assertEqual(response.status_code, 200)
        turn.refresh_from_db()
        self.assertTrue(turn.is_edited)
        self.assertEqual(turn.edited_text, 'I want a pizza')

    def test_turn_edit_reset_when_same_as_original(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_edit_reset', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        turn = Turn.objects.create(
            conversation=conv, position=0, role='user',
            original_text='hello', edited_text='something', is_edited=True
        )
        # Posting the same text as original resets the edit
        self.client.post(
            f'/conversations/{conv.pk}/turn/{turn.pk}/edit/',
            {'edited_text': 'hello'}
        )
        turn.refresh_from_db()
        self.assertFalse(turn.is_edited)
        self.assertEqual(turn.edited_text, '')

    def test_turn_display(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_display', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        turn = Turn.objects.create(
            conversation=conv, position=0, role='user', original_text='hello'
        )
        response = self.client.get(f'/conversations/{conv.pk}/turn/{turn.pk}/display/')
        self.assertEqual(response.status_code, 200)

    # ---- Tool Call Editing ----

    def test_tool_call_edit_get(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_tc_edit', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='...')
        tc = ToolCall.objects.create(
            turn=turn, tool_name='create_order',
            original_args={'customerName': 'John', 'customerPhone': '555'},
        )
        response = self.client.get(f'/conversations/{conv.pk}/tool/{tc.pk}/edit/')
        self.assertEqual(response.status_code, 200)

    def test_tool_call_edit_post(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_tc_edit_post', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='...')
        tc = ToolCall.objects.create(
            turn=turn, tool_name='cancel_order',
            original_args={'orderId': 'ORD-1', 'reason': 'changed mind'},
        )
        response = self.client.post(f'/conversations/{conv.pk}/tool/{tc.pk}/edit/', {
            'arg_orderId': 'ORD-2',
            'arg_reason': 'wrong order',
        })
        self.assertEqual(response.status_code, 200)
        tc.refresh_from_db()
        self.assertTrue(tc.is_edited)
        self.assertEqual(tc.edited_args['orderId'], 'ORD-2')

    def test_tool_call_display(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_tc_display', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        turn = Turn.objects.create(conversation=conv, position=0, role='agent', original_text='...')
        tc = ToolCall.objects.create(turn=turn, tool_name='get_specials', original_args={})
        response = self.client.get(f'/conversations/{conv.pk}/tool/{tc.pk}/display/')
        self.assertEqual(response.status_code, 200)

    # ---- Conversation Status Changes ----

    def test_complete_conversation(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_complete_001', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        response = self.client.post(f'/conversations/{conv.pk}/complete/')
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'completed')
        self.assertIsNotNone(conv.completed_at)

    def test_complete_conversation_admin(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_complete_admin', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        response = self.client.post(f'/conversations/{conv.pk}/complete/')
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'completed')

    def test_flag_conversation(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_flag_001', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        response = self.client.post(f'/conversations/{conv.pk}/flag/', {'flag_notes': 'bad audio'})
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'flagged')
        self.assertIn('[FLAGGED] bad audio', conv.annotator_notes)

    def test_flag_conversation_without_notes(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_flag_no_notes', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        response = self.client.post(f'/conversations/{conv.pk}/flag/')
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'flagged')

    def test_conversation_notes(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_notes', agent=self.agent,
            assigned_to=self.annotator, status='in_progress'
        )
        response = self.client.post(f'/conversations/{conv.pk}/notes/', {
            'annotator_notes': 'These are my notes.'
        })
        self.assertEqual(response.status_code, 200)
        conv.refresh_from_db()
        self.assertEqual(conv.annotator_notes, 'These are my notes.')

    # ---- Admin Views ----

    def test_admin_dashboard(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/')
        self.assertEqual(response.status_code, 200)

    def test_admin_agents(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/agents/')
        self.assertEqual(response.status_code, 200)

    def test_admin_required_blocks_annotator(self):
        self.client.login(username='annotator', password='annotator')
        admin_routes = [
            '/admin-panel/',
            '/admin-panel/agents/',
            '/admin-panel/assign/',
            '/admin-panel/review/',
            '/admin-panel/export/',
            '/admin-panel/analytics/',
            '/admin-panel/team/',
            '/admin-panel/prompts/',
        ]
        for url in admin_routes:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 403, f"Route {url} should block annotator")

    def test_admin_required_blocks_unauthenticated(self):
        response = self.client.get('/admin-panel/')
        self.assertEqual(response.status_code, 403)

    # ---- Approve/Reject ----

    def test_approve_conversation(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_approve_001', agent=self.agent,
            assigned_to=self.annotator, status='completed'
        )
        response = self.client.post(f'/admin-panel/review/{conv.pk}/approve/')
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'approved')
        self.assertIsNotNone(conv.reviewed_at)

    def test_reject_conversation(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_reject_001', agent=self.agent,
            assigned_to=self.annotator, status='completed'
        )
        response = self.client.post(
            f'/admin-panel/review/{conv.pk}/reject/',
            {'reviewer_notes': 'Need more edits'}
        )
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'assigned')
        self.assertEqual(conv.reviewer_notes, 'Need more edits')
        self.assertIsNone(conv.completed_at)

    # ---- Assignment ----

    def test_assign_conversations(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_assign_001', agent=self.agent, status='unassigned'
        )
        response = self.client.post('/admin-panel/assign/', {
            'conversation_ids': [conv.pk],
            'assignee': self.annotator.pk,
        })
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'assigned')
        self.assertEqual(conv.assigned_to, self.annotator)

    def test_assign_already_assigned(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_already_assigned', agent=self.agent,
            status='assigned', assigned_to=self.annotator
        )
        other = User.objects.create_user(username='other2', password='p', role='annotator')
        response = self.client.post('/admin-panel/assign/', {
            'conversation_ids': [conv.pk],
            'assignee': other.pk,
        })
        self.assertEqual(response.status_code, 302)
        conv.refresh_from_db()
        # Should NOT be reassigned since it's not unassigned
        self.assertEqual(conv.assigned_to, self.annotator)

    def test_auto_distribute(self):
        self.client.login(username='admin', password='admin')
        for i in range(4):
            Conversation.objects.create(
                elevenlabs_id=f'conv_auto_{i}', agent=self.agent, status='unassigned'
            )
        response = self.client.post('/admin-panel/assign/auto/')
        self.assertEqual(response.status_code, 302)
        assigned = Conversation.objects.filter(status='assigned').count()
        self.assertEqual(assigned, 4)

    def test_auto_distribute_no_annotators(self):
        self.client.login(username='admin', password='admin')
        User.objects.filter(role='annotator').update(is_active=False)
        Conversation.objects.create(
            elevenlabs_id='conv_auto_no_ann', agent=self.agent, status='unassigned'
        )
        response = self.client.post('/admin-panel/assign/auto/')
        self.assertEqual(response.status_code, 302)
        # Nothing should be assigned
        unassigned = Conversation.objects.filter(status='unassigned').count()
        self.assertEqual(unassigned, 1)

    def test_auto_distribute_even(self):
        self.client.login(username='admin', password='admin')
        ann2 = User.objects.create_user(username='ann2', password='p', role='annotator')
        for i in range(6):
            Conversation.objects.create(
                elevenlabs_id=f'conv_even_{i}', agent=self.agent, status='unassigned'
            )
        response = self.client.post('/admin-panel/assign/auto/')
        self.assertEqual(response.status_code, 302)
        ann1_count = Conversation.objects.filter(assigned_to=self.annotator).count()
        ann2_count = Conversation.objects.filter(assigned_to=ann2).count()
        self.assertEqual(ann1_count, 3)
        self.assertEqual(ann2_count, 3)

    # ---- Team ----

    def test_team_invite(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/team/invite/', {
            'username': 'newuser', 'password': 'newpass123',
            'first_name': 'New', 'last_name': 'User',
        })
        self.assertEqual(response.status_code, 302)
        new_user = User.objects.get(username='newuser')
        self.assertEqual(new_user.role, 'annotator')

    def test_team_invite_duplicate_username(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/team/invite/', {
            'username': 'annotator', 'password': 'pass',
        })
        # View handles duplicate gracefully, returning form with error
        self.assertEqual(response.status_code, 200)

    def test_team_toggle_active(self):
        self.client.login(username='admin', password='admin')
        self.assertTrue(self.annotator.is_active)
        self.client.post(f'/admin-panel/team/{self.annotator.pk}/toggle/')
        self.annotator.refresh_from_db()
        self.assertFalse(self.annotator.is_active)
        # Toggle back
        self.client.post(f'/admin-panel/team/{self.annotator.pk}/toggle/')
        self.annotator.refresh_from_db()
        self.assertTrue(self.annotator.is_active)

    def test_team_management_page(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/team/')
        self.assertEqual(response.status_code, 200)

    # ---- Prompts ----

    def test_prompt_create(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/prompts/add/', {
            'name': 'Test Prompt', 'content': 'Hello world', 'is_active': 'on',
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(SystemPrompt.objects.filter(name='Test Prompt').exists())

    def test_prompt_edit(self):
        self.client.login(username='admin', password='admin')
        prompt = SystemPrompt.objects.create(name='Old', content='old', is_active=False)
        response = self.client.post(f'/admin-panel/prompts/{prompt.pk}/edit/', {
            'name': 'Updated', 'content': 'new content',
        })
        self.assertEqual(response.status_code, 302)
        prompt.refresh_from_db()
        self.assertEqual(prompt.name, 'Updated')
        self.assertEqual(prompt.content, 'new content')

    def test_prompt_activate(self):
        self.client.login(username='admin', password='admin')
        prompt = SystemPrompt.objects.create(name='Inactive', content='c', is_active=False)
        response = self.client.post(f'/admin-panel/prompts/{prompt.pk}/activate/')
        self.assertEqual(response.status_code, 302)
        prompt.refresh_from_db()
        self.assertTrue(prompt.is_active)

    def test_prompt_management_page(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/prompts/')
        self.assertEqual(response.status_code, 200)

    # ---- Agents ----

    def test_agent_add(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/agents/add/', {
            'agent_id': 'new_agent', 'label': 'New Agent', 'api_key': 'key123',
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Agent.objects.filter(agent_id='new_agent').exists())

    def test_agent_edit(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post(f'/admin-panel/agents/{self.agent.pk}/edit/', {
            'agent_id': 'agent_view', 'label': 'Updated Agent', 'api_key': 'new_key',
        })
        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.label, 'Updated Agent')

    def test_agent_delete(self):
        self.client.login(username='admin', password='admin')
        agent_pk = self.agent.pk
        response = self.client.post(f'/admin-panel/agents/{agent_pk}/delete/')
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Agent.objects.filter(pk=agent_pk).exists())

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_agent_sync_view(self, MockClient):
        self.client.login(username='admin', password='admin')
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {'conversations': []}
        response = self.client.post(f'/admin-panel/agents/{self.agent.pk}/sync/')
        self.assertEqual(response.status_code, 302)

    # ---- Export ----

    def test_export_preview(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/export/preview/')
        self.assertEqual(response.status_code, 200)

    def test_export_page(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/export/')
        self.assertEqual(response.status_code, 200)

    def test_export_download_no_approved(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/export/download/?include_system_prompt&include_tools')
        self.assertEqual(response.status_code, 302)

    def _create_approved_conversations(self, count, prefix='conv_bulk'):
        """Helper to create approved conversations with user+agent turns."""
        for i in range(count):
            conv = Conversation.objects.create(
                elevenlabs_id=f'{prefix}_{i}', agent=self.agent, status='approved',
                call_timestamp=timezone.now()
            )
            Turn.objects.create(conversation=conv, position=0, role='user', original_text=f'Hi {i}')
            Turn.objects.create(conversation=conv, position=1, role='agent', original_text=f'Hello {i}')

    def test_export_download_with_data(self):
        self.client.login(username='admin', password='admin')
        SystemPrompt.objects.create(name='P', content='Prompt', is_active=True)
        self._create_approved_conversations(10, prefix='conv_dl')
        response = self.client.get('/admin-panel/export/download/?include_system_prompt&include_tools')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/jsonl')
        content = response.content.decode()
        first_line = content.strip().split('\n')[0]
        parsed = json.loads(first_line)
        self.assertIn('messages', parsed)

    def test_export_download_blocked_under_10(self):
        self.client.login(username='admin', password='admin')
        SystemPrompt.objects.create(name='P', content='Prompt', is_active=True)
        self._create_approved_conversations(5, prefix='conv_few')
        response = self.client.get('/admin-panel/export/download/?include_system_prompt&include_tools')
        self.assertEqual(response.status_code, 302)

    def test_export_download_split(self):
        self.client.login(username='admin', password='admin')
        SystemPrompt.objects.create(name='P', content='Prompt', is_active=True)
        self._create_approved_conversations(10, prefix='conv_split_dl')
        response = self.client.get('/admin-panel/export/download/?include_system_prompt&include_tools&split')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/zip')

    def test_export_creates_log(self):
        self.client.login(username='admin', password='admin')
        SystemPrompt.objects.create(name='P', content='Prompt', is_active=True)
        self._create_approved_conversations(10, prefix='conv_log')
        self.assertEqual(ExportLog.objects.count(), 0)
        self.client.get('/admin-panel/export/download/?include_system_prompt&include_tools')
        self.assertEqual(ExportLog.objects.count(), 1)

    # ---- Analytics ----

    def test_analytics_page(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/analytics/')
        self.assertEqual(response.status_code, 200)

    # ---- Review ----

    def test_review_queue(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/review/')
        self.assertEqual(response.status_code, 200)

    def test_review_conversation_detail(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_review_detail', agent=self.agent,
            assigned_to=self.annotator, status='completed'
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='test')
        response = self.client.get(f'/admin-panel/review/{conv.pk}/')
        self.assertEqual(response.status_code, 200)

    # ---- Audio Proxy ----

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_happy_path_assigned_user(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_001', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        fake_audio = b'\x00\x01\x02\x03\x04\x05'
        mock_instance.get_conversation_audio.return_value = fake_audio

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'audio/mpeg')
        self.assertEqual(response['Content-Length'], '6')
        self.assertEqual(response['Content-Disposition'], 'inline')
        self.assertEqual(response['Cache-Control'], 'private, max-age=3600')
        self.assertEqual(response.content, fake_audio)

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_happy_path_admin(self, MockClient):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_admin', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        fake_audio = b'\xFF\xFE\xFD\xFC'
        mock_instance.get_conversation_audio.return_value = fake_audio

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'audio/mpeg')
        self.assertEqual(response['Content-Length'], '4')
        self.assertEqual(response.content, fake_audio)

    def test_conversation_audio_permission_denied_unassigned_annotator(self):
        self.client.login(username='annotator', password='annotator')
        other_annotator = User.objects.create_user(
            username='other_ann', password='other', role='annotator'
        )
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_other', agent=self.agent,
            assigned_to=other_annotator, status='in_progress',
            has_audio=True
        )
        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 403)
        self.assertIn('Permission denied', response.content.decode())

    def test_conversation_audio_unauthenticated_redirect(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_unauth', agent=self.agent,
            has_audio=True
        )
        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)

    def test_conversation_audio_has_audio_false(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_no_audio', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=False
        )
        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Audio not available', response.content.decode())

    def test_conversation_audio_missing_elevenlabs_id(self):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Audio not available', response.content.decode())

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_elevenlabs_422_error(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_422', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 422
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_instance.get_conversation_audio.side_effect = http_error

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Audio no longer available', response.content.decode())

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_elevenlabs_500_error(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_500', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_instance.get_conversation_audio.side_effect = http_error

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 502)
        self.assertIn('Failed to fetch audio', response.content.decode())

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_elevenlabs_404_error(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_404', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_instance.get_conversation_audio.side_effect = http_error

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 502)
        self.assertIn('Failed to fetch audio', response.content.decode())

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_http_error_no_response(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_no_resp', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        http_error = requests.exceptions.HTTPError()
        http_error.response = None
        mock_instance.get_conversation_audio.side_effect = http_error

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 502)
        self.assertIn('Failed to fetch audio', response.content.decode())

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_generic_exception(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_exc', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        mock_instance.get_conversation_audio.side_effect = Exception("Network timeout")

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 502)
        self.assertIn('Failed to fetch audio', response.content.decode())

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_content_length_matches(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_length', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        # Large audio file simulation
        large_audio = b'\x00' * 1024 * 100  # 100KB
        mock_instance.get_conversation_audio.return_value = large_audio

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Length'], str(len(large_audio)))
        self.assertEqual(len(response.content), len(large_audio))

    @patch('conversations.views.ElevenLabsClient')
    def test_conversation_audio_empty_audio_bytes(self, MockClient):
        self.client.login(username='annotator', password='annotator')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_audio_empty', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            has_audio=True
        )
        mock_instance = MockClient.return_value
        mock_instance.get_conversation_audio.return_value = b''

        response = self.client.get(f'/conversations/{conv.pk}/audio/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Length'], '0')
        self.assertEqual(response.content, b'')

    def test_conversation_audio_nonexistent_conversation(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/99999/audio/')
        self.assertEqual(response.status_code, 404)


# =============================================================================
# ANNOTATION FEATURE TESTS (8 new features)
# =============================================================================

class AnnotationFeatureTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='admin', password='admin', role='admin'
        )
        self.annotator = User.objects.create_user(
            username='annotator', password='annotator', role='annotator'
        )
        self.agent = Agent.objects.create(
            agent_id='agent_feat', label='Feature Agent', elevenlabs_api_key='key'
        )
        self.prompt = SystemPrompt.objects.create(
            name='Test Prompt', content='You are a test assistant.', is_active=True
        )
        self.conv = Conversation.objects.create(
            elevenlabs_id='conv_feat_001', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            call_timestamp=timezone.now()
        )
        self.turn0 = Turn.objects.create(
            conversation=self.conv, position=0, role='agent',
            original_text='Welcome! How can I help?'
        )
        self.turn1 = Turn.objects.create(
            conversation=self.conv, position=1, role='user',
            original_text='I want a pizza.'
        )
        self.turn2 = Turn.objects.create(
            conversation=self.conv, position=2, role='agent',
            original_text='Let me place that order.'
        )
        self.tc = ToolCall.objects.create(
            turn=self.turn2, tool_name='create_order',
            original_args={'customerName': 'Test', 'customerPhone': '555', 'items': []},
            status_code=200,
            response_body={'success': True, 'orderId': 'ORD-1'},
        )
        self.turn3 = Turn.objects.create(
            conversation=self.conv, position=3, role='agent',
            original_text='Order placed!'
        )

    # ---- Feature 1: Turn Soft Delete ----

    def test_turn_soft_delete_toggle(self):
        self.client.login(username='annotator', password='annotator')
        self.assertFalse(self.turn1.is_deleted)
        # Delete
        response = self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn1.pk}/delete/'
        )
        self.assertEqual(response.status_code, 200)
        self.turn1.refresh_from_db()
        self.assertTrue(self.turn1.is_deleted)
        # Restore
        response = self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn1.pk}/delete/'
        )
        self.assertEqual(response.status_code, 200)
        self.turn1.refresh_from_db()
        self.assertFalse(self.turn1.is_deleted)

    def test_deleted_turn_excluded_from_export(self):
        self.conv.status = 'approved'
        self.conv.save()
        self.turn1.is_deleted = True
        self.turn1.save()
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        user_msgs = [m for m in msgs if m.get('role') == 'user']
        self.assertEqual(len(user_msgs), 0)

    def test_deleted_turn_shows_in_review(self):
        self.conv.status = 'completed'
        self.conv.save()
        self.turn1.is_deleted = True
        self.turn1.save()
        self.client.login(username='admin', password='admin')
        response = self.client.get(f'/admin-panel/review/{self.conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Deleted by annotator')

    # ---- Feature 2: Tool Call Soft Delete ----

    def test_tool_call_soft_delete_toggle(self):
        self.client.login(username='annotator', password='annotator')
        self.assertFalse(self.tc.is_deleted)
        # Delete
        response = self.client.post(
            f'/conversations/{self.conv.pk}/tool/{self.tc.pk}/delete/'
        )
        self.assertEqual(response.status_code, 200)
        self.tc.refresh_from_db()
        self.assertTrue(self.tc.is_deleted)
        # Restore
        response = self.client.post(
            f'/conversations/{self.conv.pk}/tool/{self.tc.pk}/delete/'
        )
        self.assertEqual(response.status_code, 200)
        self.tc.refresh_from_db()
        self.assertFalse(self.tc.is_deleted)

    def test_deleted_tool_call_excluded_from_export(self):
        self.conv.status = 'approved'
        self.conv.save()
        self.tc.is_deleted = True
        self.tc.save()
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        tc_msgs = [m for m in msgs if 'tool_calls' in m]
        self.assertEqual(len(tc_msgs), 0)
        tool_msgs = [m for m in msgs if m['role'] == 'tool']
        self.assertEqual(len(tool_msgs), 0)

    def test_deleted_tool_call_excluded_from_tools_used(self):
        self.conv.status = 'approved'
        self.conv.save()
        self.tc.is_deleted = True
        self.tc.save()
        result = conversation_to_messages(self.conv, include_tools=True)
        self.assertNotIn('tools', result)

    # ---- Feature 3: Weight Control ----

    def test_weight_toggle_cycle(self):
        self.client.login(username='annotator', password='annotator')
        # Agent turn starts at None
        self.assertIsNone(self.turn0.weight)
        # None -> 0
        self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn0.pk}/weight/'
        )
        self.turn0.refresh_from_db()
        self.assertEqual(self.turn0.weight, 0)
        # 0 -> 1
        self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn0.pk}/weight/'
        )
        self.turn0.refresh_from_db()
        self.assertEqual(self.turn0.weight, 1)
        # 1 -> None
        self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn0.pk}/weight/'
        )
        self.turn0.refresh_from_db()
        self.assertIsNone(self.turn0.weight)

    def test_weight_toggle_only_agent_turns(self):
        self.client.login(username='annotator', password='annotator')
        # Try toggling weight on a user turn - should not change
        self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn1.pk}/weight/'
        )
        self.turn1.refresh_from_db()
        self.assertIsNone(self.turn1.weight)

    def test_weight_respected_in_export(self):
        self.conv.status = 'approved'
        self.conv.save()
        # Set weight=0 on a turn that would normally have weight=1 (after user)
        self.turn3.weight = 0
        self.turn3.save()
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        # The last assistant message ("Order placed!") should have weight=0
        assistant_msgs = [m for m in msgs if m.get('role') == 'assistant' and 'tool_calls' not in m]
        last_assistant = assistant_msgs[-1]
        self.assertEqual(last_assistant.get('weight'), 0)

    def test_weight_auto_fallback(self):
        self.conv.status = 'approved'
        self.conv.save()
        # With no manual weight, pre-first-user messages get weight 0
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        # First assistant message (greeting) should have weight 0 (auto)
        assistant_msgs = [m for m in msgs if m.get('role') == 'assistant']
        self.assertEqual(assistant_msgs[0].get('weight'), 0)

    # ---- Feature 4: Diff View ----

    def test_diff_view_edit_stats(self):
        self.conv.status = 'completed'
        self.conv.save()
        self.turn0.is_edited = True
        self.turn0.edited_text = 'Welcome!'
        self.turn0.save()
        self.turn1.is_deleted = True
        self.turn1.save()
        self.tc.is_deleted = True
        self.tc.save()
        self.turn3.weight = 0
        self.turn3.save()

        self.client.login(username='admin', password='admin')
        response = self.client.get(f'/admin-panel/review/{self.conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Edit Summary')
        self.assertContains(response, '1 turn edited')
        self.assertContains(response, '1 turn deleted')
        self.assertContains(response, '1 tool call deleted')
        self.assertContains(response, '1 weight override')

    # ---- Feature 5: Search ----

    def test_conversation_search(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/?status=all&q=pizza')
        self.assertEqual(response.status_code, 200)
        # ID is truncated in template, check for partial match
        self.assertContains(response, 'conv_feat')

    def test_conversation_search_no_results(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/?status=all&q=nonexistent_term_xyz')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No conversations found')

    def test_conversation_search_by_elevenlabs_id(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/?status=all&q=conv_feat_001')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'conv_feat')

    def test_admin_assign_search(self):
        self.client.login(username='admin', password='admin')
        conv = Conversation.objects.create(
            elevenlabs_id='conv_search_assign', agent=self.agent, status='unassigned'
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='I want sushi')
        response = self.client.get('/admin-panel/assign/?status=unassigned&q=sushi')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'conv_sear')

    # ---- Feature 6: Bulk Approve ----

    def test_bulk_approve(self):
        self.client.login(username='admin', password='admin')
        convs = []
        for i in range(3):
            c = Conversation.objects.create(
                elevenlabs_id=f'conv_bulk_{i}', agent=self.agent,
                assigned_to=self.annotator, status='completed',
                call_timestamp=timezone.now()
            )
            convs.append(c)
        response = self.client.post('/admin-panel/review/bulk-approve/', {
            'conversation_ids': [c.pk for c in convs],
        })
        self.assertEqual(response.status_code, 302)
        for c in convs:
            c.refresh_from_db()
            self.assertEqual(c.status, 'approved')
            self.assertIsNotNone(c.reviewed_at)

    def test_bulk_approve_only_completed(self):
        self.client.login(username='admin', password='admin')
        conv_completed = Conversation.objects.create(
            elevenlabs_id='conv_bulk_completed', agent=self.agent,
            status='completed', call_timestamp=timezone.now()
        )
        conv_in_progress = Conversation.objects.create(
            elevenlabs_id='conv_bulk_ip', agent=self.agent,
            status='in_progress', call_timestamp=timezone.now()
        )
        self.client.post('/admin-panel/review/bulk-approve/', {
            'conversation_ids': [conv_completed.pk, conv_in_progress.pk],
        })
        conv_completed.refresh_from_db()
        conv_in_progress.refresh_from_db()
        self.assertEqual(conv_completed.status, 'approved')
        self.assertEqual(conv_in_progress.status, 'in_progress')

    # ---- Feature 7: Tagging ----

    def test_tag_create_and_assign(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.post(
            f'/conversations/{self.conv.pk}/tags/',
            {'action': 'add', 'tag_name': 'order-flow'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(Tag.objects.filter(name='order-flow').exists())
        self.assertIn('order-flow', [t.name for t in self.conv.tags.all()])

    def test_tag_remove(self):
        self.client.login(username='annotator', password='annotator')
        tag = Tag.objects.create(name='test-tag')
        self.conv.tags.add(tag)
        response = self.client.post(
            f'/conversations/{self.conv.pk}/tags/',
            {'action': 'remove', 'tag_id': tag.pk}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(tag, self.conv.tags.all())

    def test_tag_reuse_existing(self):
        self.client.login(username='annotator', password='annotator')
        tag = Tag.objects.create(name='existing-tag')
        self.client.post(
            f'/conversations/{self.conv.pk}/tags/',
            {'action': 'add', 'tag_name': 'existing-tag'}
        )
        self.assertEqual(Tag.objects.filter(name='existing-tag').count(), 1)
        self.assertIn(tag, self.conv.tags.all())

    def test_tag_filter_in_export(self):
        # Create two approved conversations, one tagged
        conv1 = Conversation.objects.create(
            elevenlabs_id='conv_tag_1', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv1, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv1, position=1, role='agent', original_text='Hello')
        tag = Tag.objects.create(name='high-quality')
        conv1.tags.add(tag)

        conv2 = Conversation.objects.create(
            elevenlabs_id='conv_tag_2', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv2, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv2, position=1, role='agent', original_text='Hello')

        # Without filter - both
        all_examples = generate_jsonl_examples()
        self.assertEqual(len(all_examples), 2)

        # With filter - only tagged
        filtered = generate_jsonl_examples(tag_filter='high-quality')
        self.assertEqual(len(filtered), 1)

    # ---- Feature 8: Turn Insertion ----

    def test_turn_insert_shifts_positions(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn1.pk}/insert/',
            {'role': 'agent', 'text': 'One moment please.'}
        )
        self.assertEqual(response.status_code, 200)
        # Check the new turn was created
        new_turn = Turn.objects.get(
            conversation=self.conv, is_inserted=True
        )
        self.assertEqual(new_turn.position, 2)
        self.assertEqual(new_turn.role, 'agent')
        self.assertEqual(new_turn.original_text, 'One moment please.')
        # Check subsequent turns shifted
        self.turn2.refresh_from_db()
        self.assertEqual(self.turn2.position, 3)
        self.turn3.refresh_from_db()
        self.assertEqual(self.turn3.position, 4)

    def test_turn_insert_requires_text(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.post(
            f'/conversations/{self.conv.pk}/turn/{self.turn1.pk}/insert/',
            {'role': 'agent', 'text': ''}
        )
        self.assertEqual(response.status_code, 400)

    def test_turn_insert_get_form(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get(
            f'/conversations/{self.conv.pk}/turn/{self.turn1.pk}/insert/'
        )
        self.assertEqual(response.status_code, 200)

    def test_inserted_turn_in_export(self):
        self.conv.status = 'approved'
        self.conv.save()
        # Insert a turn - shift in reverse order to avoid unique constraint
        for t in Turn.objects.filter(
            conversation=self.conv, position__gt=self.turn1.position
        ).order_by('-position'):
            t.position += 1
            t.save()
        new_turn = Turn.objects.create(
            conversation=self.conv, position=2,
            role='agent', original_text='Let me check on that.',
            is_inserted=True,
        )
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        assistant_msgs = [m for m in msgs if m.get('role') == 'assistant' and 'tool_calls' not in m]
        contents = [m['content'] for m in assistant_msgs]
        self.assertIn('Let me check on that.', contents)

    def test_inserted_turn_shows_in_review(self):
        self.conv.status = 'completed'
        self.conv.save()
        Turn.objects.create(
            conversation=self.conv, position=10,
            role='agent', original_text='Inserted text.',
            is_inserted=True,
        )
        self.client.login(username='admin', password='admin')
        response = self.client.get(f'/admin-panel/review/{self.conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Inserted')

    # ---- Cross-feature: Editor loads with new features ----

    def test_editor_loads_with_all_features(self):
        self.client.login(username='annotator', password='annotator')
        tag = Tag.objects.create(name='test-tag')
        self.conv.tags.add(tag)
        response = self.client.get(f'/conversations/{self.conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'test-tag')
        self.assertContains(response, 'W: auto')


# =============================================================================
# RAG CONTEXT TESTS
# =============================================================================

class RagContextModelTests(TestCase):
    """Tests for the rag_context field on the Turn model."""

    def setUp(self):
        self.agent = Agent.objects.create(
            agent_id='agent_rag', label='RAG Agent', elevenlabs_api_key='test-key'
        )
        self.conv = Conversation.objects.create(
            elevenlabs_id='conv_rag_model', agent=self.agent
        )

    def test_rag_context_default_is_empty_list(self):
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent', original_text='Hello'
        )
        self.assertEqual(turn.rag_context, [])

    def test_rag_context_stores_list_of_dicts(self):
        rag_data = [
            {
                'document_id': 'doc_001',
                'chunk_id': 'chunk_001',
                'content': 'Large Pepperoni Pizza $14.99',
                'vector_distance': 0.123,
            },
            {
                'document_id': 'doc_002',
                'chunk_id': 'chunk_005',
                'content': 'We are open Monday through Saturday',
                'vector_distance': 0.456,
            },
        ]
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent',
            original_text='Hello', rag_context=rag_data
        )
        turn.refresh_from_db()
        self.assertEqual(len(turn.rag_context), 2)
        self.assertEqual(turn.rag_context[0]['document_id'], 'doc_001')
        self.assertEqual(turn.rag_context[0]['content'], 'Large Pepperoni Pizza $14.99')
        self.assertAlmostEqual(turn.rag_context[1]['vector_distance'], 0.456)

    def test_rag_context_round_trips_through_database(self):
        rag_data = [{'document_id': 'doc_x', 'chunk_id': 'ch_y', 'content': 'Test', 'vector_distance': 0.1}]
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent',
            original_text='Hi', rag_context=rag_data
        )
        fetched = Turn.objects.get(pk=turn.pk)
        self.assertEqual(fetched.rag_context, rag_data)

    def test_rag_context_update_fields(self):
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent', original_text='Hi'
        )
        self.assertEqual(turn.rag_context, [])
        turn.rag_context = [{'document_id': 'd', 'chunk_id': 'c', 'content': 'text', 'vector_distance': 0.5}]
        turn.save(update_fields=['rag_context'])
        turn.refresh_from_db()
        self.assertEqual(len(turn.rag_context), 1)

    def test_rag_context_empty_list_is_falsy(self):
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent', original_text='Hi'
        )
        self.assertFalse(turn.rag_context)

    def test_rag_context_with_fetch_error(self):
        rag_data = [{
            'document_id': 'doc_err', 'chunk_id': 'ch_err',
            'content': '', 'vector_distance': 0.2,
            'fetch_error': 'HTTP 404',
        }]
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent',
            original_text='Hi', rag_context=rag_data
        )
        turn.refresh_from_db()
        self.assertEqual(turn.rag_context[0]['fetch_error'], 'HTTP 404')
        self.assertEqual(turn.rag_context[0]['content'], '')

    def test_rag_context_on_user_turn(self):
        """rag_context can technically be on any turn, though ElevenLabs puts it on agent turns."""
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='user',
            original_text='I want pizza',
            rag_context=[{'document_id': 'd', 'chunk_id': 'c', 'content': 'menu', 'vector_distance': 0.1}]
        )
        turn.refresh_from_db()
        self.assertEqual(len(turn.rag_context), 1)

    def test_rag_context_many_chunks(self):
        chunks = [
            {'document_id': f'doc_{i}', 'chunk_id': f'ch_{i}', 'content': f'chunk {i}', 'vector_distance': 0.1 * i}
            for i in range(20)
        ]
        turn = Turn.objects.create(
            conversation=self.conv, position=0, role='agent',
            original_text='Hi', rag_context=chunks
        )
        turn.refresh_from_db()
        self.assertEqual(len(turn.rag_context), 20)


class RagContextElevenLabsClientTests(TestCase):
    """Tests for the get_kb_chunk() method on ElevenLabsClient."""

    def setUp(self):
        from conversations.services.elevenlabs import ElevenLabsClient
        self.client_cls = ElevenLabsClient

    @patch('conversations.services.elevenlabs.requests.get')
    def test_get_kb_chunk_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'content': 'Large Pepperoni Pizza $14.99, toppings include...',
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = self.client_cls('test-api-key')
        result = client.get_kb_chunk('doc_abc', 'chunk_123')

        self.assertEqual(result['content'], 'Large Pepperoni Pizza $14.99, toppings include...')
        mock_get.assert_called_once_with(
            'https://api.elevenlabs.io/v1/convai/knowledge-base/doc_abc/chunk/chunk_123',
            headers={'xi-api-key': 'test-api-key'},
            timeout=10,
        )

    @patch('conversations.services.elevenlabs.requests.get')
    def test_get_kb_chunk_raises_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        client = self.client_cls('test-api-key')
        with self.assertRaises(requests.exceptions.HTTPError):
            client.get_kb_chunk('doc_missing', 'chunk_missing')

    @patch('conversations.services.elevenlabs.requests.get')
    def test_get_kb_chunk_raises_on_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        client = self.client_cls('test-api-key')
        with self.assertRaises(requests.exceptions.Timeout):
            client.get_kb_chunk('doc_x', 'chunk_y')

    @patch('conversations.services.elevenlabs.requests.get')
    def test_get_kb_chunk_uses_correct_headers(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'content': 'test'}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = self.client_cls('my-secret-key-123')
        client.get_kb_chunk('doc_1', 'ch_1')

        call_kwargs = mock_get.call_args
        self.assertEqual(call_kwargs.kwargs['headers'], {'xi-api-key': 'my-secret-key-123'})

    @patch('conversations.services.elevenlabs.requests.get')
    def test_get_kb_chunk_returns_full_json(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'content': 'Menu item details',
            'metadata': {'source': 'menu.pdf'},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = self.client_cls('key')
        result = client.get_kb_chunk('doc_1', 'ch_1')
        self.assertIn('content', result)
        self.assertIn('metadata', result)


class RagContextSyncTests(TestCase):
    """Tests for RAG context extraction during conversation sync."""

    def setUp(self):
        self.agent = Agent.objects.create(
            agent_id='agent_rag_sync', label='RAG Sync Agent', elevenlabs_api_key='test-key'
        )

    def test_sync_extracts_rag_context_from_agent_turn(self):
        """When an agent turn has rag_retrieval_info, chunks are fetched and stored."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.return_value = {'content': 'Pepperoni Pizza $14.99'}

        data = {
            'transcript': [
                {'role': 'user', 'message': 'What pizzas do you have?', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'We have pepperoni pizza!',
                    'time_in_call_secs': 3.0,
                    'rag_retrieval_info': {
                        'chunks': [
                            {'document_id': 'doc_menu', 'chunk_id': 'ch_pizza', 'vector_distance': 0.15},
                        ]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_rag_sync_1', data, mock_client)

        agent_turn = conv.turns.get(position=1)
        self.assertEqual(len(agent_turn.rag_context), 1)
        self.assertEqual(agent_turn.rag_context[0]['document_id'], 'doc_menu')
        self.assertEqual(agent_turn.rag_context[0]['chunk_id'], 'ch_pizza')
        self.assertEqual(agent_turn.rag_context[0]['content'], 'Pepperoni Pizza $14.99')
        self.assertAlmostEqual(agent_turn.rag_context[0]['vector_distance'], 0.15)
        mock_client.get_kb_chunk.assert_called_once_with('doc_menu', 'ch_pizza')

    def test_sync_extracts_multiple_rag_chunks(self):
        """Multiple chunks in a single turn are all fetched."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.side_effect = [
            {'content': 'Pepperoni Pizza $14.99'},
            {'content': 'Margherita Pizza $12.99'},
            {'content': 'Hawaiian Pizza $13.99'},
        ]

        data = {
            'transcript': [
                {'role': 'user', 'message': 'Show me pizzas', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Here are our pizzas...',
                    'time_in_call_secs': 3.0,
                    'rag_retrieval_info': {
                        'chunks': [
                            {'document_id': 'doc_m', 'chunk_id': 'ch_1', 'vector_distance': 0.1},
                            {'document_id': 'doc_m', 'chunk_id': 'ch_2', 'vector_distance': 0.2},
                            {'document_id': 'doc_m', 'chunk_id': 'ch_3', 'vector_distance': 0.3},
                        ]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_rag_multi', data, mock_client)

        agent_turn = conv.turns.get(position=1)
        self.assertEqual(len(agent_turn.rag_context), 3)
        self.assertEqual(agent_turn.rag_context[0]['content'], 'Pepperoni Pizza $14.99')
        self.assertEqual(agent_turn.rag_context[1]['content'], 'Margherita Pizza $12.99')
        self.assertEqual(agent_turn.rag_context[2]['content'], 'Hawaiian Pizza $13.99')
        self.assertEqual(mock_client.get_kb_chunk.call_count, 3)

    def test_sync_handles_chunk_fetch_failure_gracefully(self):
        """If a chunk fetch fails, store with empty content and fetch_error."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.side_effect = Exception("Connection refused")

        data = {
            'transcript': [
                {'role': 'user', 'message': 'What do you have?', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Let me check...',
                    'time_in_call_secs': 3.0,
                    'rag_retrieval_info': {
                        'chunks': [
                            {'document_id': 'doc_fail', 'chunk_id': 'ch_fail', 'vector_distance': 0.5},
                        ]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_rag_fail', data, mock_client)

        agent_turn = conv.turns.get(position=1)
        self.assertEqual(len(agent_turn.rag_context), 1)
        self.assertEqual(agent_turn.rag_context[0]['content'], '')
        self.assertIn('Connection refused', agent_turn.rag_context[0]['fetch_error'])
        self.assertEqual(agent_turn.rag_context[0]['document_id'], 'doc_fail')

    def test_sync_partial_chunk_failure(self):
        """If one chunk fails but another succeeds, both are stored."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.side_effect = [
            {'content': 'Successful chunk content'},
            Exception("API Error"),
        ]

        data = {
            'transcript': [
                {'role': 'user', 'message': 'Tell me about the menu', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Here is what we have...',
                    'time_in_call_secs': 3.0,
                    'rag_retrieval_info': {
                        'chunks': [
                            {'document_id': 'doc_ok', 'chunk_id': 'ch_ok', 'vector_distance': 0.1},
                            {'document_id': 'doc_bad', 'chunk_id': 'ch_bad', 'vector_distance': 0.9},
                        ]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_rag_partial', data, mock_client)

        agent_turn = conv.turns.get(position=1)
        self.assertEqual(len(agent_turn.rag_context), 2)
        self.assertEqual(agent_turn.rag_context[0]['content'], 'Successful chunk content')
        self.assertFalse('fetch_error' in agent_turn.rag_context[0])
        self.assertEqual(agent_turn.rag_context[1]['content'], '')
        self.assertIn('API Error', agent_turn.rag_context[1]['fetch_error'])

    def test_sync_no_rag_info_leaves_empty(self):
        """Turns without rag_retrieval_info have empty rag_context."""
        mock_client = MagicMock()

        data = {
            'transcript': [
                {'role': 'user', 'message': 'Hi', 'time_in_call_secs': 1.0},
                {'role': 'agent', 'message': 'Hello!', 'time_in_call_secs': 2.0},
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_no_rag', data, mock_client)

        for turn in conv.turns.all():
            self.assertEqual(turn.rag_context, [])
        mock_client.get_kb_chunk.assert_not_called()

    def test_sync_empty_chunks_list(self):
        """rag_retrieval_info with empty chunks list doesn't trigger fetches."""
        mock_client = MagicMock()

        data = {
            'transcript': [
                {'role': 'user', 'message': 'Hi', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Hello!', 'time_in_call_secs': 2.0,
                    'rag_retrieval_info': {'chunks': []},
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_empty_chunks', data, mock_client)

        agent_turn = conv.turns.get(position=1)
        self.assertEqual(agent_turn.rag_context, [])
        mock_client.get_kb_chunk.assert_not_called()

    def test_sync_skips_chunks_missing_ids(self):
        """Chunks without document_id or chunk_id are skipped."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.return_value = {'content': 'valid content'}

        data = {
            'transcript': [
                {'role': 'user', 'message': 'Hi', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Hello!', 'time_in_call_secs': 2.0,
                    'rag_retrieval_info': {
                        'chunks': [
                            {'document_id': '', 'chunk_id': 'ch_1', 'vector_distance': 0.1},
                            {'document_id': 'doc_1', 'chunk_id': '', 'vector_distance': 0.2},
                            {'document_id': 'doc_2', 'chunk_id': 'ch_2', 'vector_distance': 0.3},
                        ]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_missing_ids', data, mock_client)

        agent_turn = conv.turns.get(position=1)
        self.assertEqual(len(agent_turn.rag_context), 1)
        self.assertEqual(agent_turn.rag_context[0]['document_id'], 'doc_2')
        mock_client.get_kb_chunk.assert_called_once_with('doc_2', 'ch_2')

    def test_sync_rag_on_multiple_agent_turns(self):
        """Multiple agent turns can each have their own RAG context."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.side_effect = [
            {'content': 'Pizza menu section'},
            {'content': 'Hours of operation'},
        ]

        data = {
            'transcript': [
                {'role': 'user', 'message': 'What pizzas?', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'We have...',
                    'time_in_call_secs': 3.0,
                    'rag_retrieval_info': {
                        'chunks': [{'document_id': 'd1', 'chunk_id': 'c1', 'vector_distance': 0.1}]
                    },
                },
                {'role': 'user', 'message': 'What are your hours?', 'time_in_call_secs': 5.0},
                {
                    'role': 'agent', 'message': 'We are open...',
                    'time_in_call_secs': 7.0,
                    'rag_retrieval_info': {
                        'chunks': [{'document_id': 'd2', 'chunk_id': 'c2', 'vector_distance': 0.2}]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_multi_rag', data, mock_client)

        turn1 = conv.turns.get(position=1)
        turn3 = conv.turns.get(position=3)
        self.assertEqual(len(turn1.rag_context), 1)
        self.assertEqual(turn1.rag_context[0]['content'], 'Pizza menu section')
        self.assertEqual(len(turn3.rag_context), 1)
        self.assertEqual(turn3.rag_context[0]['content'], 'Hours of operation')

    def test_sync_preserves_null_vector_distance(self):
        """vector_distance can be None/missing and is stored as-is."""
        mock_client = MagicMock()
        mock_client.get_kb_chunk.return_value = {'content': 'some content'}

        data = {
            'transcript': [
                {'role': 'user', 'message': 'Hi', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Hello!', 'time_in_call_secs': 2.0,
                    'rag_retrieval_info': {
                        'chunks': [{'document_id': 'd1', 'chunk_id': 'c1'}]
                    },
                },
            ],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_null_dist', data, mock_client)

        turn = conv.turns.get(position=1)
        self.assertEqual(len(turn.rag_context), 1)
        self.assertIsNone(turn.rag_context[0]['vector_distance'])

    @patch('conversations.services.sync.ElevenLabsClient')
    def test_full_sync_with_rag(self, MockClient):
        """End-to-end sync_agent_conversations correctly extracts RAG context."""
        mock_instance = MockClient.return_value
        mock_instance.list_conversations.return_value = {
            'conversations': [{'conversation_id': 'conv_e2e_rag'}],
        }
        mock_instance.get_conversation.return_value = {
            'transcript': [
                {'role': 'user', 'message': 'What is on the menu?', 'time_in_call_secs': 1.0},
                {
                    'role': 'agent', 'message': 'Here is our menu!',
                    'time_in_call_secs': 3.0,
                    'rag_retrieval_info': {
                        'chunks': [
                            {'document_id': 'doc_m', 'chunk_id': 'ch_1', 'vector_distance': 0.12},
                        ]
                    },
                },
            ],
            'metadata': {'start_time_unix_secs': 1714423232, 'call_duration_secs': 30},
            'has_audio': False,
        }
        mock_instance.get_kb_chunk.return_value = {'content': 'Full menu: Pizza $14.99, Pasta $12.99'}

        stats = sync_agent_conversations(self.agent)
        self.assertEqual(stats['imported'], 1)

        conv = Conversation.objects.get(elevenlabs_id='conv_e2e_rag')
        agent_turn = conv.turns.get(role='agent')
        self.assertEqual(len(agent_turn.rag_context), 1)
        self.assertEqual(agent_turn.rag_context[0]['content'], 'Full menu: Pizza $14.99, Pasta $12.99')
        mock_instance.get_kb_chunk.assert_called_once_with('doc_m', 'ch_1')


class RagContextExportTests(TestCase):
    """Tests for RAG context injection into JSONL training data export."""

    def setUp(self):
        self.agent = Agent.objects.create(
            agent_id='agent_rag_export', label='RAG Export Agent', elevenlabs_api_key='key'
        )
        self.prompt = SystemPrompt.objects.create(
            name='RAG Test Prompt', content='You are a restaurant assistant.', is_active=True
        )

    def _create_conversation_with_rag(self, conv_id='conv_rag_exp_001', status='approved'):
        """Helper: creates a conversation with RAG context on agent turns."""
        conv = Conversation.objects.create(
            elevenlabs_id=conv_id, agent=self.agent, status=status,
            call_timestamp=timezone.now()
        )
        Turn.objects.create(
            conversation=conv, position=0, role='agent',
            original_text='Welcome! How can I help?'
        )
        Turn.objects.create(
            conversation=conv, position=1, role='user',
            original_text='What pizzas do you have?'
        )
        Turn.objects.create(
            conversation=conv, position=2, role='agent',
            original_text='We have pepperoni and margherita!',
            rag_context=[
                {
                    'document_id': 'doc_menu', 'chunk_id': 'ch_pizza',
                    'content': 'Pepperoni Pizza - Large $14.99, Medium $11.99\nMargherita Pizza - Large $12.99',
                    'vector_distance': 0.15,
                },
            ]
        )
        Turn.objects.create(
            conversation=conv, position=3, role='user',
            original_text='I will have a large pepperoni'
        )
        Turn.objects.create(
            conversation=conv, position=4, role='agent',
            original_text='Great choice!'
        )
        return conv

    def test_rag_context_injected_into_preceding_user_message(self):
        """RAG context from agent turn is injected into the preceding user message."""
        conv = self._create_conversation_with_rag()
        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        # Find the user message that should have RAG context
        user_msgs = [m for m in msgs if m['role'] == 'user']
        # Position 1 = "What pizzas do you have?" should have context
        pizza_msg = user_msgs[0]
        self.assertIn('Context:', pizza_msg['content'])
        self.assertIn('Pepperoni Pizza - Large $14.99', pizza_msg['content'])
        self.assertIn('Margherita Pizza - Large $12.99', pizza_msg['content'])

    def test_rag_context_not_injected_when_disabled(self):
        """When include_rag_context=False, no context is added to user messages."""
        conv = self._create_conversation_with_rag()
        result = conversation_to_messages(conv, include_rag_context=False)
        msgs = result['messages']

        user_msgs = [m for m in msgs if m['role'] == 'user']
        for msg in user_msgs:
            self.assertNotIn('Context:', msg['content'])

    def test_rag_context_preserves_original_user_text(self):
        """The original user text is preserved before the context block."""
        conv = self._create_conversation_with_rag()
        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msgs = [m for m in msgs if m['role'] == 'user']
        pizza_msg = user_msgs[0]
        self.assertTrue(pizza_msg['content'].startswith('What pizzas do you have?'))

    def test_rag_context_format_structure(self):
        """The injected context follows the format: text\\n\\nContext:\\ncontent."""
        conv = self._create_conversation_with_rag()
        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msgs = [m for m in msgs if m['role'] == 'user']
        pizza_msg = user_msgs[0]
        parts = pizza_msg['content'].split('\n\nContext:\n')
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], 'What pizzas do you have?')
        self.assertIn('Pepperoni Pizza', parts[1])

    def test_user_without_rag_context_unchanged(self):
        """User messages that don't precede agent turns with RAG are untouched."""
        conv = self._create_conversation_with_rag()
        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msgs = [m for m in msgs if m['role'] == 'user']
        # Second user message ("I will have a large pepperoni") has no RAG
        pepperoni_msg = user_msgs[1]
        self.assertEqual(pepperoni_msg['content'], 'I will have a large pepperoni')
        self.assertNotIn('Context:', pepperoni_msg['content'])

    def test_multiple_rag_chunks_joined(self):
        """Multiple chunks are joined with double newlines."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_multi_chunk_exp', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Tell me about your menu')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='We have many items!',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Pizza section: Pepperoni $14.99', 'vector_distance': 0.1},
                {'document_id': 'd1', 'chunk_id': 'c2', 'content': 'Pasta section: Spaghetti $11.99', 'vector_distance': 0.2},
                {'document_id': 'd2', 'chunk_id': 'c3', 'content': 'Drinks: Coke $2.99, Sprite $2.99', 'vector_distance': 0.3},
            ]
        )

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msg = [m for m in msgs if m['role'] == 'user'][0]
        context_part = user_msg['content'].split('\n\nContext:\n')[1]
        # Three chunks joined by double newline
        chunks = context_part.split('\n\n')
        self.assertEqual(len(chunks), 3)
        self.assertIn('Pizza section', chunks[0])
        self.assertIn('Pasta section', chunks[1])
        self.assertIn('Drinks', chunks[2])

    def test_rag_chunks_with_empty_content_skipped_in_export(self):
        """Chunks with empty content (failed fetches) are not included in export."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_empty_rag_exp', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Menu?')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Here it is!',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Good chunk content', 'vector_distance': 0.1},
                {'document_id': 'd2', 'chunk_id': 'c2', 'content': '', 'vector_distance': 0.2, 'fetch_error': 'timeout'},
            ]
        )

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msg = [m for m in msgs if m['role'] == 'user'][0]
        self.assertIn('Good chunk content', user_msg['content'])
        # Empty chunk should not produce extra content
        context_part = user_msg['content'].split('\n\nContext:\n')[1]
        self.assertEqual(context_part.strip(), 'Good chunk content')

    def test_all_chunks_empty_no_context_block(self):
        """If all chunks have empty content, no Context block is added."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_all_empty_rag', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Menu?')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Here it is!',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': '', 'vector_distance': 0.1, 'fetch_error': 'err'},
            ]
        )

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msg = [m for m in msgs if m['role'] == 'user'][0]
        self.assertEqual(user_msg['content'], 'Menu?')
        self.assertNotIn('Context:', user_msg['content'])

    def test_deleted_agent_turn_rag_not_injected(self):
        """RAG context from deleted agent turns is not injected."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_del_rag', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Menu?')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Here it is!', is_deleted=True,
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Should not appear', 'vector_distance': 0.1},
            ]
        )
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='Fallback response')

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msg = [m for m in msgs if m['role'] == 'user'][0]
        self.assertNotIn('Context:', user_msg['content'])
        self.assertNotIn('Should not appear', user_msg['content'])

    def test_deleted_user_turn_not_used_as_preceding(self):
        """Deleted user turns are not used as preceding user turn for RAG injection."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_del_user_rag', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='First question')
        Turn.objects.create(conversation=conv, position=1, role='user', original_text='Deleted question', is_deleted=True)
        Turn.objects.create(
            conversation=conv, position=2, role='agent',
            original_text='Response with RAG',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'RAG content', 'vector_distance': 0.1},
            ]
        )

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        # RAG context should be injected into position 0 user turn (the non-deleted one)
        user_msgs = [m for m in msgs if m['role'] == 'user']
        self.assertEqual(len(user_msgs), 1)
        self.assertIn('Context:', user_msgs[0]['content'])
        self.assertIn('RAG content', user_msgs[0]['content'])

    def test_agent_turn_without_preceding_user_rag_not_injected(self):
        """If agent turn with RAG has no preceding user turn, RAG is not injected."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_no_prev_user', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(
            conversation=conv, position=0, role='agent',
            original_text='Welcome!',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'greeting info', 'vector_distance': 0.1},
            ]
        )
        Turn.objects.create(conversation=conv, position=1, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='How can I help?')

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        # No user message should have "greeting info" injected since the RAG
        # agent turn at position 0 has no preceding user turn
        user_msgs = [m for m in msgs if m['role'] == 'user']
        for msg in user_msgs:
            self.assertNotIn('greeting info', msg['content'])

    def test_generate_jsonl_with_rag_context_flag(self):
        """generate_jsonl_examples passes include_rag_context through."""
        self._create_conversation_with_rag()
        examples_with = generate_jsonl_examples(include_rag_context=True)
        examples_without = generate_jsonl_examples(include_rag_context=False)

        self.assertEqual(len(examples_with), 1)
        self.assertEqual(len(examples_without), 1)

        # With RAG: user message should have Context
        with_msgs = examples_with[0]['messages']
        user_with = [m for m in with_msgs if m['role'] == 'user']
        has_context = any('Context:' in m['content'] for m in user_with)
        self.assertTrue(has_context)

        # Without RAG: no user message should have Context
        without_msgs = examples_without[0]['messages']
        user_without = [m for m in without_msgs if m['role'] == 'user']
        no_context = all('Context:' not in m['content'] for m in user_without)
        self.assertTrue(no_context)

    def test_generate_jsonl_rag_default_is_true(self):
        """By default, include_rag_context is True."""
        self._create_conversation_with_rag()
        examples = generate_jsonl_examples()
        msgs = examples[0]['messages']
        user_msgs = [m for m in msgs if m['role'] == 'user']
        has_context = any('Context:' in m['content'] for m in user_msgs)
        self.assertTrue(has_context)

    def test_rag_context_in_exported_jsonl(self):
        """Full JSONL export includes RAG context in parseable format."""
        self._create_conversation_with_rag()
        examples = generate_jsonl_examples(include_rag_context=True)
        jsonl = export_jsonl(examples)

        lines = jsonl.strip().split('\n')
        parsed = json.loads(lines[0])
        msgs = parsed['messages']

        user_msgs = [m for m in msgs if m['role'] == 'user']
        pizza_msg = user_msgs[0]
        self.assertIn('Context:', pizza_msg['content'])
        self.assertIn('Pepperoni Pizza', pizza_msg['content'])

    def test_validation_passes_with_rag_context(self):
        """Examples with RAG context injected still pass validation."""
        self._create_conversation_with_rag()
        examples = generate_jsonl_examples(include_rag_context=True)
        for ex in examples:
            errors = validate_example(ex)
            self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_rag_context_aggregated_from_consecutive_agent_turns(self):
        """If multiple consecutive agent turns have RAG, all are aggregated to preceding user."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_consec_agent_rag', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Tell me everything')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='First part...',
            rag_context=[{'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Chunk A', 'vector_distance': 0.1}]
        )
        Turn.objects.create(
            conversation=conv, position=2, role='agent',
            original_text='Second part...',
            rag_context=[{'document_id': 'd2', 'chunk_id': 'c2', 'content': 'Chunk B', 'vector_distance': 0.2}]
        )

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msg = [m for m in msgs if m['role'] == 'user'][0]
        self.assertIn('Chunk A', user_msg['content'])
        self.assertIn('Chunk B', user_msg['content'])

    def test_rag_context_with_tool_calls_turn(self):
        """RAG context on agent turns that also have tool calls works correctly."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_rag_tc', agent=self.agent, status='approved',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='I want to order')
        agent_turn = Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Placing your order...',
            rag_context=[{'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Menu items available', 'vector_distance': 0.1}]
        )
        ToolCall.objects.create(
            turn=agent_turn, tool_name='create_order',
            original_args={'customerName': 'Test', 'customerPhone': '555', 'items': []},
            status_code=200,
            response_body={'success': True},
        )
        Turn.objects.create(conversation=conv, position=2, role='agent', original_text='Order placed!')

        result = conversation_to_messages(conv, include_rag_context=True)
        msgs = result['messages']

        user_msg = [m for m in msgs if m['role'] == 'user'][0]
        self.assertIn('Menu items available', user_msg['content'])

        # Tool call should still be present
        tc_msgs = [m for m in msgs if 'tool_calls' in m]
        self.assertEqual(len(tc_msgs), 1)


class RagContextReviewViewTests(TestCase):
    """Tests for RAG context display in the admin review view."""

    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='admin_rag', password='admin', role='admin'
        )
        self.annotator = User.objects.create_user(
            username='annotator_rag', password='annotator', role='annotator'
        )
        self.agent = Agent.objects.create(
            agent_id='agent_rag_view', label='RAG View Agent', elevenlabs_api_key='key'
        )

    def test_review_shows_rag_turns_badge(self):
        """Edit summary bar shows RAG turns count."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_rag_review', agent=self.agent,
            assigned_to=self.annotator, status='completed',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Hello!',
            rag_context=[{'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Menu info', 'vector_distance': 0.1}]
        )
        Turn.objects.create(
            conversation=conv, position=2, role='agent',
            original_text='How can I help?',
            is_edited=True, edited_text='How may I help you?',
        )

        self.client.login(username='admin_rag', password='admin')
        response = self.client.get(f'/admin-panel/review/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '1 turn with RAG')

    def test_review_shows_rag_context_details(self):
        """Review page shows RAG context collapsible with chunk content."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_rag_review_det', agent=self.agent,
            assigned_to=self.annotator, status='completed',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Menu?')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Here you go!',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Large Pizza $14.99', 'vector_distance': 0.123},
            ]
        )

        self.client.login(username='admin_rag', password='admin')
        response = self.client.get(f'/admin-panel/review/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'RAG Context')
        self.assertContains(response, '1 chunk')
        self.assertContains(response, 'Large Pizza $14.99')

    def test_review_no_rag_badge_when_zero(self):
        """No RAG badge shown when no turns have RAG context."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_no_rag_review', agent=self.agent,
            assigned_to=self.annotator, status='completed',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello!')

        self.client.login(username='admin_rag', password='admin')
        response = self.client.get(f'/admin-panel/review/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'turn with RAG')
        self.assertNotContains(response, 'turns with RAG')

    def test_review_multiple_rag_turns_badge_pluralized(self):
        """Badge shows correct plural form for multiple RAG turns."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_multi_rag_review', agent=self.agent,
            assigned_to=self.annotator, status='completed',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='First', rag_context=[{'document_id': 'd', 'chunk_id': 'c', 'content': 'x', 'vector_distance': 0.1}]
        )
        Turn.objects.create(conversation=conv, position=2, role='user', original_text='More')
        Turn.objects.create(
            conversation=conv, position=3, role='agent',
            original_text='Second', rag_context=[{'document_id': 'd', 'chunk_id': 'c2', 'content': 'y', 'vector_distance': 0.2}]
        )

        self.client.login(username='admin_rag', password='admin')
        response = self.client.get(f'/admin-panel/review/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '2 turns with RAG')

    def test_rag_turns_count_not_included_in_total_changes(self):
        """rag_turns count doesn't inflate the total_changes count.
        When only RAG is present (no edits), the bar shows 'Info:' not 'Edit Summary:'."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_rag_no_change', agent=self.agent,
            assigned_to=self.annotator, status='completed',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Hello!',
            rag_context=[{'document_id': 'd', 'chunk_id': 'c', 'content': 'ctx', 'vector_distance': 0.1}]
        )

        self.client.login(username='admin_rag', password='admin')
        response = self.client.get(f'/admin-panel/review/{conv.pk}/')
        # No edits/deletes/inserts means total_changes = 0, but RAG bar still shows
        self.assertNotContains(response, 'Edit Summary:')
        self.assertContains(response, 'Info:')
        self.assertContains(response, '1 turn with RAG')


class RagContextExportViewTests(TestCase):
    """Tests for the export page RAG context checkbox and download."""

    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='admin_rag_exp', password='admin', role='admin'
        )
        self.agent = Agent.objects.create(
            agent_id='agent_rag_exp_v', label='RAG Export View Agent', elevenlabs_api_key='key'
        )

    def test_export_page_shows_rag_checkbox(self):
        """Export page contains the 'Include RAG context' checkbox."""
        self.client.login(username='admin_rag_exp', password='admin')
        response = self.client.get('/admin-panel/export/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'include_rag_context')
        self.assertContains(response, 'Include RAG context in user messages')

    def test_export_download_with_rag_enabled(self):
        """Export with include_rag_context includes RAG context in output."""
        SystemPrompt.objects.create(name='P', content='Prompt', is_active=True)
        for i in range(10):
            conv = Conversation.objects.create(
                elevenlabs_id=f'conv_rag_dl_{i}', agent=self.agent,
                status='approved', call_timestamp=timezone.now()
            )
            Turn.objects.create(conversation=conv, position=0, role='user', original_text=f'Question {i}')
            Turn.objects.create(
                conversation=conv, position=1, role='agent',
                original_text=f'Answer {i}',
                rag_context=[{'document_id': 'd', 'chunk_id': f'c{i}', 'content': f'Context chunk {i}', 'vector_distance': 0.1}]
            )

        self.client.login(username='admin_rag_exp', password='admin')
        response = self.client.get(
            '/admin-panel/export/download/?include_system_prompt&include_tools&include_rag_context'
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        first_line = json.loads(content.strip().split('\n')[0])
        user_msgs = [m for m in first_line['messages'] if m['role'] == 'user']
        has_context = any('Context:' in m['content'] for m in user_msgs)
        self.assertTrue(has_context)

    def test_export_download_without_rag(self):
        """Export without include_rag_context omits RAG context from output."""
        SystemPrompt.objects.create(name='P', content='Prompt', is_active=True)
        for i in range(10):
            conv = Conversation.objects.create(
                elevenlabs_id=f'conv_norag_dl_{i}', agent=self.agent,
                status='approved', call_timestamp=timezone.now()
            )
            Turn.objects.create(conversation=conv, position=0, role='user', original_text=f'Q {i}')
            Turn.objects.create(
                conversation=conv, position=1, role='agent',
                original_text=f'A {i}',
                rag_context=[{'document_id': 'd', 'chunk_id': f'c{i}', 'content': f'Ctx {i}', 'vector_distance': 0.1}]
            )

        self.client.login(username='admin_rag_exp', password='admin')
        # Note: include_rag_context NOT in query params
        response = self.client.get(
            '/admin-panel/export/download/?include_system_prompt&include_tools'
        )
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        first_line = json.loads(content.strip().split('\n')[0])
        user_msgs = [m for m in first_line['messages'] if m['role'] == 'user']
        no_context = all('Context:' not in m['content'] for m in user_msgs)
        self.assertTrue(no_context)


class RagContextEditorViewTests(TestCase):
    """Tests for RAG context display in the annotator editor view."""

    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='admin_rag_ed', password='admin', role='admin'
        )
        self.annotator = User.objects.create_user(
            username='annotator_rag_ed', password='annotator', role='annotator'
        )
        self.agent = Agent.objects.create(
            agent_id='agent_rag_editor', label='RAG Editor Agent', elevenlabs_api_key='key'
        )

    def test_editor_shows_rag_context_on_turn(self):
        """Editor turn display shows RAG context collapsible."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_rag_editor', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Menu?')
        Turn.objects.create(
            conversation=conv, position=1, role='agent',
            original_text='Here it is!',
            rag_context=[
                {'document_id': 'd1', 'chunk_id': 'c1', 'content': 'Pizza Menu: Pepperoni $14.99', 'vector_distance': 0.15},
                {'document_id': 'd1', 'chunk_id': 'c2', 'content': 'Pasta Menu: Spaghetti $11.99', 'vector_distance': 0.25},
            ]
        )

        self.client.login(username='annotator_rag_ed', password='annotator')
        response = self.client.get(f'/conversations/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'RAG Context')
        self.assertContains(response, '2 chunks')

    def test_editor_no_rag_no_section(self):
        """Turns without RAG context don't show RAG section."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_no_rag_editor', agent=self.agent,
            assigned_to=self.annotator, status='in_progress',
            call_timestamp=timezone.now()
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello!')

        self.client.login(username='annotator_rag_ed', password='annotator')
        response = self.client.get(f'/conversations/{conv.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'RAG Context')


class RagContextBackfillTests(TestCase):
    """Tests for the backfill_rag_context management command."""

    def setUp(self):
        self.agent = Agent.objects.create(
            agent_id='agent_backfill', label='Backfill Agent', elevenlabs_api_key='test-key'
        )

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_fetches_chunks_from_raw_data(self, MockClient):
        """Backfill command parses raw_data transcript and fetches chunk content."""
        mock_instance = MockClient.return_value
        mock_instance.get_kb_chunk.return_value = {'content': 'Fetched menu data'}

        conv = Conversation.objects.create(
            elevenlabs_id='conv_backfill_1', agent=self.agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi', 'time_in_call_secs': 1.0},
                    {
                        'role': 'agent', 'message': 'Hello!', 'time_in_call_secs': 2.0,
                        'rag_retrieval_info': {
                            'chunks': [
                                {'document_id': 'doc_bf', 'chunk_id': 'ch_bf', 'vector_distance': 0.2},
                            ]
                        },
                    },
                ],
            }
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello!')

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', stdout=out)

        turn = Turn.objects.get(conversation=conv, position=1)
        self.assertEqual(len(turn.rag_context), 1)
        self.assertEqual(turn.rag_context[0]['content'], 'Fetched menu data')
        self.assertEqual(turn.rag_context[0]['document_id'], 'doc_bf')
        mock_instance.get_kb_chunk.assert_called_once_with('doc_bf', 'ch_bf')
        self.assertIn('Updated: 1', out.getvalue())

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_skips_already_filled(self, MockClient):
        """Backfill skips turns that already have rag_context."""
        mock_instance = MockClient.return_value

        conv = Conversation.objects.create(
            elevenlabs_id='conv_backfill_skip', agent=self.agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi'},
                    {
                        'role': 'agent', 'message': 'Hello!',
                        'rag_retrieval_info': {
                            'chunks': [{'document_id': 'doc', 'chunk_id': 'ch', 'vector_distance': 0.1}]
                        },
                    },
                ],
            }
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(
            conversation=conv, position=1, role='agent', original_text='Hello!',
            rag_context=[{'document_id': 'doc', 'chunk_id': 'ch', 'content': 'existing', 'vector_distance': 0.1}]
        )

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', stdout=out)

        self.assertIn('Skipped (already filled): 1', out.getvalue())
        mock_instance.get_kb_chunk.assert_not_called()

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_dry_run(self, MockClient):
        """Dry run shows what would be updated without making changes."""
        mock_instance = MockClient.return_value
        mock_instance.get_kb_chunk.return_value = {'content': 'data'}

        conv = Conversation.objects.create(
            elevenlabs_id='conv_backfill_dry', agent=self.agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi'},
                    {
                        'role': 'agent', 'message': 'Hello!',
                        'rag_retrieval_info': {
                            'chunks': [{'document_id': 'd', 'chunk_id': 'c', 'vector_distance': 0.1}]
                        },
                    },
                ],
            }
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello!')

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', '--dry-run', stdout=out)

        output = out.getvalue()
        self.assertIn('[DRY RUN]', output)
        # Turn should NOT be updated
        turn = Turn.objects.get(conversation=conv, position=1)
        self.assertEqual(turn.rag_context, [])

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_skips_conversations_without_raw_data(self, MockClient):
        """Conversations with empty raw_data are skipped."""
        Conversation.objects.create(
            elevenlabs_id='conv_no_raw', agent=self.agent, raw_data={}
        )

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', stdout=out)

        self.assertIn('Updated: 0', out.getvalue())
        MockClient.assert_not_called()

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_skips_transcripts_without_rag(self, MockClient):
        """Conversations without rag_retrieval_info in any turn are skipped."""
        conv = Conversation.objects.create(
            elevenlabs_id='conv_no_rag_raw', agent=self.agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi'},
                    {'role': 'agent', 'message': 'Hello!'},
                ],
            }
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello!')

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', stdout=out)

        self.assertIn('Updated: 0', out.getvalue())
        MockClient.assert_not_called()

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_handles_chunk_fetch_errors(self, MockClient):
        """Backfill handles API errors when fetching chunks."""
        mock_instance = MockClient.return_value
        mock_instance.get_kb_chunk.side_effect = Exception("API unavailable")

        conv = Conversation.objects.create(
            elevenlabs_id='conv_backfill_err', agent=self.agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi'},
                    {
                        'role': 'agent', 'message': 'Hello!',
                        'rag_retrieval_info': {
                            'chunks': [{'document_id': 'd', 'chunk_id': 'c', 'vector_distance': 0.1}]
                        },
                    },
                ],
            }
        )
        Turn.objects.create(conversation=conv, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv, position=1, role='agent', original_text='Hello!')

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', stdout=out)

        output = out.getvalue()
        self.assertIn('Chunk fetch errors: 1', output)
        # Turn should still be updated with error info
        turn = Turn.objects.get(conversation=conv, position=1)
        self.assertEqual(len(turn.rag_context), 1)
        self.assertEqual(turn.rag_context[0]['content'], '')
        self.assertIn('API unavailable', turn.rag_context[0]['fetch_error'])

    @patch('conversations.management.commands.backfill_rag_context.ElevenLabsClient')
    def test_backfill_agent_filter(self, MockClient):
        """--agent-id flag limits backfill to specific agent."""
        mock_instance = MockClient.return_value
        mock_instance.get_kb_chunk.return_value = {'content': 'data'}

        other_agent = Agent.objects.create(
            agent_id='other_bf', label='Other', elevenlabs_api_key='key2'
        )

        conv1 = Conversation.objects.create(
            elevenlabs_id='conv_bf_agent1', agent=self.agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi'},
                    {'role': 'agent', 'message': 'Hello!',
                     'rag_retrieval_info': {'chunks': [{'document_id': 'd', 'chunk_id': 'c', 'vector_distance': 0.1}]}},
                ],
            }
        )
        Turn.objects.create(conversation=conv1, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv1, position=1, role='agent', original_text='Hello!')

        conv2 = Conversation.objects.create(
            elevenlabs_id='conv_bf_agent2', agent=other_agent,
            raw_data={
                'transcript': [
                    {'role': 'user', 'message': 'Hi'},
                    {'role': 'agent', 'message': 'Hello!',
                     'rag_retrieval_info': {'chunks': [{'document_id': 'd2', 'chunk_id': 'c2', 'vector_distance': 0.1}]}},
                ],
            }
        )
        Turn.objects.create(conversation=conv2, position=0, role='user', original_text='Hi')
        Turn.objects.create(conversation=conv2, position=1, role='agent', original_text='Hello!')

        from django.core.management import call_command
        from io import StringIO
        out = StringIO()
        call_command('backfill_rag_context', '--agent-id', str(self.agent.pk), stdout=out)

        # Only self.agent's conversation should be updated
        turn1 = Turn.objects.get(conversation=conv1, position=1)
        turn2 = Turn.objects.get(conversation=conv2, position=1)
        self.assertEqual(len(turn1.rag_context), 1)
        self.assertEqual(turn2.rag_context, [])

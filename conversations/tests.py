import json
import requests
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.db import IntegrityError
from django.utils import timezone

from accounts.models import User
from conversations.models import Agent, Conversation, Turn, ToolCall, SystemPrompt, ExportLog
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
        mock_turn_user.tool_calls.all.return_value = []

        mock_turn_tc = MagicMock()
        mock_turn_tc.role = 'agent'
        mock_turn_tc.display_text = 'Checking...'
        mock_turn_tc.tool_calls.all.return_value = [mock_tc]

        mock_turn_final = MagicMock()
        mock_turn_final.role = 'agent'
        mock_turn_final.display_text = 'Here!'
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
        conv = _import_conversation(self.agent, 'conv_with_tc', data)
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
        conv = _import_conversation(self.agent, 'conv_req_body', data)
        tc = conv.turns.first().tool_calls.first()
        self.assertEqual(tc.original_args['date'], '2026-02-15')

    def test_import_conversation_no_transcript(self):
        data = {'transcript': [], 'metadata': {}, 'has_audio': False}
        conv = _import_conversation(self.agent, 'conv_no_transcript', data)
        self.assertEqual(conv.turns.count(), 0)

    def test_import_conversation_invalid_role(self):
        data = {
            'transcript': [{'role': 'system', 'message': 'internal'}],
            'metadata': {},
            'has_audio': False,
        }
        conv = _import_conversation(self.agent, 'conv_bad_role', data)
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
        conv = _import_conversation(self.agent, 'conv_bad_resp', data)
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
        with self.assertRaises(IntegrityError):
            self.client.post('/admin-panel/team/invite/', {
                'username': 'annotator', 'password': 'pass',
            })

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

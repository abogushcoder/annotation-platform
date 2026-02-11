import json
from django.test import TestCase, Client
from django.utils import timezone

from accounts.models import User
from conversations.models import Agent, Conversation, Turn, ToolCall, SystemPrompt
from conversations.services.export import (
    conversation_to_messages, validate_example, generate_jsonl_examples,
    export_jsonl, split_train_validation, count_tokens
)


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

    def test_conversation_creation(self):
        conv = Conversation.objects.create(
            elevenlabs_id='conv_001', agent=self.agent, status='unassigned'
        )
        self.assertEqual(conv.status, 'unassigned')
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

    def test_system_prompt_only_one_active(self):
        p1 = SystemPrompt.objects.create(name='v1', content='prompt 1', is_active=True)
        p2 = SystemPrompt.objects.create(name='v2', content='prompt 2', is_active=True)
        p1.refresh_from_db()
        self.assertFalse(p1.is_active)
        self.assertTrue(p2.is_active)


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
        t3 = Turn.objects.create(
            conversation=self.conv, position=3, role='agent',
            original_text='Order placed!'
        )

    def test_conversation_to_messages(self):
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        # Should have system + turns
        self.assertEqual(msgs[0]['role'], 'system')
        self.assertEqual(msgs[0]['content'], 'You are a test assistant.')
        # Should have user and assistant messages
        roles = [m['role'] for m in msgs]
        self.assertIn('user', roles)
        self.assertIn('assistant', roles)

    def test_tool_call_format(self):
        result = conversation_to_messages(self.conv)
        msgs = result['messages']
        # Find tool call message
        tc_msgs = [m for m in msgs if 'tool_calls' in m]
        self.assertEqual(len(tc_msgs), 1)
        tc = tc_msgs[0]['tool_calls'][0]
        self.assertEqual(tc['function']['name'], 'create_order')
        # Arguments should be JSON string
        args = json.loads(tc['function']['arguments'])
        self.assertEqual(args['customerName'], 'Test')
        # Should have tool response
        tool_msgs = [m for m in msgs if m['role'] == 'tool']
        self.assertEqual(len(tool_msgs), 1)

    def test_validate_valid_example(self):
        result = conversation_to_messages(self.conv)
        errors = validate_example(result)
        self.assertEqual(errors, [])

    def test_validate_invalid_example(self):
        errors = validate_example({'messages': []})
        self.assertIn('No messages', errors)

    def test_generate_jsonl(self):
        examples = generate_jsonl_examples()
        self.assertEqual(len(examples), 1)

    def test_export_jsonl_format(self):
        examples = generate_jsonl_examples()
        jsonl = export_jsonl(examples)
        lines = jsonl.strip().split('\n')
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertIn('messages', parsed)
        self.assertIn('tools', parsed)

    def test_split_train_validation(self):
        # Need multiple examples
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

    def test_count_tokens(self):
        examples = generate_jsonl_examples()
        token_count = count_tokens(examples)
        self.assertGreater(token_count, 0)

    def test_tools_included(self):
        result = conversation_to_messages(self.conv, include_tools=True)
        self.assertIn('tools', result)
        tool_names = [t['function']['name'] for t in result['tools']]
        self.assertIn('create_order', tool_names)

    def test_no_tools_when_disabled(self):
        result = conversation_to_messages(self.conv, include_tools=False)
        self.assertNotIn('tools', result)


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

    def test_annotator_list(self):
        self.client.login(username='annotator', password='annotator')
        response = self.client.get('/conversations/')
        self.assertEqual(response.status_code, 200)

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
        response = self.client.get('/admin-panel/')
        self.assertEqual(response.status_code, 403)

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
        # Status should auto-transition to in_progress
        conv.refresh_from_db()
        self.assertEqual(conv.status, 'in_progress')

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
        # GET returns edit form
        response = self.client.get(f'/conversations/{conv.pk}/turn/{turn.pk}/edit/')
        self.assertEqual(response.status_code, 200)
        # POST saves edit
        response = self.client.post(
            f'/conversations/{conv.pk}/turn/{turn.pk}/edit/',
            {'edited_text': 'I want a pizza'}
        )
        self.assertEqual(response.status_code, 200)
        turn.refresh_from_db()
        self.assertTrue(turn.is_edited)
        self.assertEqual(turn.edited_text, 'I want a pizza')

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

    def test_team_invite(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/team/invite/', {
            'username': 'newuser', 'password': 'newpass123',
            'first_name': 'New', 'last_name': 'User',
        })
        self.assertEqual(response.status_code, 302)
        new_user = User.objects.get(username='newuser')
        self.assertEqual(new_user.role, 'annotator')

    def test_analytics_page(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/analytics/')
        self.assertEqual(response.status_code, 200)

    def test_prompt_create(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/prompts/add/', {
            'name': 'Test Prompt', 'content': 'Hello world', 'is_active': 'on',
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(SystemPrompt.objects.filter(name='Test Prompt').exists())

    def test_agent_add(self):
        self.client.login(username='admin', password='admin')
        response = self.client.post('/admin-panel/agents/add/', {
            'agent_id': 'new_agent', 'label': 'New Agent', 'api_key': 'key123',
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Agent.objects.filter(agent_id='new_agent').exists())

    def test_export_preview(self):
        self.client.login(username='admin', password='admin')
        response = self.client.get('/admin-panel/export/preview/')
        self.assertEqual(response.status_code, 200)

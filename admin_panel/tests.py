from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone

from accounts.models import User
from conversations.models import Agent, Conversation, Turn, ToolCall


class ResetConversationsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(
            username='admin', password='testpass', role='admin',
        )
        self.annotator = User.objects.create_user(
            username='annotator', password='testpass', role='annotator',
        )
        self.agent = Agent.objects.create(
            agent_id='agent_123',
            label='Test Agent',
            elevenlabs_api_key='fake-key',
            last_synced_at=timezone.now(),
        )
        # Create 2 conversations with turns and tool calls
        for i in range(2):
            conv = Conversation.objects.create(
                elevenlabs_id=f'conv_{i}',
                agent=self.agent,
                status='unassigned',
            )
            turn = Turn.objects.create(
                conversation=conv, position=0, role='user',
                original_text='Hello',
            )
            ToolCall.objects.create(
                turn=turn, tool_name='create_order',
                original_args={'item': 'pizza'},
            )

    def test_reset_requires_post(self):
        self.client.login(username='admin', password='testpass')
        resp = self.client.get(reverse('reset_conversations'))
        # GET should redirect without deleting
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Conversation.objects.count(), 2)

    def test_reset_requires_admin(self):
        self.client.login(username='annotator', password='testpass')
        resp = self.client.post(reverse('reset_conversations'))
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(Conversation.objects.count(), 2)

    def test_reset_deletes_conversations_and_cascades(self):
        self.client.login(username='admin', password='testpass')
        self.assertEqual(Conversation.objects.count(), 2)
        self.assertEqual(Turn.objects.count(), 2)
        self.assertEqual(ToolCall.objects.count(), 2)

        resp = self.client.post(reverse('reset_conversations'))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Conversation.objects.count(), 0)
        self.assertEqual(Turn.objects.count(), 0)
        self.assertEqual(ToolCall.objects.count(), 0)

    def test_reset_clears_agent_last_synced(self):
        self.client.login(username='admin', password='testpass')
        self.assertIsNotNone(self.agent.last_synced_at)

        self.client.post(reverse('reset_conversations'))
        self.agent.refresh_from_db()
        self.assertIsNone(self.agent.last_synced_at)

    def test_reset_preserves_agents_and_users(self):
        self.client.login(username='admin', password='testpass')
        self.client.post(reverse('reset_conversations'))
        self.assertEqual(Agent.objects.count(), 1)
        self.assertEqual(User.objects.count(), 2)

"""
Seed realistic test conversations covering all RAG context scenarios.

Creates 10 conversations under a test agent with various RAG configurations
for end-to-end testing of RAG context display and export.

Idempotent: checks for existing data by elevenlabs_id prefix 'rag_test_'.
"""
import json
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from accounts.models import User
from conversations.models import Agent, Conversation, SystemPrompt, Tag, ToolCall, Turn


# ---------------------------------------------------------------------------
# Realistic RAG chunk content (restaurant menu data)
# ---------------------------------------------------------------------------
PIZZA_MENU_CHUNK = (
    "PIZZA MENU\n"
    "Large Pepperoni Pizza - $14.99\n"
    "Medium Pepperoni Pizza - $11.99\n"
    "Large Margherita Pizza - $12.99\n"
    "Medium Margherita Pizza - $9.99\n"
    "Large BBQ Chicken Pizza - $15.99\n"
    "Medium BBQ Chicken Pizza - $13.99\n"
    "Large Hawaiian Pizza - $14.99\n"
    "Add Extra Cheese - $2.00\n"
    "Add Jalape\u00f1os - $1.50"
)

HOURS_LOCATION_CHUNK = (
    "HOURS & LOCATION\n"
    "Open Monday-Saturday 11am-10pm\n"
    "Sunday 12pm-9pm\n"
    "Address: 742 Evergreen Terrace, Springfield\n"
    "Phone: (555) 012-3456\n"
    "Free delivery within 3 miles"
)

PASTA_MENU_CHUNK = (
    "PASTA MENU\n"
    "Spaghetti Bolognese - $11.99\n"
    "Fettuccine Alfredo - $13.99\n"
    "Penne Arrabiata - $10.99\n"
    "Baked Ziti - $12.99\n"
    "Add Garlic Bread - $3.99"
)

SPECIALS_CHUNK = (
    "TODAY'S SPECIALS\n"
    "Tuesday Deal: Buy 1 Large Pizza, Get Medium 50% Off\n"
    "Family Combo: 2 Large Pizzas + Garlic Bread + 2L Soda - $34.99\n"
    "Kids Eat Free on Sundays (12 and under, with adult entree)"
)

SYSTEM_PROMPT_TEXT = (
    "You are Tony, a friendly AI assistant for Tony's Pizzeria. "
    "You help customers place pickup orders and make reservations. "
    "You are warm, helpful, and knowledgeable about the menu. "
    "Always confirm the order details before placing it. "
    "If an item is unavailable, suggest a similar alternative. "
    "For reservations, always check availability before confirming."
)


def _chunk(doc_id, chunk_id, content, distance=0.15, fetch_error=None):
    """Build a RAG chunk dict."""
    c = {
        'document_id': doc_id,
        'chunk_id': chunk_id,
        'content': content,
        'vector_distance': distance,
    }
    if fetch_error:
        c['fetch_error'] = fetch_error
    return c


# ---------------------------------------------------------------------------
# Conversation definitions
# ---------------------------------------------------------------------------

def _conv_basic():
    """#1: Basic single-chunk RAG."""
    return {
        'elevenlabs_id': 'rag_test_basic',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Thank you for calling Tony's Pizzeria! How can I help you today?"},
            {'role': 'user', 'text': "Hi, what kind of pizzas do you have?"},
            {'role': 'agent', 'text': "We have a great selection! Our large pepperoni is $14.99, margherita is $12.99, BBQ chicken is $15.99, and Hawaiian is $14.99. We also have medium sizes available. Would you like to order something?",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.12)]},
            {'role': 'user', 'text': "I'll take a large pepperoni please."},
            {'role': 'agent', 'text': "Great choice! One large pepperoni pizza for $14.99. Can I get your name and phone number?"},
        ],
    }


def _conv_multi_chunk():
    """#2: Multiple chunks joined."""
    return {
        'elevenlabs_id': 'rag_test_multi_chunk',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria, how can I help?"},
            {'role': 'user', 'text': "What's on your menu and when are you open?"},
            {'role': 'agent', 'text': "We've got pizzas starting at $9.99 and pastas from $10.99. We're open Monday through Saturday 11am to 10pm, and Sunday noon to 9pm. Want to hear about our specials?",
             'rag_context': [
                 _chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.10),
                 _chunk('doc_info_001', 'chunk_hours_01', HOURS_LOCATION_CHUNK, 0.18),
                 _chunk('doc_menu_001', 'chunk_pasta_01', PASTA_MENU_CHUNK, 0.22),
             ]},
            {'role': 'user', 'text': "Sure, what are the specials?"},
            {'role': 'agent', 'text': "Today we have buy one large pizza get a medium 50% off! We also have a family combo for $34.99.",
             'rag_context': [_chunk('doc_specials_001', 'chunk_specials_01', SPECIALS_CHUNK, 0.08)]},
        ],
    }


def _conv_with_tools():
    """#3: RAG coexisting with tool calls."""
    return {
        'elevenlabs_id': 'rag_test_with_tools',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria, what can I get for you?"},
            {'role': 'user', 'text': "I'd like to order a large margherita and a spaghetti bolognese for pickup."},
            {'role': 'agent', 'text': "Let me put that order in for you. One large margherita pizza at $12.99 and spaghetti bolognese at $11.99. Can I get your name and number?",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.14)]},
            {'role': 'user', 'text': "John Smith, 555-0199."},
            {'role': 'agent', 'text': "Your order has been placed! Your total is $24.98 before tax. It'll be ready in about 20 minutes.",
             'tool_calls': [{
                 'tool_name': 'create_order',
                 'args': {
                     'customerName': 'John Smith',
                     'customerPhone': '555-0199',
                     'items': [
                         {'itemName': 'Large Margherita Pizza', 'quantity': 1, 'modifiers': []},
                         {'itemName': 'Spaghetti Bolognese', 'quantity': 1, 'modifiers': []},
                     ],
                 },
                 'response': {'orderId': 'ORD-1234', 'estimatedTime': '20 min', 'total': 24.98},
                 'status_code': 200,
             }]},
        ],
    }


def _conv_failed_fetch():
    """#4: Graceful degradation with empty content + fetch_error."""
    return {
        'elevenlabs_id': 'rag_test_failed_fetch',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria, how can I help you today?"},
            {'role': 'user', 'text': "Do you have any specials going on?"},
            {'role': 'agent', 'text': "I'm sorry, I'm having a little trouble pulling up our specials right now. But I can tell you about our regular menu! We have great pizzas and pasta dishes. Would you like to hear about those?",
             'rag_context': [_chunk('doc_specials_001', 'chunk_specials_01', '', 0.15,
                                    fetch_error='HTTPError: 404 - chunk not found')]},
            {'role': 'user', 'text': "Sure, tell me about the pizzas."},
            {'role': 'agent', 'text': "We have pepperoni, margherita, BBQ chicken, and Hawaiian pizzas in medium and large sizes, starting at $9.99.",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.11)]},
        ],
    }


def _conv_partial_fail():
    """#5: Mixed success/failure - 2 chunks, 1 OK, 1 failed."""
    return {
        'elevenlabs_id': 'rag_test_partial_fail',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Welcome to Tony's! How can I help?"},
            {'role': 'user', 'text': "What pasta dishes do you have, and can I get delivery?"},
            {'role': 'agent', 'text': "We have spaghetti bolognese, fettuccine alfredo, penne arrabiata, and baked ziti. And yes, we offer free delivery within 3 miles!",
             'rag_context': [
                 _chunk('doc_menu_001', 'chunk_pasta_01', PASTA_MENU_CHUNK, 0.13),
                 _chunk('doc_info_001', 'chunk_delivery_01', '', 0.20,
                        fetch_error='ConnectionTimeout: request timed out'),
             ]},
            {'role': 'user', 'text': "Great, I'll have the fettuccine alfredo."},
            {'role': 'agent', 'text': "One fettuccine alfredo for $13.99. Name and phone number please?"},
        ],
    }


def _conv_no_rag():
    """#6: Control - no RAG at all."""
    return {
        'elevenlabs_id': 'rag_test_no_rag',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria, this is Tony speaking!"},
            {'role': 'user', 'text': "Hey Tony, I placed an order about 30 minutes ago. Order number ORD-5678."},
            {'role': 'agent', 'text': "Let me check on that for you. Yes, I see your order. It should be ready in about 5 more minutes!"},
            {'role': 'user', 'text': "Perfect, thanks!"},
            {'role': 'agent', 'text': "You're welcome! See you soon."},
        ],
    }


def _conv_multi_turn():
    """#7: RAG on 2 different agent turns (non-consecutive)."""
    return {
        'elevenlabs_id': 'rag_test_multi_turn',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria, how can I help you today?"},
            {'role': 'user', 'text': "What pizzas do you have?"},
            {'role': 'agent', 'text': "We have pepperoni, margherita, BBQ chicken, and Hawaiian. Large and medium sizes available!",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.10)]},
            {'role': 'user', 'text': "Nice. And what are your hours?"},
            {'role': 'agent', 'text': "We're not too far - 742 Evergreen Terrace. We're open every day!"},
            {'role': 'user', 'text': "What are the exact hours though?"},
            {'role': 'agent', 'text': "Monday through Saturday we're open 11am to 10pm, and Sunday noon to 9pm.",
             'rag_context': [_chunk('doc_info_001', 'chunk_hours_01', HOURS_LOCATION_CHUNK, 0.14)]},
            {'role': 'user', 'text': "Great, thanks!"},
            {'role': 'agent', 'text': "You're welcome! Call back anytime to order."},
        ],
    }


def _conv_consecutive():
    """#8: 2 consecutive agent turns with RAG (aggregated to single preceding user)."""
    return {
        'elevenlabs_id': 'rag_test_consecutive',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria!"},
            {'role': 'user', 'text': "Tell me everything about your menu."},
            # Two consecutive agent turns with RAG - both should aggregate into the preceding user message
            {'role': 'agent', 'text': "For pizzas, we have pepperoni, margherita, BBQ chicken, and Hawaiian, starting at $9.99 for a medium.",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.09)]},
            {'role': 'agent', 'text': "And for pasta, we have spaghetti bolognese, fettuccine alfredo, penne arrabiata, and baked ziti. Plus today's specials!",
             'rag_context': [
                 _chunk('doc_menu_001', 'chunk_pasta_01', PASTA_MENU_CHUNK, 0.12),
                 _chunk('doc_specials_001', 'chunk_specials_01', SPECIALS_CHUNK, 0.16),
             ]},
            {'role': 'user', 'text': "Wow, that's a lot of options!"},
            {'role': 'agent', 'text': "Sure is! Let me know what catches your eye."},
        ],
    }


def _conv_deleted_turn():
    """#9: RAG on a deleted agent turn (should be excluded from export)."""
    return {
        'elevenlabs_id': 'rag_test_deleted_turn',
        'status': 'approved',
        'turns': [
            {'role': 'agent', 'text': "Tony's Pizzeria, how can I help?"},
            {'role': 'user', 'text': "What pasta dishes do you offer?"},
            # This turn will be soft-deleted
            {'role': 'agent', 'text': "[garbled audio - agent repeated itself]",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pasta_01', PASTA_MENU_CHUNK, 0.11)],
             'is_deleted': True},
            {'role': 'agent', 'text': "We have spaghetti bolognese, fettuccine alfredo, penne arrabiata, and baked ziti!",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pasta_01', PASTA_MENU_CHUNK, 0.11)]},
            {'role': 'user', 'text': "I'll have the baked ziti."},
            {'role': 'agent', 'text': "One baked ziti, $12.99. Name and number?"},
        ],
    }


def _conv_for_review():
    """#10: For the review queue - completed status, edited turns."""
    return {
        'elevenlabs_id': 'rag_test_for_review',
        'status': 'completed',
        'turns': [
            {'role': 'agent', 'text': "Thank you for calling Tony's Pizzeria! What can I get for you?"},
            {'role': 'user', 'text': "What kinds of pizza do you have?"},
            {'role': 'agent', 'text': "We have pepperoni, margherita, BBQ chicken, and Hawaiian pizzas. Our large pepperoni is $14.99 and it's our most popular!",
             'rag_context': [_chunk('doc_menu_001', 'chunk_pizza_01', PIZZA_MENU_CHUNK, 0.10)],
             'edited_text': "We have pepperoni, margherita, BBQ chicken, and Hawaiian pizzas. Our large pepperoni is $14.99 -- it's our most popular!"},
            {'role': 'user', 'text': "Sounds good, I'll take two large pepperonis."},
            {'role': 'agent', 'text': "Two large pepperoni pizzas, that'll be $29.98 before tax. Can I get your name?"},
        ],
    }


ALL_CONVERSATIONS = [
    _conv_basic,
    _conv_multi_chunk,
    _conv_with_tools,
    _conv_failed_fetch,
    _conv_partial_fail,
    _conv_no_rag,
    _conv_multi_turn,
    _conv_consecutive,
    _conv_deleted_turn,
    _conv_for_review,
]


class Command(BaseCommand):
    help = 'Seed test conversations covering all RAG context scenarios'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force', action='store_true',
            help='Delete existing rag_test_* data and recreate',
        )

    def handle(self, *args, **options):
        force = options['force']

        # Check for existing data
        existing = Conversation.objects.filter(elevenlabs_id__startswith='rag_test_').count()
        if existing and not force:
            self.stdout.write(self.style.WARNING(
                f"Found {existing} existing rag_test_* conversations. "
                f"Use --force to delete and recreate."
            ))
            return

        if force and existing:
            self.stdout.write(f"Deleting {existing} existing rag_test_* conversations...")
            Conversation.objects.filter(elevenlabs_id__startswith='rag_test_').delete()

        # --- Create users ---
        admin_user, created = User.objects.get_or_create(
            username='admin',
            defaults={'role': 'admin', 'is_staff': True, 'is_superuser': True},
        )
        if created:
            admin_user.set_password('admin')
            admin_user.save()
            self.stdout.write(self.style.SUCCESS("Created admin user (admin/admin)"))
        else:
            self.stdout.write("Admin user already exists")

        annotator_user, created = User.objects.get_or_create(
            username='annotator',
            defaults={'role': 'annotator'},
        )
        if created:
            annotator_user.set_password('annotator')
            annotator_user.save()
            self.stdout.write(self.style.SUCCESS("Created annotator user (annotator/annotator)"))
        else:
            self.stdout.write("Annotator user already exists")

        # --- Create agent ---
        agent, created = Agent.objects.get_or_create(
            agent_id='rag_test_agent',
            defaults={
                'label': "Tony's Pizzeria (RAG Test)",
                'elevenlabs_api_key': 'test_key_not_real',
            },
        )
        if created:
            self.stdout.write(self.style.SUCCESS("Created test agent: Tony's Pizzeria (RAG Test)"))
        else:
            self.stdout.write("Test agent already exists")

        # --- Create system prompt ---
        prompt, created = SystemPrompt.objects.get_or_create(
            name='RAG Test - Tony Pizzeria',
            defaults={
                'content': SYSTEM_PROMPT_TEXT,
                'is_active': True,
            },
        )
        if created:
            self.stdout.write(self.style.SUCCESS("Created active system prompt"))
        else:
            # Ensure it's active
            if not prompt.is_active:
                prompt.is_active = True
                prompt.save()
            self.stdout.write("System prompt already exists (ensured active)")

        # --- Create tag ---
        rag_tag, _ = Tag.objects.get_or_create(
            name='rag-test',
            defaults={'color': '#8b5cf6'},
        )

        # --- Create conversations ---
        now = timezone.now()
        for i, conv_fn in enumerate(ALL_CONVERSATIONS):
            conv_def = conv_fn()
            eid = conv_def['elevenlabs_id']

            call_ts = now - timedelta(hours=len(ALL_CONVERSATIONS) - i)

            conv = Conversation.objects.create(
                elevenlabs_id=eid,
                agent=agent,
                assigned_to=annotator_user,
                status=conv_def['status'],
                call_duration_secs=45 + i * 10,
                call_timestamp=call_ts,
                completed_at=now if conv_def['status'] in ('completed', 'approved') else None,
                reviewed_at=now if conv_def['status'] == 'approved' else None,
            )
            conv.tags.add(rag_tag)

            # Create turns
            for pos, turn_def in enumerate(conv_def['turns']):
                edited_text = turn_def.get('edited_text', '')
                is_edited = bool(edited_text)
                is_deleted = turn_def.get('is_deleted', False)

                turn = Turn.objects.create(
                    conversation=conv,
                    position=pos,
                    role=turn_def['role'],
                    original_text=turn_def['text'],
                    edited_text=edited_text,
                    is_edited=is_edited,
                    is_deleted=is_deleted,
                    time_in_call_secs=pos * 5.0,
                    rag_context=turn_def.get('rag_context', []),
                )

                # Create tool calls if present
                for tc_def in turn_def.get('tool_calls', []):
                    ToolCall.objects.create(
                        turn=turn,
                        tool_name=tc_def['tool_name'],
                        original_args=tc_def['args'],
                        status_code=tc_def.get('status_code', 200),
                        response_body=tc_def.get('response', {'status': 'ok'}),
                    )

            self.stdout.write(f"  Created conversation: {eid} ({conv_def['status']})")

        self.stdout.write(self.style.SUCCESS(
            f"\nDone! Created {len(ALL_CONVERSATIONS)} test conversations.\n"
            f"  Admin login:     admin / admin\n"
            f"  Annotator login: annotator / annotator\n"
            f"  All tagged with: rag-test"
        ))

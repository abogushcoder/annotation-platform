#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from conversations.models import Conversation, ToolCall

conv = Conversation.objects.get(pk=10)
tool_calls = ToolCall.objects.filter(turn__conversation=conv)

print(f'Tool calls count: {tool_calls.count()}')

for tc in tool_calls[:5]:
    print(f'\nTC: {tc.tool_name}')
    print(f'display_args type: {type(tc.display_args)}')
    print(f'display_args: {tc.display_args}')
    try:
        items = list(tc.display_args.items())
        print(f'Items count: {len(items)}')
        for i, item in enumerate(items[:3]):
            print(f'  Item {i}: type={type(item)}, len={len(item)}, value={item}')
    except Exception as e:
        print(f'Error iterating items: {e}')

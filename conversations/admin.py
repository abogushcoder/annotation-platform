from django.contrib import admin
from .models import Agent, Conversation, Turn, ToolCall, SystemPrompt


@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = ('label', 'agent_id', 'last_synced_at')


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('elevenlabs_id', 'agent', 'status', 'assigned_to', 'call_timestamp')
    list_filter = ('status', 'agent')


@admin.register(Turn)
class TurnAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'position', 'role', 'is_edited')


@admin.register(ToolCall)
class ToolCallAdmin(admin.ModelAdmin):
    list_display = ('tool_name', 'turn', 'is_edited')


@admin.register(SystemPrompt)
class SystemPromptAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'created_at')

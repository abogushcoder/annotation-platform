from django.conf import settings
from django.db import models


class Agent(models.Model):
    agent_id = models.CharField(max_length=100, unique=True, help_text="ElevenLabs agent ID")
    label = models.CharField(max_length=200, help_text="Friendly name, e.g. 'Tony's Pizzeria'")
    elevenlabs_api_key = models.CharField(max_length=200, help_text="API key for this agent's workspace")
    last_synced_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.label


class Conversation(models.Model):
    class Status(models.TextChoices):
        UNASSIGNED = 'unassigned', 'Unassigned'
        ASSIGNED = 'assigned', 'Assigned'
        IN_PROGRESS = 'in_progress', 'In Progress'
        COMPLETED = 'completed', 'Completed'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'
        FLAGGED = 'flagged', 'Flagged'

    elevenlabs_id = models.CharField(max_length=100, unique=True, help_text="ElevenLabs conversation ID")
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='conversations')
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='assigned_conversations'
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.UNASSIGNED)
    call_duration_secs = models.IntegerField(null=True, blank=True)
    call_timestamp = models.DateTimeField(null=True, blank=True, help_text="When the call started")
    has_audio = models.BooleanField(default=False)
    audio_s3_key = models.CharField(max_length=500, blank=True, default='')
    user_audio_s3_key = models.CharField(max_length=500, blank=True, default='')
    agent_audio_s3_key = models.CharField(max_length=500, blank=True, default='')
    raw_data = models.JSONField(default=dict, help_text="Full raw response from ElevenLabs API")
    tags = models.ManyToManyField('Tag', blank=True, related_name='conversations')
    annotator_notes = models.TextField(blank=True, default='')
    reviewer_notes = models.TextField(blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-call_timestamp']

    def __str__(self):
        return f"{self.elevenlabs_id} ({self.get_status_display()})"


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    color = models.CharField(max_length=7, default='#6366f1')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Turn(models.Model):
    class Role(models.TextChoices):
        USER = 'user', 'Customer'
        AGENT = 'agent', 'Restaurant AI'

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='turns')
    position = models.IntegerField(help_text="Order within conversation, 0-indexed")
    role = models.CharField(max_length=10, choices=Role.choices)
    original_text = models.TextField(help_text="Original transcript text from ElevenLabs")
    edited_text = models.TextField(blank=True, default='', help_text="Annotator's edited version")
    time_in_call_secs = models.FloatField(null=True, blank=True)
    is_edited = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)
    is_inserted = models.BooleanField(default=False)
    weight = models.IntegerField(null=True, blank=True, help_text="Training weight (0=don't learn, 1=learn). Null=auto.")
    rag_context = models.JSONField(
        default=list, blank=True,
        help_text="RAG chunks retrieved for this turn: [{document_id, chunk_id, content, vector_distance}]"
    )

    class Meta:
        ordering = ['conversation', 'position']
        unique_together = ['conversation', 'position']

    @property
    def display_text(self):
        return self.edited_text if self.is_edited else self.original_text

    def __str__(self):
        return f"Turn {self.position} ({self.get_role_display()})"


class ToolCall(models.Model):
    turn = models.ForeignKey(Turn, on_delete=models.CASCADE, related_name='tool_calls')
    tool_name = models.CharField(max_length=100)
    original_args = models.JSONField(default=dict, help_text="Original arguments from the LLM")
    edited_args = models.JSONField(default=dict, blank=True, help_text="Annotator's corrected arguments")
    status_code = models.IntegerField(null=True, blank=True)
    response_body = models.JSONField(default=dict, blank=True, help_text="Response from the webhook")
    error_message = models.TextField(blank=True, default='')
    is_edited = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)

    @property
    def display_args(self):
        return self.edited_args if self.is_edited else self.original_args

    def __str__(self):
        return f"{self.tool_name} (turn {self.turn.position})"


class SystemPrompt(models.Model):
    name = models.CharField(max_length=200, help_text="Label for this prompt version")
    content = models.TextField(help_text="The full system prompt text")
    is_active = models.BooleanField(default=False, help_text="Only one can be active at a time")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({'active' if self.is_active else 'inactive'})"

    def save(self, *args, **kwargs):
        if self.is_active:
            SystemPrompt.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)


class ExportLog(models.Model):
    exported_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True
    )
    conversation_count = models.IntegerField(default=0)
    token_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Export {self.created_at.strftime('%Y-%m-%d %H:%M')} ({self.conversation_count} convs)"

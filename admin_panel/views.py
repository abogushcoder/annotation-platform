import json
import logging
from functools import wraps

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q, Avg, F
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone

from accounts.models import User
from conversations.models import Agent, Conversation, Turn, ToolCall, SystemPrompt, ExportLog

logger = logging.getLogger(__name__)


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_admin():
            return render(request, 'admin_panel/403.html', status=403)
        return view_func(request, *args, **kwargs)
    return wrapper


# ---- Dashboard ----

@admin_required
def admin_dashboard(request):
    total = Conversation.objects.count()
    unassigned = Conversation.objects.filter(status='unassigned').count()
    assigned = Conversation.objects.filter(status='assigned').count()
    in_progress = Conversation.objects.filter(status='in_progress').count()
    completed = Conversation.objects.filter(status='completed').count()
    approved = Conversation.objects.filter(status='approved').count()
    flagged = Conversation.objects.filter(status='flagged').count()

    # Per-annotator stats
    annotators = User.objects.filter(role='annotator', is_active=True)
    team_stats = []
    for ann in annotators:
        ann_total = Conversation.objects.filter(assigned_to=ann).count()
        ann_completed = Conversation.objects.filter(
            assigned_to=ann, status__in=['completed', 'approved']
        ).count()
        pct = int((ann_completed / ann_total * 100)) if ann_total > 0 else 0
        team_stats.append({
            'user': ann,
            'total': ann_total,
            'completed': ann_completed,
            'pct': pct,
        })

    review_queue = Conversation.objects.filter(status='completed').select_related(
        'assigned_to', 'agent'
    ).order_by('-completed_at')[:10]

    return render(request, 'admin_panel/dashboard.html', {
        'total': total, 'unassigned': unassigned, 'assigned': assigned,
        'in_progress': in_progress, 'completed': completed,
        'approved': approved, 'flagged': flagged,
        'team_stats': team_stats, 'review_queue': review_queue,
    })


# ---- Agent Management ----

@admin_required
def agent_list(request):
    agents = Agent.objects.annotate(conv_count=Count('conversations')).all()
    return render(request, 'admin_panel/agents.html', {'agents': agents})


@admin_required
def agent_add(request):
    if request.method == 'POST':
        agent_id = request.POST['agent_id']
        label = request.POST.get('label') or ''
        api_key = request.POST.get('api_key') or settings.ELEVENLABS_API_KEY

        # If label wasn't provided, try to get it from the selected agent
        if not label and api_key:
            try:
                from conversations.services.elevenlabs import ElevenLabsClient
                client = ElevenLabsClient(api_key)
                for a in client.list_agents():
                    if a['agent_id'] == agent_id:
                        label = a.get('name', agent_id)
                        break
            except Exception:
                label = agent_id

        Agent.objects.create(
            agent_id=agent_id,
            label=label or agent_id,
            elevenlabs_api_key=api_key,
        )

        # Auto-import system prompt from the agent's ElevenLabs config
        if api_key:
            try:
                from conversations.services.elevenlabs import ElevenLabsClient
                client = ElevenLabsClient(api_key)
                agent_config = client.get_agent(agent_id)
                prompt_text = (
                    agent_config
                    .get('conversation_config', {})
                    .get('agent', {})
                    .get('prompt', {})
                    .get('prompt', '')
                )
                if prompt_text:
                    prompt_name = f"{label or agent_id} - System Prompt"
                    existing = SystemPrompt.objects.filter(name=prompt_name).first()
                    if not existing:
                        has_active = SystemPrompt.objects.filter(is_active=True).exists()
                        SystemPrompt.objects.create(
                            name=prompt_name,
                            content=prompt_text,
                            is_active=not has_active,
                        )
                        messages.info(
                            request,
                            f'System prompt imported from ElevenLabs agent'
                            f'{" and set as active" if not has_active else " (inactive — another prompt is already active)"}.'
                        )
            except Exception:
                pass  # Non-critical — user can still add prompts manually

        messages.success(request, 'Agent added successfully.')
        return redirect('agent_list')

    # Fetch available agents from ElevenLabs for the dropdown
    elevenlabs_agents = []
    existing_ids = set(Agent.objects.values_list('agent_id', flat=True))
    if settings.ELEVENLABS_API_KEY:
        try:
            from conversations.services.elevenlabs import ElevenLabsClient
            client = ElevenLabsClient(settings.ELEVENLABS_API_KEY)
            elevenlabs_agents = [
                a for a in client.list_agents()
                if a['agent_id'] not in existing_ids
            ]
        except Exception as e:
            messages.warning(request, f'Could not fetch agents from ElevenLabs: {e}')

    return render(request, 'admin_panel/agent_form.html', {
        'action': 'Add',
        'elevenlabs_agents': elevenlabs_agents,
    })


@admin_required
def agent_edit(request, pk):
    agent = get_object_or_404(Agent, pk=pk)
    if request.method == 'POST':
        agent.agent_id = request.POST['agent_id']
        agent.label = request.POST['label']
        agent.elevenlabs_api_key = request.POST['api_key']
        agent.save()
        messages.success(request, 'Agent updated.')
        return redirect('agent_list')
    return render(request, 'admin_panel/agent_form.html', {'action': 'Edit', 'agent': agent})


@admin_required
def agent_delete(request, pk):
    agent = get_object_or_404(Agent, pk=pk)
    if request.method == 'POST':
        agent.delete()
        messages.success(request, 'Agent deleted.')
    return redirect('agent_list')


@admin_required
def agent_sync(request, pk):
    agent = get_object_or_404(Agent, pk=pk)
    if request.method == 'POST':
        from conversations.services.sync import sync_agent_conversations
        try:
            stats = sync_agent_conversations(agent)
            messages.success(
                request,
                f"Sync complete: {stats['imported']} imported, "
                f"{stats['skipped']} skipped, {stats['errors']} errors."
            )
        except Exception as e:
            messages.error(request, f"Sync failed: {e}")
    return redirect('agent_list')


# ---- Assignment ----

@admin_required
def assign_conversations(request):
    if request.method == 'POST':
        conv_ids = request.POST.getlist('conversation_ids')
        assignee_id = request.POST.get('assignee')
        if conv_ids and assignee_id:
            assignee = get_object_or_404(User, pk=assignee_id)
            updated = Conversation.objects.filter(
                pk__in=conv_ids, status='unassigned'
            ).update(status='assigned', assigned_to=assignee)
            messages.success(request, f'{updated} conversations assigned to {assignee.username}.')
        return redirect('assign_conversations')

    agent_filter = request.GET.get('agent')
    status_filter = request.GET.get('status', 'unassigned')

    conversations = Conversation.objects.select_related('agent', 'assigned_to')
    if status_filter:
        conversations = conversations.filter(status=status_filter)
    if agent_filter:
        conversations = conversations.filter(agent_id=agent_filter)

    agents = Agent.objects.all()
    annotators = User.objects.filter(role='annotator', is_active=True)

    unassigned_count = Conversation.objects.filter(status='unassigned').count()
    assigned_count = Conversation.objects.filter(status='assigned').count()
    in_progress_count = Conversation.objects.filter(status='in_progress').count()

    return render(request, 'admin_panel/assign.html', {
        'conversations': conversations,
        'agents': agents,
        'annotators': annotators,
        'status_filter': status_filter,
        'agent_filter': agent_filter,
        'unassigned_count': unassigned_count,
        'assigned_count': assigned_count,
        'in_progress_count': in_progress_count,
    })


@admin_required
def auto_distribute(request):
    if request.method == 'POST':
        annotators = list(User.objects.filter(role='annotator', is_active=True))
        if not annotators:
            messages.error(request, 'No active annotators.')
            return redirect('assign_conversations')

        unassigned = list(Conversation.objects.filter(status='unassigned'))
        for i, conv in enumerate(unassigned):
            conv.assigned_to = annotators[i % len(annotators)]
            conv.status = 'assigned'
            conv.save()

        messages.success(request, f'{len(unassigned)} conversations distributed.')
    return redirect('assign_conversations')


# ---- Review ----

@admin_required
def review_queue(request):
    conversations = Conversation.objects.filter(
        status='completed'
    ).select_related('assigned_to', 'agent').order_by('-completed_at')

    return render(request, 'admin_panel/review.html', {
        'conversations': conversations,
    })


@admin_required
def review_conversation(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    turns = conversation.turns.prefetch_related('tool_calls').all()

    return render(request, 'admin_panel/review_detail.html', {
        'conversation': conversation,
        'turns': turns,
    })


@admin_required
def approve_conversation(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if request.method == 'POST':
        conversation.status = 'approved'
        conversation.reviewed_at = timezone.now()
        conversation.save()
        messages.success(request, f'Conversation {conversation.elevenlabs_id} approved.')
    return redirect('review_queue')


@admin_required
def reject_conversation(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if request.method == 'POST':
        conversation.status = 'assigned'
        conversation.reviewer_notes = request.POST.get('reviewer_notes', '')
        conversation.reviewed_at = timezone.now()
        conversation.completed_at = None
        conversation.save()
        messages.success(request, f'Conversation {conversation.elevenlabs_id} rejected.')
    return redirect('review_queue')


# ---- Export ----

@admin_required
def export_page(request):
    from conversations.services.export import generate_jsonl_examples, count_tokens, estimate_training_cost

    agents = Agent.objects.all()
    approved_count = Conversation.objects.filter(status='approved').count()
    active_prompt = SystemPrompt.objects.filter(is_active=True).first()

    # Pre-calculate token estimate
    token_count = 0
    estimated_cost = 0
    if approved_count > 0:
        examples = generate_jsonl_examples()
        token_count = count_tokens(examples)
        estimated_cost = estimate_training_cost(token_count)

    return render(request, 'admin_panel/export.html', {
        'agents': agents,
        'approved_count': approved_count,
        'active_prompt': active_prompt,
        'token_count': token_count,
        'estimated_cost': estimated_cost,
    })


@admin_required
def export_preview(request):
    from conversations.services.export import generate_jsonl_examples
    import json
    examples = generate_jsonl_examples(limit=3)
    formatted = [json.dumps(ex, indent=2) for ex in examples]
    html = ""
    for i, f in enumerate(formatted):
        html += f'<div class="mb-4"><h4 class="text-sm font-semibold text-gray-700 mb-1">Example {i+1}</h4>'
        html += f'<pre class="bg-gray-900 text-green-400 text-xs p-3 rounded overflow-x-auto max-h-64">{f}</pre></div>'
    if not formatted:
        html = '<p class="text-gray-500 text-sm">No approved conversations to preview.</p>'
    return HttpResponse(html)


@admin_required
def export_download(request):
    from conversations.services.export import (
        generate_jsonl_examples, export_jsonl, split_train_validation,
        count_tokens, estimate_training_cost
    )
    import zipfile
    import io
    from datetime import date

    filter_type = request.GET.get('filter', 'all')
    agent_id = request.GET.get('agent_id')
    include_system = 'include_system_prompt' in request.GET
    include_tools = 'include_tools' in request.GET
    split = 'split' in request.GET
    tool_calls_only = 'tool_calls_only' in request.GET

    kwargs = {
        'include_system_prompt': include_system,
        'include_tools': include_tools,
        'tool_calls_only': tool_calls_only,
    }
    if filter_type == 'agent' and agent_id:
        kwargs['agent_id'] = agent_id

    examples = generate_jsonl_examples(**kwargs)

    if not examples:
        messages.warning(request, 'No valid examples to export.')
        return redirect('export_page')

    if len(examples) < 10:
        messages.warning(
            request,
            f'OpenAI requires at least 10 training examples. You have {len(examples)}.'
        )
        return redirect('export_page')

    today = date.today().isoformat()

    if split:
        train, val = split_train_validation(examples)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f'training_data_{today}.jsonl', export_jsonl(train))
            zf.writestr(f'validation_data_{today}.jsonl', export_jsonl(val))
        buf.seek(0)
        response = HttpResponse(buf.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="training_data_{today}.zip"'
    else:
        jsonl_content = export_jsonl(examples)
        response = HttpResponse(jsonl_content, content_type='application/jsonl')
        response['Content-Disposition'] = f'attachment; filename="training_data_{today}.jsonl"'

    # Log export
    ExportLog.objects.create(
        exported_by=request.user,
        conversation_count=len(examples),
        token_count=count_tokens(examples),
    )

    return response


# ---- Analytics ----

@admin_required
def analytics_page(request):
    from django.db.models import Count, Q, Avg

    annotators = User.objects.filter(role='annotator', is_active=True)

    # Per-annotator metrics
    annotator_metrics = []
    for ann in annotators:
        total_assigned = Conversation.objects.filter(assigned_to=ann).count()
        completed = Conversation.objects.filter(
            assigned_to=ann, status__in=['completed', 'approved']
        ).count()
        approved = Conversation.objects.filter(assigned_to=ann, status='approved').count()
        rejected = Conversation.objects.filter(
            assigned_to=ann, reviewer_notes__gt=''
        ).count()
        flagged = Conversation.objects.filter(assigned_to=ann, status='flagged').count()

        completion_rate = int(completed / total_assigned * 100) if total_assigned > 0 else 0
        rejection_rate = int(rejected / completed * 100) if completed > 0 else 0
        flag_rate = int(flagged / total_assigned * 100) if total_assigned > 0 else 0

        # Edit rates
        ann_turns = Turn.objects.filter(conversation__assigned_to=ann)
        total_turns = ann_turns.count()
        edited_turns = ann_turns.filter(is_edited=True).count()
        edit_rate = int(edited_turns / total_turns * 100) if total_turns > 0 else 0

        ann_tc = ToolCall.objects.filter(turn__conversation__assigned_to=ann)
        total_tc = ann_tc.count()
        edited_tc = ann_tc.filter(is_edited=True).count()
        tc_edit_rate = int(edited_tc / total_tc * 100) if total_tc > 0 else 0

        annotator_metrics.append({
            'user': ann,
            'total_assigned': total_assigned,
            'completed': completed,
            'approved': approved,
            'completion_rate': completion_rate,
            'rejection_rate': rejection_rate,
            'flag_rate': flag_rate,
            'edit_rate': edit_rate,
            'tc_edit_rate': tc_edit_rate,
        })

    # Pipeline funnel
    pipeline = {
        'unassigned': Conversation.objects.filter(status='unassigned').count(),
        'assigned': Conversation.objects.filter(status='assigned').count(),
        'in_progress': Conversation.objects.filter(status='in_progress').count(),
        'completed': Conversation.objects.filter(status='completed').count(),
        'approved': Conversation.objects.filter(status='approved').count(),
        'flagged': Conversation.objects.filter(status='flagged').count(),
    }
    pipeline['total'] = sum(pipeline.values())

    # Export history
    export_history = ExportLog.objects.select_related('exported_by')[:10]

    # Overall edit rates
    total_turns = Turn.objects.count()
    total_edited_turns = Turn.objects.filter(is_edited=True).count()
    overall_edit_rate = int(total_edited_turns / total_turns * 100) if total_turns > 0 else 0

    total_tc = ToolCall.objects.count()
    total_edited_tc = ToolCall.objects.filter(is_edited=True).count()
    overall_tc_edit_rate = int(total_edited_tc / total_tc * 100) if total_tc > 0 else 0

    return render(request, 'admin_panel/analytics.html', {
        'annotator_metrics': annotator_metrics,
        'pipeline': pipeline,
        'export_history': export_history,
        'overall_edit_rate': overall_edit_rate,
        'overall_tc_edit_rate': overall_tc_edit_rate,
    })


# ---- Team ----

@admin_required
def team_management(request):
    members = User.objects.filter(role='annotator').order_by('-is_active', 'username')
    return render(request, 'admin_panel/team.html', {'members': members})


@admin_required
def team_invite(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST.get('email', '')
        password = request.POST['password']
        first_name = request.POST.get('first_name', '')
        last_name = request.POST.get('last_name', '')

        if User.objects.filter(username=username).exists():
            messages.error(request, f'Username "{username}" is already taken. Please choose another.')
            return render(request, 'admin_panel/team_invite.html', {
                'form_data': {
                    'username': username, 'email': email,
                    'first_name': first_name, 'last_name': last_name,
                }
            })

        user = User.objects.create_user(
            username=username, email=email, password=password,
            first_name=first_name, last_name=last_name,
            role='annotator',
        )
        messages.success(request, f'Invited {username} as annotator.')
        return redirect('team_management')
    return render(request, 'admin_panel/team_invite.html')


@admin_required
def team_toggle_active(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == 'POST':
        user.is_active = not user.is_active
        user.save()
        status = 'activated' if user.is_active else 'deactivated'
        messages.success(request, f'{user.username} {status}.')
    return redirect('team_management')


# ---- System Prompts ----

@admin_required
def prompt_management(request):
    prompts = SystemPrompt.objects.order_by('-is_active', '-created_at')
    return render(request, 'admin_panel/prompts.html', {'prompts': prompts})


@admin_required
def prompt_add(request):
    if request.method == 'POST':
        SystemPrompt.objects.create(
            name=request.POST['name'],
            content=request.POST['content'],
            is_active=request.POST.get('is_active') == 'on',
        )
        messages.success(request, 'Prompt created.')
        return redirect('prompt_management')
    return render(request, 'admin_panel/prompt_form.html', {'action': 'Create'})


@admin_required
def prompt_edit(request, pk):
    prompt = get_object_or_404(SystemPrompt, pk=pk)
    if request.method == 'POST':
        prompt.name = request.POST['name']
        prompt.content = request.POST['content']
        prompt.is_active = request.POST.get('is_active') == 'on'
        prompt.save()
        messages.success(request, 'Prompt updated.')
        return redirect('prompt_management')
    return render(request, 'admin_panel/prompt_form.html', {'action': 'Edit', 'prompt': prompt})


@admin_required
def prompt_activate(request, pk):
    prompt = get_object_or_404(SystemPrompt, pk=pk)
    if request.method == 'POST':
        prompt.is_active = True
        prompt.save()
        messages.success(request, f'"{prompt.name}" is now active.')
    return redirect('prompt_management')

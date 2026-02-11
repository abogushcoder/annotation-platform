from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from .models import Conversation, Turn, ToolCall


@login_required
def conversation_list(request):
    status_filter = request.GET.get('status', 'assigned')
    conversations = Conversation.objects.filter(assigned_to=request.user)
    if status_filter and status_filter != 'all':
        conversations = conversations.filter(status=status_filter)

    assigned_count = Conversation.objects.filter(assigned_to=request.user, status='assigned').count()
    in_progress_count = Conversation.objects.filter(assigned_to=request.user, status='in_progress').count()
    completed_count = Conversation.objects.filter(assigned_to=request.user, status='completed').count()

    return render(request, 'conversations/list.html', {
        'conversations': conversations,
        'status_filter': status_filter,
        'assigned_count': assigned_count,
        'in_progress_count': in_progress_count,
        'completed_count': completed_count,
    })


@login_required
def conversation_editor(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if not request.user.is_admin() and conversation.assigned_to != request.user:
        return HttpResponse("Permission denied", status=403)

    turns = conversation.turns.prefetch_related('tool_calls').all()

    if conversation.status == 'assigned' and conversation.assigned_to == request.user:
        conversation.status = 'in_progress'
        conversation.save()

    return render(request, 'conversations/editor.html', {
        'conversation': conversation,
        'turns': turns,
    })


@login_required
def turn_edit(request, pk, turn_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    turn = get_object_or_404(Turn, pk=turn_id, conversation=conversation)

    if request.method == 'POST':
        edited_text = request.POST.get('edited_text', '').strip()
        if edited_text and edited_text != turn.original_text:
            turn.edited_text = edited_text
            turn.is_edited = True
        else:
            turn.edited_text = ''
            turn.is_edited = False
        turn.save()
        return render(request, 'conversations/partials/turn_display.html', {
            'turn': turn, 'conversation': conversation
        })

    return render(request, 'conversations/partials/turn_edit.html', {
        'turn': turn, 'conversation': conversation
    })


@login_required
def turn_display(request, pk, turn_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    turn = get_object_or_404(Turn, pk=turn_id, conversation=conversation)
    return render(request, 'conversations/partials/turn_display.html', {
        'turn': turn, 'conversation': conversation
    })


@login_required
def tool_call_edit(request, pk, tc_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    tc = get_object_or_404(ToolCall, pk=tc_id, turn__conversation=conversation)

    if request.method == 'POST':
        import json
        edited_args = {}
        for key, value in request.POST.items():
            if key.startswith('arg_'):
                field_name = key[4:]
                try:
                    edited_args[field_name] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    edited_args[field_name] = value

        if edited_args != tc.original_args:
            tc.edited_args = edited_args
            tc.is_edited = True
        else:
            tc.edited_args = {}
            tc.is_edited = False
        tc.save()
        return render(request, 'conversations/partials/tool_call_card.html', {
            'tc': tc, 'conversation': conversation
        })

    return render(request, 'conversations/partials/tool_call_form.html', {
        'tc': tc, 'conversation': conversation
    })


@login_required
def tool_call_display(request, pk, tc_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    tc = get_object_or_404(ToolCall, pk=tc_id, turn__conversation=conversation)
    return render(request, 'conversations/partials/tool_call_card.html', {
        'tc': tc, 'conversation': conversation
    })


@login_required
def conversation_complete(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if request.method == 'POST':
        if conversation.assigned_to == request.user or request.user.is_admin():
            conversation.status = 'completed'
            conversation.completed_at = timezone.now()
            conversation.save()
    return redirect('conversation_editor', pk=pk)


@login_required
def conversation_flag(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if request.method == 'POST':
        if conversation.assigned_to == request.user or request.user.is_admin():
            conversation.status = 'flagged'
            flag_notes = request.POST.get('flag_notes', '')
            if flag_notes:
                conversation.annotator_notes += f"\n[FLAGGED] {flag_notes}"
            conversation.save()
    return redirect('conversation_editor', pk=pk)


@login_required
def conversation_notes(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if request.method == 'POST':
        conversation.annotator_notes = request.POST.get('annotator_notes', '')
        conversation.save()
        return HttpResponse('<span class="text-green-600 text-sm">Saved</span>')
    return HttpResponse('')

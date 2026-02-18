import requests

from django.contrib.auth.decorators import login_required
from django.db.models import Q, F
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from .models import Conversation, Turn, ToolCall, Tag
from .services.elevenlabs import ElevenLabsClient


@login_required
def conversation_list(request):
    status_filter = request.GET.get('status', 'assigned')
    search_query = request.GET.get('q', '')
    conversations = Conversation.objects.filter(assigned_to=request.user)
    if status_filter and status_filter != 'all':
        conversations = conversations.filter(status=status_filter)
    if search_query:
        conversations = conversations.filter(
            Q(turns__original_text__icontains=search_query) |
            Q(turns__edited_text__icontains=search_query) |
            Q(elevenlabs_id__icontains=search_query)
        ).distinct()

    assigned_count = Conversation.objects.filter(assigned_to=request.user, status='assigned').count()
    in_progress_count = Conversation.objects.filter(assigned_to=request.user, status='in_progress').count()
    completed_count = Conversation.objects.filter(assigned_to=request.user, status='completed').count()

    return render(request, 'conversations/list.html', {
        'conversations': conversations,
        'status_filter': status_filter,
        'search_query': search_query,
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

    if conversation.status == 'assigned' and (conversation.assigned_to == request.user or request.user.is_admin()):
        conversation.status = 'in_progress'
        conversation.save()

    all_tags = Tag.objects.all()

    return render(request, 'conversations/editor.html', {
        'conversation': conversation,
        'turns': turns,
        'all_tags': all_tags,
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


@login_required
def conversation_audio(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if not request.user.is_admin() and conversation.assigned_to != request.user:
        return HttpResponse("Permission denied", status=403)

    if not conversation.has_audio or not conversation.elevenlabs_id:
        return HttpResponse("Audio not available", status=404)

    try:
        client = ElevenLabsClient(conversation.agent.elevenlabs_api_key)
        audio_bytes = client.get_conversation_audio(conversation.elevenlabs_id)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 422:
            return HttpResponse("Audio no longer available", status=404)
        return HttpResponse("Failed to fetch audio", status=502)
    except Exception:
        return HttpResponse("Failed to fetch audio", status=502)

    response = HttpResponse(audio_bytes, content_type='audio/mpeg')
    response['Content-Length'] = len(audio_bytes)
    response['Content-Disposition'] = 'inline'
    response['Cache-Control'] = 'private, max-age=3600'
    return response


@login_required
def turn_delete(request, pk, turn_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    turn = get_object_or_404(Turn, pk=turn_id, conversation=conversation)
    if request.method == 'POST':
        turn.is_deleted = not turn.is_deleted
        turn.save()
    return render(request, 'conversations/partials/turn_display.html', {
        'turn': turn, 'conversation': conversation
    })


@login_required
def tool_call_delete(request, pk, tc_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    tc = get_object_or_404(ToolCall, pk=tc_id, turn__conversation=conversation)
    if request.method == 'POST':
        tc.is_deleted = not tc.is_deleted
        tc.save()
    return render(request, 'conversations/partials/tool_call_card.html', {
        'tc': tc, 'conversation': conversation
    })


@login_required
def turn_toggle_weight(request, pk, turn_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    turn = get_object_or_404(Turn, pk=turn_id, conversation=conversation)
    if request.method == 'POST' and turn.role == 'agent':
        if turn.weight is None:
            turn.weight = 0
        elif turn.weight == 0:
            turn.weight = 1
        else:
            turn.weight = None
        turn.save()
    return render(request, 'conversations/partials/turn_display.html', {
        'turn': turn, 'conversation': conversation
    })


@login_required
def turn_insert(request, pk, after_turn_id):
    conversation = get_object_or_404(Conversation, pk=pk)
    after_turn = get_object_or_404(Turn, pk=after_turn_id, conversation=conversation)

    if request.method == 'POST':
        role = request.POST.get('role', 'agent')
        text = request.POST.get('text', '').strip()
        if not text:
            return HttpResponse('Text required', status=400)

        # Shift positions in reverse order to avoid unique constraint violations
        for t in Turn.objects.filter(
            conversation=conversation, position__gt=after_turn.position
        ).order_by('-position'):
            t.position += 1
            t.save()

        Turn.objects.create(
            conversation=conversation,
            position=after_turn.position + 1,
            role=role,
            original_text=text,
            is_inserted=True,
        )

        turns = conversation.turns.prefetch_related('tool_calls').all()
        return render(request, 'conversations/partials/turn_list.html', {
            'turns': turns, 'conversation': conversation
        })

    return render(request, 'conversations/partials/turn_insert_form.html', {
        'conversation': conversation, 'after_turn': after_turn
    })


@login_required
def conversation_tag_manage(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk)
    if request.method == 'POST':
        action = request.POST.get('action')
        tag_name = request.POST.get('tag_name', '').strip()
        if action == 'add' and tag_name:
            tag, _ = Tag.objects.get_or_create(name=tag_name)
            conversation.tags.add(tag)
        elif action == 'remove':
            tag_id = request.POST.get('tag_id')
            if tag_id:
                conversation.tags.remove(tag_id)
    all_tags = Tag.objects.all()
    return render(request, 'conversations/partials/tag_bar.html', {
        'conversation': conversation, 'all_tags': all_tags
    })

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render, redirect


def admin_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_admin():
            return HttpResponse("Permission denied", status=403)
        return view_func(request, *args, **kwargs)
    return wrapper


@admin_required
def admin_dashboard(request):
    return render(request, 'admin_panel/dashboard.html')


@admin_required
def agent_list(request):
    return render(request, 'admin_panel/agents.html')


@admin_required
def agent_add(request):
    return render(request, 'admin_panel/agents.html')


@admin_required
def agent_edit(request, pk):
    return render(request, 'admin_panel/agents.html')


@admin_required
def agent_delete(request, pk):
    return redirect('agent_list')


@admin_required
def agent_sync(request, pk):
    return redirect('agent_list')


@admin_required
def assign_conversations(request):
    return render(request, 'admin_panel/assign.html')


@admin_required
def auto_distribute(request):
    return redirect('assign_conversations')


@admin_required
def review_queue(request):
    return render(request, 'admin_panel/review.html')


@admin_required
def review_conversation(request, pk):
    return render(request, 'admin_panel/review.html')


@admin_required
def approve_conversation(request, pk):
    return redirect('review_queue')


@admin_required
def reject_conversation(request, pk):
    return redirect('review_queue')


@admin_required
def export_page(request):
    return render(request, 'admin_panel/export.html')


@admin_required
def export_preview(request):
    return HttpResponse('{}')


@admin_required
def export_download(request):
    return HttpResponse('{}')


@admin_required
def analytics_page(request):
    return render(request, 'admin_panel/analytics.html')


@admin_required
def team_management(request):
    return render(request, 'admin_panel/team.html')


@admin_required
def team_invite(request):
    return render(request, 'admin_panel/team.html')


@admin_required
def team_toggle_active(request, pk):
    return redirect('team_management')


@admin_required
def prompt_management(request):
    return render(request, 'admin_panel/prompts.html')


@admin_required
def prompt_add(request):
    return render(request, 'admin_panel/prompts.html')


@admin_required
def prompt_edit(request, pk):
    return render(request, 'admin_panel/prompts.html')


@admin_required
def prompt_activate(request, pk):
    return redirect('prompt_management')

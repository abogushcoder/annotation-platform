from django.urls import path
from . import views

urlpatterns = [
    path('', views.conversation_list, name='annotator_dashboard'),
    path('<int:pk>/', views.conversation_editor, name='conversation_editor'),
    path('<int:pk>/turn/<int:turn_id>/edit/', views.turn_edit, name='turn_edit'),
    path('<int:pk>/turn/<int:turn_id>/display/', views.turn_display, name='turn_display'),
    path('<int:pk>/tool/<int:tc_id>/edit/', views.tool_call_edit, name='tool_call_edit'),
    path('<int:pk>/tool/<int:tc_id>/display/', views.tool_call_display, name='tool_call_display'),
    path('<int:pk>/complete/', views.conversation_complete, name='conversation_complete'),
    path('<int:pk>/flag/', views.conversation_flag, name='conversation_flag'),
    path('<int:pk>/notes/', views.conversation_notes, name='conversation_notes'),
    path('<int:pk>/audio/', views.conversation_audio, name='conversation_audio'),
]

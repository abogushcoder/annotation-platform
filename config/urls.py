from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

from accounts.views import login_view, logout_view, dashboard_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda r: __import__('django.shortcuts', fromlist=['redirect']).redirect('dashboard')),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('conversations/', include('conversations.urls')),
    path('admin-panel/', include('admin_panel.urls')),
]

handler404 = lambda request, exception: render(request, '404.html', status=404)

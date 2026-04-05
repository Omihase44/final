from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'), 
    path('Admin_login/',views.Admin_login,name='Admin_login'), 
    path('brain/',views.brain,name='brain'),
    path('Alzhimers/',views.Alzhimers,name='Alzhimers'),
    path('logout/',views.logout,name='logout'),
    path('View_Users/',views.View_Users,name='View_Users'), 
    path('base/',views.base,name='base'),

   
    ]


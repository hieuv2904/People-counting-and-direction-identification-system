from django.urls import path
from . import views

urlpatterns = [
    path("", views.homePage, name='homePage'),
    path("cameras/", views.home, name='cameras'),
    path("alertLogs/",views.AlertLogs,name="alertLogs"),
    path('get_alert_logs/', views.get_alert_logs, name='get_alert_logs'),
    path("cam0/", views.Cam0, name='cam0'),
    path("cam1/", views.Cam1, name='cam1')
    # path("cam2/", views.Cam2, name='cam2'),
    # path("cam3/", views.Cam3, name='cam3'),
    # path("cam4/", views.Cam4, name='cam4'),
    # path("cam5/", views.Cam5, name='cam5'),
]
from django.contrib import admin
from django.urls import path
from prediction import views

urlpatterns = [
    path('detection', views.detection, name='detection')
]

"""
URL configuration for healthcare_assistant project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from collections import namedtuple

from django.contrib import admin
from django.urls import path
from assistant.views import index, analyze_symptoms, speech_to_text  # Import speech_to_text

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('analyze/', analyze_symptoms, name='analyze_symptoms'),
    path('speech-to-text/', speech_to_text, name='speech_to_text'),  # Add this line
]

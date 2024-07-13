from django import forms
from .models import *

class AlertLogForms(forms.ModelForm):
    class Meta:
        model = Alert_log
        fields = ['alert', 'camera_number', 'clip_link']
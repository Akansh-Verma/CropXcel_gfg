from django import forms
from .models import Image


class ImageForm(forms.Form):
    model = Image
    fields = ['image']

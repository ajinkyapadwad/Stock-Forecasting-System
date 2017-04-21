from django import forms

class NameForm(forms.Form):
    newform = forms.CharField(label='Your naaam', max_length=100)
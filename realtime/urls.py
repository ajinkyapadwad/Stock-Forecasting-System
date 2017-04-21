from django.conf.urls import url
# from django.contrib import admin
# from DombiFinance import views as DombiFinance_views 


from . import views

urlpatterns = [

    url(r'^index', views.index, name='index'),
    url(r'^prediction', views.prediction, name='prediction'),
    url(r'^news', views.news, name='news'),
    url(r'^contact', views.contact, name='contact'),
    url(r'^newprediction', views.newprediction, name='newprediction'),
]
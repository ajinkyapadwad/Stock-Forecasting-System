from django.conf.urls import url
from django.contrib import admin
from DombiFinance import views as DombiFinance_views 

from realtime import views

from . import views

urlpatterns = [
    #url(r'^$', views.index, name='index'),
    url(r'^$', views.index, name='index'),
    url(r'^$', DombiFinance_views.index,),
    url(r'^admin/', admin.site.urls),
]
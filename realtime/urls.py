from django.conf.urls import url
# from django.contrib import admin
# from DombiFinance import views as DombiFinance_views 


from . import views

urlpatterns = [

    url(r'^index', views.index, name='index'),
    url(r'^pred', views.pred, name='pred'),
    url(r'^news', views.news, name='news'),
    url(r'^contact', views.contact, name='contact'),
    url(r'^passing', views.passing, name='passing'),
    #url( r'^search/(?P<name>[a-zA-Z0-9_.-]+)/$', views.search, name='search_stock' ),
]

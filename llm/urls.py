from django.urls import path
# from .views import PhotoDetail, GetUploadURL
from . import views

urlpatterns = [
    path('latest-analysis/', views.LatestAnalysisResultView.as_view()),
    path('latest-symbolanalysis/', views.LatestSymbolResultView.as_view()),
    path('recent_analysis/', views.RecentAnalysisResultsView.as_view()),
]

from django.urls import path
from .views import  get_training_info, normalize_data, remove_outliers, upload_csv,train_model,predict_model

urlpatterns = [
    path('api/upload/', upload_csv, name='upload_csv'),
    path('api/train/', train_model, name='train_model'),
     path('api/predict/', predict_model, name='predict_model'),
      path('api/normalize/',normalize_data, name='normalize_data'),
        path('api/remove_outliers/',remove_outliers, name='remove_outliers'),
        path('api/get_training_info/',get_training_info, name='get_training_info'),
    
]


 
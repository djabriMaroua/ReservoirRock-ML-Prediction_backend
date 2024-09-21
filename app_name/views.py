# views.py

import os
from platform import processor
from statistics import LinearRegression
from django.http import JsonResponse
from lightgbm import LGBMRegressor
from networkx import random_tree
import numpy as np
from sklearn.metrics import r2_score


from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge, TheilSenRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVR, OneClassSVM
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from rest_framework.decorators import api_view
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr  # Import spearmanr for Spearman's rank correlation
import time
from lightgbm import LGBMRegressor
from numpy import sqrt
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, TheilSenRegressor, HuberRegressor, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
 
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from stage_backend import settings

@api_view(['POST'])
def upload_csv(request):
    # Check if the request contains any files
    if not request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_files = request.FILES.getlist('file')  # Get all uploaded files
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = f'{settings.MEDIA_ROOT}/{uploaded_file.name}'
        file_paths.append(file_path)
        
        # Save each file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
    
    # You can process each file as needed, here we process the first one as an example
    try:
       
        df = pd.read_csv(file_path)
        empty_columns = df.columns[df.isna().all()].tolist()
        files_paths = []
        print(f"Empty columns: {empty_columns}")
        print(f"Number of empty columns: {len(empty_columns)}")
        if len(empty_columns)==0:  # If there are missing values
            print("all comns present ")
            final_path = os.path.join(settings.MEDIA_ROOT, 'prediction', uploaded_file.name)
            print("file pathhhhhhhh :",final_path)
            files_paths.append(final_path)
            # with open(files_paths, 'wb+') as destination:
            #    for chunk in uploaded_file.chunks():
            #     destination.write(chunk)
        else:  # If all columns have values
            print("nillll")
            print(df)
        #     # final_path = os.path.join(settings.MEDIA_ROOT, 'media', uploaded_file.name)
        #     # file_paths['media'].append(final_path)
        
        # # Move the file to the correct folder
        # os.rename(file_path, final_path)
        return JsonResponse({'message': 'Files uploaded successfully!'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)



# # views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
import pandas as pd
import json

# @api_view(['POST'])
# def train_model(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             file_name = data.get('file_name', '')  # Retrieve file name if available
#             X_columns = data.get('X_columns', [])
#             y_column = data.get('y_column', '')
#             model_type = data.get('model_type', '')

#             print("Training data:", data)
#             folder_path = 'media/'
#             csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
#             # Load the CSV file data from a predefined location
#             # This is just an example; adjust the file path accordingly
             
#             train_dfs = []
#             for file in csv_files:
#                 df = pd.read_csv(file)
#                 df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
#                 df.columns = df.columns.str.strip()
#                 train_dfs.append(df)

#             df_train = pd.concat(train_dfs, axis=0)
            
#             # Print some rows of the CSV file to the console
#             print("CSV File Sample Data:")
#             print(df_train.head())
          
#             # # Prepare a sample of the CSV data to send back
#             sample_data = df.head().to_dict(orient='records')  # Convert to a list of dictionaries
#             print("df train colmns",df_train.columns)

#             if y_column not in df_train.columns:
#                 print("target colmns not found")

#             # Define features and target
#             X = df_train[X_columns]
#             y = df_train[y_column]

#             # Clean the data
#             df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
#             df_train.dropna(subset=X_columns + [y_column], inplace=True)

#             # # Apply log transformation to the target variable
#             log_transformer = FunctionTransformer(func=np.log)
#             y_log_transformed = log_transformer.fit_transform(y.values.reshape(-1, 1))

#             # # Power transform the features
#             transformer = PowerTransformer(method='yeo-johnson')
#             ct = ColumnTransformer(
#                 transformers=[('transform', transformer, X_columns)],
#                 remainder='passthrough'
#             )
#             X_transformed = ct.fit_transform(X)

#             # # Scale the features
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X_transformed)

#             # # Outlier detection with One-Class SVM
#             clf = OneClassSVM(nu=0.1)
#             yhat = clf.fit_predict(X_scaled)

#             # # Filter out the outliers
#             mask = yhat != -1
#             X_inliers = X_scaled[mask]
#             y_inliers = y_log_transformed[mask]

#             # # # Split the data into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(X_inliers, y_inliers, test_size=0.2, random_state=42)

#             # # List of available models
#             models = {
#                 'LinearRegression': LinearRegression(),
#                 'Ridge': Ridge(),
#                 'Lasso': Lasso(),
#                 'ElasticNet': ElasticNet(),
#                 'KNeighborsRegressor': KNeighborsRegressor(),
#                 'RandomForestRegressor': RandomForestRegressor(random_state=random_tree),
#                 'GradientBoostingRegressor': GradientBoostingRegressor(random_state=random_tree),
#                 'AdaBoostRegressor': AdaBoostRegressor(random_state=random_tree),
#                 'BaggingRegressor': BaggingRegressor(random_state=random_tree),
#                 'ExtraTreesRegressor': ExtraTreesRegressor(random_state=random_tree),
#                 'DecisionTreeRegressor': DecisionTreeRegressor(random_state=random_tree),
#                 'SVR': SVR(),
#                 'MLPRegressor': MLPRegressor(random_state=random_tree),
#                 'XGBRegressor': XGBRegressor(random_state=random_tree),
#                 'LGBMRegressor': LGBMRegressor(random_state=random_tree),
#                 'TheilSenRegressor': TheilSenRegressor(),
#                 'HuberRegressor': HuberRegressor(),
#                 'KernelRidge': KernelRidge(),
#             }

#             # # Check if the selected model is available
#             model = models.get(model_type)
#             if not model:
#                 return JsonResponse({'error': 'Invalid model type selected'}, status=400)

#             # # Train the selected model
#             model.fit(X_train, y_train.ravel())
#             y_pred = model.predict(X_test)

#             # Calculate evaluation metrics
#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)
#             spearman_corr, _ = spearmanr(y_test.ravel(), y_pred)
#             print("lolllllllll",mse,r2,spearman_corr)
#             # Return the metrics
#             return JsonResponse({
#                 'model': model_type,
#                 'mse': mse,
#                 'r2': r2,
#                 'spearman_corr': spearman_corr,
#             })
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
        
import uuid
from datetime import datetime

import joblib
import shutil

# ... rest of your imports ...

@api_view(['POST'])
def train_model(request):
    if request.method == 'POST':
        try:
            # Parse the request body
            data = json.loads(request.body)
            file_name = data.get('file_name', '')  # Retrieve file name if available
            X_columns = data.get('X_columns', [])
            y_column = data.get('y_column', '')
            model_type = data.get('model_type', '')

            print("Training data:", data)
            folder_path = 'media/'
            csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            train_dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
                df.columns = df.columns.str.strip()
                train_dfs.append(df)

            df_train = pd.concat(train_dfs, axis=0)
            
            # Print some rows of the CSV file to the console
            print("CSV File Sample Data:")
            print(df_train.head())
            print("df train columns:", df_train.columns)
            
            if y_column not in df_train.columns:
                print("Target column not found")
                return JsonResponse({'error': f'{y_column} column not found in the data'}, status=400)
            
            # Define features and target
            X = df_train[X_columns]
            y = df_train[y_column]

            # Clean the data
            df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_train.dropna(subset=X_columns + [y_column], inplace=True)

            # Apply log transformation to the target variable
            log_transformer = FunctionTransformer(func=np.log)
            y_log_transformed = log_transformer.fit_transform(y.values.reshape(-1, 1))
            # print("the log transfoemed while ytaing ", y_log_transformed)
            # y_print=np.exp(y_log_transformed)
            # print("log untransformed",y_print)
            # Power transform the features
            transformer = PowerTransformer(method='yeo-johnson')
            ct = ColumnTransformer(
                transformers=[('transform', transformer, X_columns)],
                remainder='passthrough'
            )
            X_transformed = ct.fit_transform(X)

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_transformed)

            # Outlier detection with One-Class SVM
            clf = OneClassSVM(nu=0.1)
            yhat = clf.fit_predict(X_scaled)

            # Filter out the outliers
            # mask = yhat != -1
            # X_inliers = X_scaled[mask]
            # y_inliers = y_log_transformed[mask]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_log_transformed, test_size=0.2, random_state=42)

            # List of available models
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'RandomForestRegressor': RandomForestRegressor(random_state=42),
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
                'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
                'BaggingRegressor': BaggingRegressor(random_state=42),
                'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
                'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
                'SVR': SVR(),
                'MLPRegressor': MLPRegressor(random_state=42),
                'XGBRegressor': XGBRegressor(random_state=42),
                'LGBMRegressor': LGBMRegressor(random_state=42),
                'TheilSenRegressor': TheilSenRegressor(),
                'HuberRegressor': HuberRegressor(),
                'KernelRidge': KernelRidge(),
            }

            # Check if the selected model is available
            model = models.get(model_type)
            if not model:
                return JsonResponse({'error': 'Invalid model type selected'}, status=400)

            # Train the selected model
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_test)
            print(y_pred)
            y_pred_denorm = np.exp(y_pred)  # Inverse of log transformation
            y_test_denorm = np.exp(y_test)  # Inverse of log transformation
            print("denorlize")
            print(y_pred)
            print("before denormize ")
            print(y_test)
            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            spearman_corr, _ = spearmanr(y_test.ravel(), y_pred)
            print("Metrics:mse,r2; spearman_corr", mse, r2, spearman_corr)
          
            # **Save the trained model and preprocessing objects**
            # Create the 'models' directory if it doesn't exist
            models_dir = 'models'
            archive_dir = 'archive'

            # Ensure that the directories are not empty
            if not models_dir:
                raise ValueError("The models directory path is not defined.")
            if not archive_dir:
                raise ValueError("The archive directory path is not defined.")

            # Create the 'models' directory if it doesn't exist
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            # Proceed with your model saving logic
            model_filename = f'{models_dir}/{model_type}_model.pkl'
            joblib.dump(model, model_filename)

            # Save the preprocessing steps
            preprocessor_filename = f'{models_dir}/{model_type}_preprocessor.pkl'
            joblib.dump(processor, preprocessor_filename)

            # Ensure the archive directory exists
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)

            # Store the training info in a JSON file
            training_id = str(uuid.uuid4())
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            training_info = {
                'id': training_id,
                'action': 'training',
                'model_name': model_type,
                'files_used': csv_files,
                'X_columns': X_columns,
                'y_column': y_column,
                'metrics': {
                    'mse': mse,
                    'r2': r2,
                    'spearman_corr': spearman_corr,
                },
                'date': current_datetime,
            }

            json_dir = 'json'
            json_filename = f'{json_dir}/training_info.json'

            # Create the json directory if it doesn't exist
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)

            # Load existing data from the JSON file
            if os.path.exists(json_filename):
                with open(json_filename, 'r') as json_file:
                    try:
                        existing_data = json.load(json_file)
                    except json.JSONDecodeError:
                        # If the file is empty or invalid, start with an empty list
                        existing_data = []
            else:
                existing_data = []

            # Ensure existing_data is a list
            if not isinstance(existing_data, list):
                existing_data = []

            # Append new training info
            existing_data.append(training_info)

            # Write updated data back to the JSON file
            with open(json_filename, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
            # Move files to the archive folder
            # shutil.move(model_filename, os.path.join(archive_dir, f'{model_type}_model.pkl'))
            # shutil.move(preprocessor_filename, os.path.join(archive_dir, f'{model_type}_preprocessor.pkl'))
            # shutil.move(json_filename, os.path.join(archive_dir, f'{training_id}_training_info.json'))

            # Clean up the media folder
            media_folder = 'media'
            for file in os.listdir(media_folder):
                file_path = os.path.join(media_folder, file)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            return JsonResponse({
                'message': 'Model trained and saved successfully',
                'model': model_type,
                'mse': mse,
                'r2': r2,
                'spearman_corr': spearman_corr,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
# import os
# import joblib  # For saving/loading trained models
# from django.http import JsonResponse, HttpResponse
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
# from rest_framework.decorators import api_view
# import os
# import joblib  # For saving/loading trained models
# import json
# from django.http import JsonResponse
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PowerTransformer
# from rest_framework.decorators import api_view
# @api_view(['POST'])
# def predict_model(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             file_name = data.get('file_name', '')
#             X_columns = data.get('X_columns', [])
#             y_column = data.get('y_column', '')
#             model_type = data.get('model_type', '')
            
#             # # Load the uploaded CSV file
#             folder_path = 'media/'
#             folder_path = 'media/'
#             csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
#             # Load the CSV file data from a predefined location
#             # This is just an example; adjust the file path accordingly
             
#             train_dfs = []
#             for file in csv_files:
#                 df = pd.read_csv(file)
#                 df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
#                 df.columns = df.columns.str.strip()
#                 train_dfs.append(df)

#             df_train = pd.concat(train_dfs, axis=0)
            
#             # Print some rows of the CSV file to the console
#             print("CSV File Sample Data:")
#             print(df_train.head())            
             
#             # folder_path = 'media/'
#             # file_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
#             # print(file_path)
#             # # df = pd.read_csv(file_path)
#             # if not csv_files:
#             #     return JsonResponse({'error': 'No CSV file found in the folder'}, status=404)
            

#             # Print some rows of the CSV file to the console
#             # print("CSV File Sample Data:")
#             # print(df_train.head())
#             # if y_column not in df.columns:
#             #     return JsonResponse({'error': f'{y_column} column not found in the file'}, status=400)
            

#             # Denormalize (Inverse transform) the predictions
#             # predictions_transformed = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
#             # predictions_original = transformer.inverse_transform(predictions_transformed)

#             # Prepare the features
#             X = df[X_columns]
#             df.replace([np.inf, -np.inf], np.nan, inplace=True)
#             df.dropna(subset=X_columns, inplace=True)

#             # # Load the trained model from a file
#             model_path = f'models/{model_type}_model.pkl'
#             if not os.path.exists(model_path):
#                 return JsonResponse({'error': 'Model not found'}, status=404)
            
#             model = joblib.load(model_path)
#             log_transformer = FunctionTransformer(np.log1p)  # np.log1p for log(1 + x) to handle zero values
#             X_log_transformed = log_transformer.fit_transform(X)

#             # # Preprocessing steps
#             transformer = PowerTransformer(method='yeo-johnson')
#             X_transformed = transformer.fit_transform(X)
             
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X_transformed)

#             # # Predict missing values
#             predictions = model.predict(X_scaled)
#             # predictions_transformed = scaler.inverse_transform(predictions.reshape(-1, 1))
#             # predictions_original = transformer.inverse_transform(predictions_transformed)
#             # predictions_scaled = scaler.inverse_transform(predictions.reshape(-1, 1))  # Inverse scaling
#             # predictions_original = log_transformer.inverse_transform(predictions_scaled)  # Inverse log transformation


#             # # Add predictions to the original DataFrame
#             df[y_column + '_predicted'] = predictions

#             # # Save the result to a new CSV file
#             result_file_path = os.path.join(folder_path, 'predicted_results.csv')
#             df.to_csv(result_file_path, index=False)
#             return JsonResponse({'message': 'Prediction completed'})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#         #     return JsonResponse({'message': 'Prediction completed', 'file_path': result_file_path})
#         # except Exception as e:
#         #     return JsonResponse({'error': str(e)}, status=500)



# import os
# import joblib  # For saving/loading trained models
# import json
# from django.http import JsonResponse
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PowerTransformer
# from rest_framework.decorators import api_view

# @api_view(['POST'])
# def predict_model(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             file_name = data.get('file_name', '')
#             X_columns = data.get('X_columns', [])
#             y_column = data.get('y_column', '')
#             model_type = data.get('model_type', '')

#             # Load the uploaded CSV files
#             folder_path = 'media/'
#             csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

#             # Load all the CSV files and concatenate them
#             train_dfs = []
#             for file in csv_files:
#                 df = pd.read_csv(file)
#                 df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
#                 df.columns = df.columns.str.strip()
#                 train_dfs.append(df)

#             df_train = pd.concat(train_dfs, axis=0)

#             # Check if the target column exists in the DataFrame
#             if y_column not in df_train.columns:
#                 return JsonResponse({'error': f'{y_column} column not found in the file'}, status=400)

#             # Prepare the features (X) and target (y)
#             X = df_train[X_columns]
#             y = df_train[[y_column]]  # Ensure y is a DataFrame
#             print(y)
#             df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
#             df_train.dropna(subset=X_columns + [y_column], inplace=True)

#             # Load the trained model
#             model_path = f'models/{model_type}_model.pkl'
#             if not os.path.exists(model_path):
#                 return JsonResponse({'error': 'Model not found'}, status=404)
            
#             model = joblib.load(model_path)

#             # Preprocessing steps for the features (X)
#             transformer_X = PowerTransformer(method='yeo-johnson')
#             X_transformed = transformer_X.fit_transform(X)

#             scaler_X = StandardScaler()
#             X_scaled = scaler_X.fit_transform(X_transformed)

#             # Preprocessing steps for the target (y)
#             transformer_y = PowerTransformer(method='yeo-johnson')
#             y_transformed = transformer_y.fit_transform(y)

#             scaler_y = StandardScaler()
#             y_scaled = scaler_y.fit_transform(y_transformed)

#             # Predict on the scaled data
#             predictions_scaled = model.predict(X_scaled)

#             # Denormalize the predictions by reversing the transformations
#             predictions_transformed = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
#             predictions_original = transformer_y.inverse_transform(predictions_transformed)

#             # Add the denormalized predictions to the original DataFrame
#             df_train[y_column + '_predicted'] = predictions_original

#             # Save the result to a new CSV file
#             result_file_path = os.path.join(folder_path, 'predicted_results.csv')
#             df_train.to_csv(result_file_path, index=False)
            
#             return JsonResponse({'message': 'Prediction completed'})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=z500)
# import os
# import joblib  # For saving/loading trained models
# import json
# from django.http import JsonResponse
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PowerTransformer
# from rest_framework.decorators import api_view

# @api_view(['POST'])
# def predict_model(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             file_name = data.get('file_name', '')
#             X_columns = data.get('X_columns', [])
#             y_column = data.get('y_column', '')
#             model_type = data.get('model_type', '')

#             # Load the uploaded CSV files
#             folder_path = 'media/'
#             csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

#             # Load all the CSV files and concatenate them
#             train_dfs = []
#             for file in csv_files:
#                 df = pd.read_csv(file)
#                 df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
#                 df.columns = df.columns.str.strip()
#                 train_dfs.append(df)

#             df_train = pd.concat(train_dfs, axis=0)

#             # Check if the target column exists in the DataFrame
#             if y_column not in df_train.columns:
#                 return JsonResponse({'error': f'{y_column} column not found in the file'}, status=400)
            
#             # Prepare the features (X) and target (y)
#             X = df_train[X_columns]
#             y = df_train[[y_column]]  # Ensure y is a DataFrame
            
#             df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
#             df_train.dropna(subset=X_columns + [y_column], inplace=True)

#             # Load the trained model
#             model_path = f'models/{model_type}_model.pkl'
#             if not os.path.exists(model_path):
#                 return JsonResponse({'error': 'Model not found'}, status=404)
            
#             model = joblib.load(model_path)

#             # Preprocessing steps for the features (X)
#             transformer_X = PowerTransformer(method='yeo-johnson')
#             X_transformed = transformer_X.fit_transform(X)

#             scaler_X = StandardScaler()
#             X_scaled = scaler_X.fit_transform(X_transformed)

#             # Preprocessing steps for the target (y)
#             y_transformed = np.log(y)  # Apply log1p transformation (log(x))

#             scaler_y = StandardScaler()
#             y_scaled = scaler_y.fit_transform(y_transformed)
             
#             # Predict on the scaled data
#             predictions_scaled = model.predict(X_scaled)
#             print("predictedscaled",predictions_scaled)
#             # Denormalize the predictions by reversing the transformations
#             # predictions_transformed = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
#             # print("the values ddedkkkk")
#             # print( predictions_transformed)
#             predictions_original = np.exp(predictions_scaled)  # Apply inverse log transformation (exp(x) - 1)
#             print("the values dded")
#             print(predictions_original)
#             # Add the denormalized predictions to the original DataFrame
#             df_train[y_column + '_predicted'] = predictions_original
           
#             # Save the result to a new CSV file
#             result_file_path = os.path.join(folder_path, 'predicted_results.csv')
#             df_train.to_csv(result_file_path, index=False)
            
#             return JsonResponse({'message': 'Prediction completed'})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

# from django.http import JsonResponse, FileResponse
# from rest_framework.decorators import api_view
# import os
# import pandas as pd
# import numpy as np
# import joblib
# import json
# from sklearn.preprocessing import PowerTransformer, StandardScaler

# @api_view(['POST'])
# def predict_model(request):
#     if request.method == 'POST':
#         try:
#             # Parse the request body
#             data = json.loads(request.body)
#             file_name = data.get('file_name', '')  # Retrieve file name if available
#             X_columns = data.get('X_columns', [])
#             y_column = data.get('y_column', '')
#             model_type = data.get('model_type', '')

#             print("Training data:", data)
#             folder_path = 'media/'
#             csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
#             train_dfs = []
#             for file in csv_files:
#                 df = pd.read_csv(file)
#                 df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
#                 df.columns = df.columns.str.strip()
#                 train_dfs.append(df)

            

            
#             df_train = pd.concat(train_dfs, axis=0)

#             # Check if the target column exists in the DataFrame
#             # if y_column not in df_train.columns:
#             #     return JsonResponse({'error': f'{y_column} column not found in the file'}, status=400)
            
#             # Prepare the features (X) and target (y)
#             X = df_train[X_columns]
#             y = df_train[[y_column]]  # Ensure y is a DataFrame
            
#             df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
#             df_train.dropna(subset=X_columns + [y_column], inplace=True)

#             # Load the trained model
#             model_path = f'models/{model_type}_model.pkl'
#             if not os.path.exists(model_path):
#                 return JsonResponse({'error': 'Model not found'}, status=404)
            
#             model = joblib.load(model_path)

#             # Preprocessing steps for the features (X)
#             transformer_X = PowerTransformer(method='yeo-johnson')
#             X_transformed = transformer_X.fit_transform(X)

#             scaler_X = StandardScaler()
#             X_scaled = scaler_X.fit_transform(X_transformed)

#             # Preprocessing steps for the target (y)
#             y_transformed = np.log(y)  # Apply log1p transformation (log(x))

#             scaler_y = StandardScaler()
#             y_scaled = scaler_y.fit_transform(y_transformed)
             
#             # Predict on the scaled data
#             predictions_scaled = model.predict(X_scaled)

#             # Denormalize the predictions by reversing the transformations
#             predictions_original = np.exp(predictions_scaled)  # Apply inverse log transformation

#             # Add the denormalized predictions to the original DataFrame
#             df_train[y_column + '_predicted'] = predictions_original
            
#             # Save the result to a new CSV file
#             result_file_path = os.path.join(folder_path, 'predicted_results.csv')
#             df_train.to_csv(result_file_path, index=False)
#             print(result_file_path)
#             # Send the predicted results CSV file as a response
#             response = FileResponse(open(result_file_path, 'rb'), as_attachment=True, filename='predicted_results.csv')
#             print(response)
#             return response

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

@api_view(['POST'])
def predict_model(request):
    if request.method == 'POST':
        try:
            # Parse the request body
            data = json.loads(request.body)
            file_name = data.get('file_name', '')  # Retrieve file name if available
            X_columns = data.get('X_columns', [])
            y_column = data.get('y_column', '')  # Column to predict (but not use in X)
            model_type = data.get('model_type', '')

            print("Training data:", data)
            folder_path = 'media/'
            csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            train_dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
                df.columns = df.columns.str.strip()
                train_dfs.append(df)

            # Concatenate all the CSV files into one DataFrame
            df_train = pd.concat(train_dfs, axis=0)

            # Check if all the required X_columns exist in the DataFrame
            missing_columns = [col for col in X_columns if col not in df_train.columns]
            if missing_columns:
                return JsonResponse({'error': f'Missing columns: {", ".join(missing_columns)}'}, status=400)

            # Prepare the features (X), but exclude the target column (y_column)
            X = df_train[X_columns]

            # Handle missing values and infinities in the input data
            df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_train.dropna(subset=X_columns, inplace=True)

            # Load the trained model
            model_path = f'models/{model_type}_model.pkl'
            if not os.path.exists(model_path):
                return JsonResponse({'error': 'Model not found'}, status=404)

            model = joblib.load(model_path)

            # Preprocessing steps for the features (X)
            transformer_X = PowerTransformer(method='yeo-johnson')
            X_transformed = transformer_X.fit_transform(X)

            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_transformed)

            # Predict the target (y_column) on the scaled data (X_scaled)
            predictions_scaled = model.predict(X_scaled)

            # Denormalize the predictions (if any transformation was applied during training)
            predictions_original = np.exp(predictions_scaled)  # Apply inverse log transformation (if log was used)

            # Add the denormalized predictions as a new column in the DataFrame
            df_train[y_column + '_predicted'] = predictions_original

            # Save the result to a new CSV file
            result_file_path = os.path.join(folder_path, 'predicted_results.csv')
            df_train.to_csv(result_file_path, index=False)

            # Send the predicted results CSV file as a response
            response = FileResponse(open(result_file_path, 'rb'), as_attachment=True, filename='predicted_results.csv')
            return response

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)



import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.http import JsonResponse, FileResponse

@api_view(['POST'])
def normalize_data(request):
    if request.method == 'POST':
        try:
            # Define the folder path and read CSV files
            folder_path = 'media/'
            csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            train_dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
                df.columns = df.columns.str.strip()
                train_dfs.append(df)

            # Concatenate all the dataframes
            df_train = pd.concat(train_dfs, axis=0)
            
            # Define columns to use
            X_columns = ['A', 'B'] 
            Y_columns = ['C'] 
            X = df_train[X_columns]
            y = df_train[Y_columns]

            # Apply log transformation to the target variable
            log_transformer = FunctionTransformer(func=np.log)
            y_log_transformed = log_transformer.fit_transform(y.values.reshape(-1, 1))

            # Power transform the features
            transformer = PowerTransformer(method='yeo-johnson')
            ct = ColumnTransformer(
                transformers=[('transform', transformer, X_columns)],
                remainder='passthrough'
            )
            X_transformed = ct.fit_transform(X)

            # Prepare the transformed data as dictionaries with column names
            X_transformed_dict = pd.DataFrame(X_transformed, columns=X_columns).to_dict(orient='list')
            y_log_transformed_list = pd.DataFrame(y_log_transformed, columns=Y_columns).to_dict(orient='list')

            # Return the transformed data with column names in the response
            return JsonResponse({
                'message': 'Data normalized successfully',
                'X_transformed': X_transformed_dict,
                'y_log_transformed': y_log_transformed_list
            })

        except Exception as e:
            # Catch and print the exception for debugging
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

@api_view(['POST'])
def remove_outliers(request):
    if request.method == 'POST':
        try:
            folder_path = 'media/'
            csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            train_dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                df.columns = df.columns.str.replace(r'\\t+', ' ', regex=True)
                df.columns = df.columns.str.strip()
                train_dfs.append(df)

            df_train = pd.concat(train_dfs, axis=0)
            y_column = ['C'] 
            X_columns = ['A', 'B']
            Y_columns = ['C']
            X = df_train[X_columns]
            y = df_train[Y_columns]

            # Clean the data: replace inf values with NaN and drop rows with NaN in the specified columns
            df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_train.dropna(subset=X_columns + Y_columns, inplace=True)

            # Apply log transformation to the target variable
            log_transformer = FunctionTransformer(func=np.log)
            y_log_transformed = log_transformer.fit_transform(y.values.reshape(-1, 1))

            # Power transform the features
            transformer = PowerTransformer(method='yeo-johnson')
            ct = ColumnTransformer(
                transformers=[('transform', transformer, X_columns)],
                remainder='passthrough'
            )
            X_transformed = ct.fit_transform(X)

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_transformed)

            # Outlier detection with One-Class SVM
            clf = OneClassSVM(nu=0.1)
            yhat = clf.fit_predict(X_scaled)

            # If everything went well, return success response
            return JsonResponse({
                'message': 'Data normalized successfully',
            })

        except Exception as e:
            # Catch and print the exception for debugging
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)



import os
import json
from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view

@api_view(['GET'])
def get_training_info(request):
    try:
        # Define the path to the JSON file
        json_dir = 'json'
        json_filename = os.path.join(json_dir, 'training_info.json')

        # Check if the JSON file exists
        if not os.path.exists(json_filename):
            return JsonResponse({'error': 'JSON file not found'}, status=404)

        # Read the JSON file
        with open(json_filename, 'r') as json_file:
            try:
                training_data = json.load(json_file)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Error reading JSON file'}, status=500)

        # Return the JSON data as a response
        return JsonResponse(training_data, safe=False)

    except Exception as e:
        # Handle any other exceptions
        return JsonResponse({'error': str(e)}, status=500)

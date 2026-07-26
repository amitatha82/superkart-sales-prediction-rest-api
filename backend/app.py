
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Initialize the Flask application
superkart_prediction_api = Flask("Superkart Sales Predictor")

# Define the path where the model is saved within the container
MODEL_PATH = "Superkart_prediction_model_v1_0.joblib"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle case where model loading fails

# Define a route for the home page (GET request)
@superkart_prediction_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Superkart Price Prediction API!"


# Define an endpoint for single prediction (POST request)
@superkart_prediction_api.post('/v1/predict')
def predict_sales():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        json_data = request.get_json()
        if not isinstance(json_data, list):
            json_data = [json_data]

        query_df = pd.DataFrame(json_data)

        # Rename the store age column when the compatibility alias is provided
        if 'Store_Age' not in query_df.columns and 'Store_Age_Years' in query_df.columns:
            query_df = query_df.rename(columns={'Store_Age_Years': 'Store_Age'})

        # Normalize inconsistent sugar content labels
        if 'Product_Sugar_Content' in query_df.columns:
            query_df['Product_Sugar_Content'] = query_df['Product_Sugar_Content'].replace({'reg': 'Regular'})

        # Expected columns based on the retrained model data
        expected_columns = [
            'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_MRP', 'Store_Size', 'Store_Location_City_Type',
            'Store_Type', 'Store_Age', 'Product_Type_Category',
            'Product_Id_char'
        ]
        missing_columns = [col for col in expected_columns if col not in query_df.columns]
        if missing_columns:
            raise ValueError(f"The following required fields are missing: {missing_columns}")

        # Convert numeric fields and reject missing or non-finite values
        numeric_columns = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age']
        query_df[numeric_columns] = query_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        if query_df[numeric_columns].isna().any().any() or not np.isfinite(query_df[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError("Numeric fields must contain finite numeric values.")

        # Reorder columns to match the training data order
        query_df = query_df[expected_columns]

        # Make predictions using the loaded pipeline model
        predictions = model.predict(query_df)

        # Convert predictions to a list for JSON serialization
        return jsonify({'predicted_sales': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define an endpoint for batch prediction (POST request)
@superkart_prediction_api.post('/v1/predictbatch')
def predict_sales_batch():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the CSV file into a Pandas DataFrame
        input_data = pd.read_csv(file)

        # Rename the store age column when the compatibility alias is provided
        if 'Store_Age' not in input_data.columns and 'Store_Age_Years' in input_data.columns:
            input_data = input_data.rename(columns={'Store_Age_Years': 'Store_Age'})

        # Normalize inconsistent sugar content labels
        if 'Product_Sugar_Content' in input_data.columns:
            input_data['Product_Sugar_Content'] = input_data['Product_Sugar_Content'].replace({'reg': 'Regular'})

        # Expected columns based on the retrained model data
        expected_columns = [
            'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
            'Product_MRP', 'Store_Size', 'Store_Location_City_Type',
            'Store_Type', 'Store_Age', 'Product_Type_Category',
            'Product_Id_char'
        ]
        missing_columns = [col for col in expected_columns if col not in input_data.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing: {missing_columns}")

        # Convert numeric fields and reject missing or non-finite values
        numeric_columns = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age']
        input_data[numeric_columns] = input_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        if input_data[numeric_columns].isna().any().any() or not np.isfinite(input_data[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError("Numeric columns must contain finite numeric values.")

        # Reorder columns to match the training data order
        input_data = input_data[expected_columns]

        # Make predictions for all entries in the DataFrame
        predictions_batch = model.predict(input_data).tolist()

        # Return the predictions list as a JSON response
        return jsonify({'predicted_sales_batch': predictions_batch})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_prediction_api.run(debug=True)

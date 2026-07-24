
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
    return "Welcome to the Superkart Sales Prediction API!"

# Define an endpoint for single prediction (POST request)
@superkart_prediction_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales as a JSON response.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        json_data = request.json

        # Ensure json_data is a list of dictionaries, even for single predictions,
        # to properly create a DataFrame
        if not isinstance(json_data, list):
            json_data = [json_data]

        query_df = pd.DataFrame(json_data)

        # Expected columns based on the training data
        expected_columns = [
            'Product_Id', 'Product_Weight', 'Product_Sugar_Content',
            'Product_Allocated_Area', 'Product_Type', 'Product_MRP',
            'Store_Id', 'Store_Size', 'Store_Location_City_Type',
            'Store_Type', 'Store_Age'
        ]

        # Reorder columns to match the training data order
        # This assumes the input JSON contains all expected columns.
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
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details for multiple entries
    and returns the predicted sales as a list in the JSON response.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the CSV file into a Pandas DataFrame
        input_data = pd.read_csv(file)

        # Expected columns based on the training data
        expected_columns = [
            'Product_Id', 'Product_Weight', 'Product_Sugar_Content',
            'Product_Allocated_Area', 'Product_Type', 'Product_MRP',
            'Store_Id', 'Store_Size', 'Store_Location_City_Type',
            'Store_Type', 'Store_Age'
        ]

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

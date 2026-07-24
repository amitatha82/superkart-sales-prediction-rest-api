
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
# This should point to the service name 'backend' and the exposed port (5000) within the Docker network
BACKEND_URL = "http://backend:5000"

# Set the title of the Streamlit app
st.title("Superkart Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Define the expected columns and their types/options for user input
expected_columns = [
    'Product_Id', 'Product_Weight', 'Product_Sugar_Content',
    'Product_Allocated_Area', 'Product_Type', 'Product_MRP',
    'Store_Id', 'Store_Size', 'Store_Location_City_Type',
    'Store_Type', 'Store_Age'
]

# Collect user input for product and store features
product_id = st.text_input("Product ID", value="FD6114") # Example ID
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66, format="%.2f")
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
product_type = st.selectbox("Product Type", [
    'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene',
    'Snack Foods', 'Meat', 'Hard Drinks', 'Fruits and Vegetables', 'Household',
    'Breakfast', 'Soft Drinks', 'Starchy Foods', 'Others', 'Seafood', 'Bread'
])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08, format="%.2f")
store_id = st.text_input("Store ID", value="OUT004") # Example ID
store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 2', 'Tier 1', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type2', 'Departmental Store', 'Supermarket Type1', 'Food Mart'])
store_age = st.number_input("Store Age (years since establishment)", min_value=0, value=11)

# Create a dictionary from the collected inputs
input_data = {
    'Product_Id': product_id,
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_Type': product_type,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Store_Age': store_age
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict Sales", type="primary"):
    try:
        # Send data to Flask API's single prediction endpoint
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data)

        if response.status_code == 200:
            prediction = response.json()['predicted_sales'][0] # Assuming single prediction returns a list with one item
            st.success(f"Predicted Sales: ${prediction:,.2f}")
        else:
            st.error(f"Error from backend: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the backend is running.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch Sales", type="primary"):
        try:
            # Send file to Flask API's batch prediction endpoint
            files = {'file': uploaded_file.getvalue()}
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)

            if response.status_code == 200:
                predictions = response.json()['predicted_sales_batch'] # Assuming batch prediction returns a list
                st.success("Batch predictions completed!")
                # Display predictions in a DataFrame for better readability
                predictions_df = pd.DataFrame(predictions, columns=["Predicted_Sales"])
                st.dataframe(predictions_df)
            else:
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

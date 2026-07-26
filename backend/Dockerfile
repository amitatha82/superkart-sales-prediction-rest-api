# Use a slim Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY deployment_files/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the saved model and the Flask application code
COPY deployment_files/Superkart_prediction_model_v1_0.joblib .
COPY deployment_files/app.py .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define the command to start the application using Gunicorn with 4 worker processes
# - `-w 4`: Uses 4 worker processes for handling requests
# - `-b 0.0.0.0:5000`: Binds the server to port 5000 on all network interfaces
# - `app:superkart_prediction_api`: Runs the Flask app (assuming `app.py` contains the Flask instance named `superkart_prediction_api`)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_prediction_api"]

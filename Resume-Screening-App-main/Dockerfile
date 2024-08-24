# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model files into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set an environment variable for the Streamlit app
ENV STREAMLIT_APP_FILE=2.py

# Command to run the application
CMD ["streamlit", "run", "2.py"]
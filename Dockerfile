# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to install dependencies
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg and OpenGL (libgl1) for OpenCV
RUN apt-get update && apt-get install -y ffmpeg libgl1 espeak-ng

# Copy the rest of the application code to the container
COPY . .

# Install ffmpeg (if needed)
# RUN apt-get update && apt-get install -y ffmpeg

# Expose the port the app runs on
# EXPOSE 8000

# Command to run the FastAPI app with optional arguments
CMD ["python", "app.py", "--server", "--vision", "gpt-4-vision-preview", "--assistant", "egov"]

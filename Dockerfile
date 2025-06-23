# Use a lightweight Python image
FROM python:3.11.3-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (code, weights, video, templates, etc.)
COPY . .

# Expose port 5000 for the web server
EXPOSE 5000

# Run Flask app using Gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

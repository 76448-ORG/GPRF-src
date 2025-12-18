# Use a slim, stable Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the Uvicorn server (port 8001 defined in docker-compose)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
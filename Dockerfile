# # Use an official Python runtime as a parent image
# FROM python:3.12-slim

# # Set the working directory in the container
# WORKDIR /usr/src/app

# # Copy the current directory contents into the container at /usr/src/app
# COPY . .

# # Install python-dotenv to load environment variables from .env file
# RUN pip install --no-cache-dir -r requirements.txt


# Run main.py when the container launches
# CMD ["python", "bot.py"]

FROM python:3.12-slim

WORKDIR /usr/src/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Render
EXPOSE 8080

# Run bot (Flask will start automatically inside bot.py)
CMD ["python", "bot.py"]

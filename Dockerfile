# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /main

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit configuration
RUN mkdir -p ~/.streamlit
RUN echo "\
[server]\n\
enableCORS = false\n\
headless = true\n\
port = 8501\n\
" > ~/.streamlit/config.toml

# Copy the application code into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Define the entry point for the container
CMD ["streamlit", "run", "main.py", "--server.port=8501"]

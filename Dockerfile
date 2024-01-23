# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV POETRY_VERSION 1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    # Add any other libraries or dependencies your specific package might need
&& rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy the project files into the container
COPY pyproject.toml poetry.lock $APP_HOME/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of your application's code
COPY . $APP_HOME

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run --server.port 8501 --browser.serverAddress 0.0.0.0 01_💬_Chat_With_Documents.py

# Use an appropriate base image, e.g., python:3.10-slim
FROM python:3.10-slim

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /app

# Copy your poetry.lock file and pyproject.toml file
COPY poetry.lock pyproject.toml /app/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Copy your application code into the container
COPY . /app/

EXPOSE 8080

CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--port", "8080"]
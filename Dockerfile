FROM python:3.10-slim-bookworm
WORKDIR /app

# Copy code and .env
COPY . /app

# Install dependencies
RUN apt update -y && apt install awscli -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command
CMD ["python3", "app.py"]
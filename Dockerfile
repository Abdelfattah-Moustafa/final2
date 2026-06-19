FROM python:3.12-slim

# System dependencies for OpenCV / MediaPipe.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

# Apply migrations + load data, then start the server (see scripts/entrypoint.sh).
CMD ["./scripts/entrypoint.sh"]

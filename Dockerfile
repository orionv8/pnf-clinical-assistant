FROM python:3.9-slim

# Create a non-privileged user
RUN addgroup --system appuser && adduser --system --group appuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Change ownership of the app directory to the new user
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user
USER appuser

ENV PORT=8080

# Serve the chatbot UI via FastAPI + uvicorn
# GET  /          -> index.html (chatbot frontend)
# GET  /health    -> liveness probe
# POST /api/pnf/ask -> PNF drug search endpoint
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

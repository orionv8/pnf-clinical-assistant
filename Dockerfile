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
CMD ["streamlit", "run", "app.py", "--server.port", "8080"]

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

# Expose port Hugging Face expects
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "7860"]
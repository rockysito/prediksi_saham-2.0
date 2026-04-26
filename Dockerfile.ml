FROM python:3.10.0-slim

WORKDIR /app

# Install ML dependencies
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

COPY . .

# Expose API and Streamlit ports
EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

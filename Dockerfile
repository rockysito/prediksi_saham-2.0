FROM python:3.10-slim

# Persyaratan wajib dari Hugging Face: Buat user non-root bernama "user" dengan id 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirement list
COPY --chown=user requirements_ml.txt requirements_scraper.txt ./

# Install semua dependensi
RUN pip install --no-cache-dir -r requirements_ml.txt \
    && pip install --no-cache-dir -r requirements_scraper.txt \
    && pip install --no-cache-dir plotly vaderSentiment

# Copy seluruh file project
COPY --chown=user . .

# Hugging Face Spaces mengarahkan public web ke port 7860
EXPOSE 7860

# Menjalankan script Python yang me-run FastAPI, Streamlit, dan Scheduler sekaligus
CMD ["python", "start.py"]

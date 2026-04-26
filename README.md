# 📈 AI Trade Pro: Algorithmic Stock Prediction System

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/godless1405/prediksi-saham-pro)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?style=flat-square)

AI Trade Pro is an advanced, production-ready algorithmic trading analysis platform. It combines **Deep Learning (LSTM)**, **Statistical Forecasting (ARIMA)**, and **Natural Language Processing (VADER Sentiment Analysis)** to predict the short-term trajectory of major global stocks.

👉 **[TRY THE LIVE WEB APPLICATION HERE!](https://huggingface.co/spaces/godless1405/prediksi-saham-pro)**

---

## 🌟 Key Features

1. **AI Confluence Signal Generation**
   The system doesn't just output raw numbers; it produces an automated Trading Recommendation (`STRONG BUY`, `HOLD`, `STRONG SELL`). It calculates this by cross-referencing the quantitative price prediction from the AI models against the qualitative global market sentiment.

2. **Real-Time NLP Sentiment Engine**
   Automatically scrapes the top 5 most recent global news headlines regarding the selected asset from Yahoo Finance, parsing the text using VADER NLP to gauge whether the current global macroeconomic mood is Bullish or Bearish.

3. **Hybrid Machine Learning Architecture**
   - **LSTM (Long Short-Term Memory):** A deep learning neural network designed to capture complex, non-linear sequences in historical price action.
   - **ARIMA (AutoRegressive Integrated Moving Average):** A highly rigorous statistical model that computes linear regressions and moving averages instantly.

4. **Dynamic Watchlist Methodology**
   To prevent server timeouts and simulate institutional trading environments, the system focuses on a curated "Watchlist" of major tech stocks (`AAPL`, `TSLA`, `MSFT`, `NVDA`, `GOOGL`, `BBCA.JK`). 

## 🏗️ System Architecture

This project was built from scratch utilizing a modern Microservices architecture, allowing the frontend and backend to scale independently.

```text
prediksi_saham_2.0/
├── api/
│   └── main.py              # FastAPI Backend (REST endpoints for prediction & sentiment)
├── frontend/
│   └── app.py               # Streamlit Premium Dashboard (User Interface)
├── src/
│   ├── models/              # Object-Oriented LSTM & ARIMA model wrappers
│   └── pipelines/           
│       └── scheduler.py     # Background worker for nightly model retraining
├── Dockerfile               # Configuration for Hugging Face Spaces deployment
└── start.py                 # Unified entrypoint to run API & UI concurrently
```

## 🚀 How to Run Locally

If you wish to run this system on your local machine for further development:

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd prediksi_saham_2.0
   ```

2. **Create a Virtual Environment (Python 3.10 is recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements_ml.txt
   pip install -r requirements_scraper.txt
   pip install plotly vaderSentiment
   ```

4. **Start the Application:**
   Run the unified deployment script used for the cloud:
   ```bash
   python start.py
   ```
   *The Streamlit UI will be available at `http://localhost:7860` and the FastAPI docs at `http://localhost:8000/docs`.*

## 🧠 The Journey: From Notebook to Production
This version (2.0) is a massive refactor of personal Jupyter Notebook research files. The original unstructured scripts were decomposed into modular Python Classes (`StockARIMAModel`, `StockLSTMModel`), encapsulated in RESTful APIs, and deployed via Docker to the cloud, proving the transition from *Data Science Experimentation* to *Software Engineering Deployment*.

---
*Disclaimer: This software is for educational, portfolio, and research purposes only. Do not use algorithmic predictions as your sole financial or investment advice.*

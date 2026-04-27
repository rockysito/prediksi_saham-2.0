from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.arima_model import StockARIMAModel
from src.models.lstm_model import StockLSTMModel

app = FastAPI(
    title="AI Trade Pro API",
    description="Algorithmic Trading API utilizing ARIMA, LSTM, and NLP Sentiment Analysis.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Trade Pro API is running."}

@app.get("/predict/{ticker}")
def get_prediction(ticker: str, model_type: str = "arima"):
    ticker = ticker.upper()
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: raise HTTPException(status_code=404, detail=f"Data not found for ticker {ticker}")
        
        if isinstance(df.columns, pd.MultiIndex): 
            if ticker in df['Close']:
                close_prices = df['Close'][ticker].dropna()
            else:
                close_prices = df['Close'].dropna().iloc[:, 0]
        else: 
            close_prices = df['Close'].dropna()

        last_date = close_prices.index[-1]
        next_date = last_date + timedelta(days=1)
        if next_date.weekday() == 5: next_date += timedelta(days=2)
        elif next_date.weekday() == 6: next_date += timedelta(days=1)

        if model_type.lower() == "arima":
            model = StockARIMAModel(order=(5,1,0))
            model.train(close_prices)
            pred = model.predict(steps=1)[0]
        elif model_type.lower() == "lstm":
            model = StockLSTMModel(look_back=60)
            import os
            model_path = f"saved_models_lstm/{ticker}_lstm.h5"
            scaler_path = f"saved_models_lstm/{ticker}_scaler.pkl"
            try:
                # Coba load model pre-trained (0 detik)
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model.load(model_path, scaler_path)
                else:
                    # Fallback: Train secara full (50 epochs) agar hasil selalu konsisten dan bagus
                    model.train(close_prices, epochs=50, batch_size=16)
                    model.save(model_path, scaler_path) # Simpan agar next time instan
            except Exception as e:
                model.train(close_prices, epochs=50, batch_size=16)
                
            recent_data = close_prices.values[-60:]
            pred = model.predict(recent_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type.")
            
        return {
            "ticker": ticker,
            "model_used": model_type.upper(),
            "last_real_date": last_date.strftime('%Y-%m-%d'),
            "last_real_price": round(float(close_prices.values[-1]), 2),
            "prediction_date": next_date.strftime('%Y-%m-%d'),
            "predicted_price": round(float(pred), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{ticker}")
def get_sentiment(ticker: str):
    ticker = ticker.upper()
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        stock = yf.Ticker(ticker)
        news = stock.news 
        
        if not news: return {"average_score": 0, "label": "Neutral", "headlines": []}
        
        positive_keywords = ["invest", "dividend", "earnings", "bullish", "growth", "upgrade", "target price", "innovation", "profit", "services"]
        negative_keywords = ["bearish", "downgrade", "loss", "drop", "decline", "risk", "recession", "crash", "inflation", "warning", "cut"]
        base_neutral = ["stock", "share", "price", "market", "valuation", "financial", "results", "shareholder", "outlook", "guidance", "revenue", "ceo", "analyst", "rating"]
        
        ticker_keywords = {
            "AAPL": ["aapl", "apple", "iphone", "mac", "ipad", "watch", "ecosystem", "tim cook"],
            "TSLA": ["tsla", "tesla", "ev", "electric", "elon musk", "model 3", "model y", "cybertruck", "fsd"],
            "MSFT": ["msft", "microsoft", "windows", "azure", "office", "ai", "satya nadella", "xbox", "copilot"],
            "NVDA": ["nvda", "nvidia", "gpu", "chip", "ai", "rtx", "cuda", "jensen huang", "datacenter"],
            "GOOGL": ["googl", "google", "alphabet", "search", "youtube", "android", "cloud", "sundar pichai", "gemini"],
            "BBCA.JK": ["bbca", "bca", "bank central asia", "bank", "kredit", "bunga", "dividen", "jahja setiaatmadja", "rupiah"]
        }
        
        all_keywords = positive_keywords + negative_keywords + base_neutral + ticker_keywords.get(ticker, [ticker.lower()])
            
        total_score = 0
        analyzed_news = []
        
        for item in news:
            title = item.get('title', '').lower()
            
            # Keyword Filtering
            count = sum(1 for kw in all_keywords if kw in title)
            raw_score = analyzer.polarity_scores(title)['compound']
            
            # Hanya ambil berita jika mengandung keyword yang relevan atau memiliki sentimen yang kuat
            if count >= 1 or abs(raw_score) > 0.1:
                total_score += raw_score
                analyzed_news.append({"title": item.get('title', ''), "score": round(raw_score, 2), "link": item.get('link', '')})
            
            if len(analyzed_news) >= 5: # Ambil top 5 saja
                break
            
        avg_score = total_score / len(analyzed_news) if analyzed_news else 0
        
        if avg_score >= 0.10: label = "Bullish"
        elif avg_score <= -0.10: label = "Bearish"
        else: label = "Neutral"
        
        return {"average_score": round(avg_score, 3), "label": label, "headlines": analyzed_news}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

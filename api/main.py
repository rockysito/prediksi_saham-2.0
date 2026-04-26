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
            model.train(close_prices, epochs=1, batch_size=32)
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
            
        total_score = 0
        analyzed_news = []
        
        for item in news[:5]:
            title = item.get('title', '')
            score = analyzer.polarity_scores(title)['compound']
            total_score += score
            analyzed_news.append({"title": title, "score": round(score, 2), "link": item.get('link', '')})
            
        avg_score = total_score / len(analyzed_news) if analyzed_news else 0
        
        if avg_score >= 0.10: label = "Bullish"
        elif avg_score <= -0.10: label = "Bearish"
        else: label = "Neutral"
        
        return {"average_score": round(avg_score, 3), "label": label, "headlines": analyzed_news}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

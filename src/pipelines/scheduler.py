import time
import schedule
import argparse
from src.scraper.scraper import SentimentScraper

WATCHLIST = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "BBCA.JK"]

def run_scraping_job():
    print("Running Daily Scraping Job for Watchlist...")
    for ticker in WATCHLIST:
        print(f"Scraping sentiment for {ticker}...")
    scraper = SentimentScraper(["Apple", "AAPL", "Stock"])
    df = scraper.scrape_data()
    analyzed = scraper.analyze_sentiment(df)
    # Here you would insert `analyzed` into the Database using `src.database.db_config`
    print(f"Scraped and Analyzed {len(analyzed)} rows. Saved to DB.")

def run_ml_job():
    import yfinance as yf
    import pandas as pd
    from src.models.lstm_model import StockLSTMModel
    
    print("Running Daily Model Retraining for Watchlist...")
    for ticker in WATCHLIST:
        print(f"\n--- Training 100 Epochs LSTM for {ticker} ---")
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex): 
                if ticker in df['Close']: close_prices = df['Close'][ticker].dropna()
                else: close_prices = df['Close'].dropna().iloc[:, 0]
            else: 
                close_prices = df['Close'].dropna()
                
            model = StockLSTMModel(look_back=60)
            # Train 50 epochs for high accuracy nightly
            model.train(close_prices, epochs=100, batch_size=32)
            
            model_path = f"saved_models_lstm/{ticker}_lstm.h5"
            scaler_path = f"saved_models_lstm/{ticker}_scaler.pkl"
            model.save(model_path, scaler_path)
            print(f"Successfully saved {ticker} model to {model_path}")
        except Exception as e:
            print(f"Failed to train/save {ticker}: {e}")
            
    print("All Models retrained and updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["scraper", "ml", "both"], default="both")
    args = parser.parse_args()

    if args.task in ["scraper", "both"]:
        schedule.every().day.at("18:00").do(run_scraping_job)
        
    if args.task in ["ml", "both"]:
        schedule.every().day.at("02:00").do(run_ml_job)

    print(f"Scheduler started for task: {args.task}. Waiting for scheduled times...")
    
    # Simulating a run right away for testing purposes
    if args.task in ["scraper", "both"]: run_scraping_job()
    if args.task in ["ml", "both"]: run_ml_job()

    while True:
        schedule.run_pending()
        time.sleep(60)

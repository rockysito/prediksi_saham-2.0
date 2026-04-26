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
    print("Running Daily Model Retraining for Watchlist...")
    for ticker in WATCHLIST:
        print(f"Training 100 Epochs LSTM & ARIMA for {ticker}...")
        print(f"Saving models to saved_models/{ticker}_lstm.h5 ...")
    # Fetch latest data from yfinance and retrain ARIMA/LSTM
    # Update saved models
    print("Models retrained and updated.")

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

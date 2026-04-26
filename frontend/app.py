import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="AI Trade Pro", layout="wide", page_icon="📈")

# Custom CSS for a premium look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.2rem; color: #B0BEC5; margin-bottom: 30px; }
    div[data-testid="metric-container"] {
        background-color: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for Settings
st.sidebar.markdown("### ⚙️ Konfigurasi Saham")
WATCHLIST = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "BBCA.JK"]
selected_ticker = st.sidebar.selectbox("Simbol Ticker Saham (Watchlist):", WATCHLIST, index=0)
st.sidebar.markdown("*Catatan: AI hanya memprediksi saham di dalam Watchlist ini karena model LSTM dilatih khusus secara intensif setiap malam.*")

st.markdown(f'<p class="main-header">📈 AI Trade Pro: {selected_ticker}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Algorithmic Trading Dashboard powered by ARIMA, LSTM & Real-Time NLP Sentiment.</p>', unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    return df

@st.cache_data(ttl=300)
def fetch_sentiment(ticker):
    try:
        response = requests.get(f"http://127.0.0.1:8000/sentiment/{ticker}", timeout=15)
        if response.status_code == 200: return response.json()
    except Exception: return None

def fetch_prediction(model, ticker):
    try:
        response = requests.get(f"http://127.0.0.1:8000/predict/{ticker}?model_type={model.lower()}", timeout=60)
        if response.status_code == 200: return response.json()
    except Exception: return None

df = fetch_stock_data(selected_ticker)
sent_res = fetch_sentiment(selected_ticker)

tab1, tab2 = st.tabs(["📊 Technical Analysis & Market Sentiment", "🤖 AI Prediction & Signal"])

with tab1:
    col1, col2 = st.columns([2.5, 1.5])
    with col1:
        st.markdown(f"### Interactive Price Action: {selected_ticker} (6 Months)")
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                if selected_ticker in df['Close']:
                    close_prices = df['Close'][selected_ticker]
                    open_prices = df['Open'][selected_ticker]
                    high_prices = df['High'][selected_ticker]
                    low_prices = df['Low'][selected_ticker]
                else:
                    close_prices = df['Close'].iloc[:, 0]
                    open_prices = df['Open'].iloc[:, 0]
                    high_prices = df['High'].iloc[:, 0]
                    low_prices = df['Low'].iloc[:, 0]
            else:
                close_prices = df['Close']
                open_prices = df['Open']
                high_prices = df['High']
                low_prices = df['Low']

            df_chart = pd.DataFrame({'Close': close_prices})
            df_chart['MA20'] = df_chart['Close'].rolling(window=20).mean()
            df_chart['MA50'] = df_chart['Close'].rolling(window=50).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=close_prices.index, open=open_prices, high=high_prices, low=low_prices, close=close_prices, name='Price'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='#FFA726', width=2), name='MA 20'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], line=dict(color='#29B6F6', width=2), name='MA 50'))
            
            fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0), height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Gagal menarik data untuk ticker {selected_ticker}. Pastikan simbol benar.")

    with col2:
        st.markdown("### Real-Time NLP Sentiment")
        if sent_res:
            st.metric(label="VADER Compound Score", value=sent_res['average_score'], delta=sent_res['label'])
            st.markdown("#### Market Drivers (Top 5 Headlines):")
            if len(sent_res['headlines']) > 0:
                for news in sent_res['headlines']:
                    color = "🟢" if news['score'] > 0 else ("🔴" if news['score'] < 0 else "⚪")
                    st.markdown(f"- {color} [{news['title']}]({news['link']}) *(Score: {news['score']})*")
            else:
                st.info(f"Belum ada berita terbaru berbahasa Inggris di Yahoo Finance untuk {selected_ticker}.")
        else:
            st.warning("Sentiment API is unreachable. Please ensure the backend is running.")

with tab2:
    st.markdown("### Run Advanced AI Inference")
    col_p1, col_p2 = st.columns([1, 2.5])
    
    with col_p1:
        st.info("Pilih algoritma utama untuk memprediksi harga penutupan hari bursa berikutnya.")
        model_choice = st.selectbox("AI Architecture:", ["ARIMA (Statistical)", "LSTM (Deep Learning)"])
        run_btn = st.button(f"🚀 Initialize AI Engine for {selected_ticker}", use_container_width=True)

    with col_p2:
        if run_btn:
            model_key = "lstm" if "LSTM" in model_choice else "arima"
            with st.spinner(f"Running mathematical inference via {model_key.upper()} untuk {selected_ticker}..."):
                pred_res = fetch_prediction(model_key, selected_ticker)
            
            if pred_res and sent_res:
                last_price = pred_res['last_real_price']
                predicted = pred_res['predicted_price']
                trend_naik = predicted > last_price
                sent_score = sent_res['average_score']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Last Close Price", f"${last_price}", f"Date: {pred_res['last_real_date']}")
                m2.metric(f"Predicted Price ({model_key.upper()})", f"${predicted}", f"{'Up' if trend_naik else 'Down'} trend expected", delta_color="normal" if trend_naik else "inverse")
                m3.metric("Prediction Target Date", pred_res['prediction_date'])
                
                st.markdown("<br/>", unsafe_allow_html=True)
                
                if trend_naik and sent_score >= 0.1:
                    msg, bc, bg = "🔥 STRONG BUY SIGNAL<br/>Confluence: AI predicts upward trend + Positive global news.", "#00C853", "rgba(0, 200, 83, 0.1)"
                elif not trend_naik and sent_score <= -0.1:
                    msg, bc, bg = "📉 STRONG SELL SIGNAL<br/>Confluence: AI predicts downward trend + Negative global news.", "#D50000", "rgba(213, 0, 0, 0.1)"
                elif trend_naik and sent_score <= -0.1:
                    msg, bc, bg = "⚖️ CAUTION / HOLD<br/>Divergence: AI predicts UP, but Sentiment is NEGATIVE.", "#FFD600", "rgba(255, 214, 0, 0.1)"
                else:
                    msg, bc, bg = "⚖️ NEUTRAL / HOLD<br/>No strong confluence. Proceed with strict risk management.", "#FFD600", "rgba(255, 214, 0, 0.1)"
                
                st.markdown(f'<div style="border: 2px solid {bc}; background-color: {bg}; padding: 15px; border-radius: 8px; text-align: center;"><h4 style="margin:0px;">{msg}</h4></div>', unsafe_allow_html=True)
            elif not pred_res:
                st.error("Prediction Engine failed. Cek terminal untuk error log atau pastikan Ticker valid.")

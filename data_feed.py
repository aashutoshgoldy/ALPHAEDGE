import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

POPULAR_SYMBOLS = {
    # --- COMMODITIES (MCX / Global Futures & ETFs) ---
    "🥇 GOLD Futures (MCX / Global)": "GC=F",
    "🥈 SILVER Futures (MCX / Global)": "SI=F",
    "🛢️ CRUDE OIL Futures (MCX / WTI)": "CL=F",
    "🔥 NATURAL GAS Futures (MCX / Henry Hub)": "NG=F",
    "🥉 COPPER Futures": "HG=F",
    "🟡 GOLDBEES (NSE Gold ETF)": "GOLDBEES.NS",
    "⚪ SILVERBEES (NSE Silver ETF)": "SILVERBEES.NS",

    # --- INDIAN INDICES ---
    "🇮🇳 NIFTY 50 Index": "^NSEI",
    "🇮🇳 BANK NIFTY Index": "^NSEBANK",

    # --- MOMENTUM & HIGH BETA INDIAN STOCKS ---
    "📈 TRENT (Momentum)": "TRENT.NS",
    "🚀 ZOMATO (High Beta)": "ZOMATO.NS",
    "🛡️ HAL (Defence Momentum)": "HAL.NS",
    "🏢 RELIANCE": "RELIANCE.NS",
    "🚗 TATA MOTORS": "TATAMOTORS.NS",
    "🏦 HDFC BANK": "HDFCBANK.NS",
    "🏦 ICICI BANK": "ICICIBANK.NS",
    "🏦 STATE BANK OF INDIA (SBIN)": "SBIN.NS",
    "💳 BAJAJ FINANCE": "BAJFINANCE.NS",

    # --- CRYPTO & US ---
    "₿ BITCOIN (Crypto)": "BTC-USD",
    "🟣 SOLANA (High Beta Crypto)": "SOL-USD",
    "💻 NVIDIA (US Tech)": "NVDA",
    "⚡ TESLA (US High Beta)": "TSLA"
}

def clean_yfinance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns if present and ensure clean OHLCV"""
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-index
        df.columns = [col[0] for col in df.columns]

    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing essential price column: {col}")

    if 'Volume' not in df.columns:
        df['Volume'] = 100000

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def generate_synthetic_data(symbol: str, days: int = 365, base_price: float = 24000.0) -> pd.DataFrame:
    """Fallback generator for realistic market data when offline"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    returns = np.random.normal(0.0008, 0.015, size=days)
    price = base_price * np.exp(np.cumsum(returns))

    high = price * (1 + np.abs(np.random.normal(0, 0.008, size=days)))
    low = price * (1 - np.abs(np.random.normal(0, 0.008, size=days)))
    open_p = price * (1 + np.random.normal(0, 0.004, size=days))
    volume = np.random.randint(500000, 3000000, size=days)

    df = pd.DataFrame({
        'Open': open_p,
        'High': np.maximum(high, np.maximum(open_p, price)),
        'Low': np.minimum(low, np.minimum(open_p, price)),
        'Close': price,
        'Volume': volume
    }, index=dates)
    return df

def fetch_market_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical candles via yfinance with automatic interval-period compatibility.
    period: 1d, 5d, 7d, 1mo, 3mo, 6mo, 1y, 2y, 5y
    interval: 1m, 2m, 5m, 15m, 30m, 1h, 1d
    """
    # Yahoo Finance strict limit safety clamps
    if interval == "1m":
        if period not in ["1d", "3d", "5d", "7d"]:
            period = "5d"
    elif interval in ["2m", "5m", "15m", "30m", "90m"]:
        if period in ["1y", "2y", "5y", "max"]:
            period = "60d"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is not None and len(df) >= 5:
            return clean_yfinance_dataframe(df)
        else:
            print(f"[Warning] yfinance returned empty dataframe for {symbol} ({period}, {interval}). Using fallback.")
            return generate_synthetic_data(symbol)
    except Exception as e:
        print(f"[Error fetching data for {symbol}]: {e}. Using fallback.")
        return generate_synthetic_data(symbol)

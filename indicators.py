import numpy as np
import pandas as pd

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) for volatility and stop-loss calculation"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Supertrend indicator calculation.
    Returns: (supertrend_series, direction_series) where direction is 1 (bullish) or -1 (bearish)
    """
    hl2 = (df['High'] + df['Low']) / 2
    atr = calculate_atr(df, period)
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='int32')
    
    close = df['Close'].values
    upper_b = upper_band.values
    lower_b = lower_band.values
    
    curr_dir = 1
    curr_st = lower_b[0] if len(lower_b) > 0 else 0
    
    st_vals = []
    dir_vals = []
    
    for i in range(len(df)):
        if i == 0:
            st_vals.append(curr_st)
            dir_vals.append(curr_dir)
            continue
            
        c = close[i]
        c_prev = close[i-1]
        ub = upper_b[i]
        lb = lower_b[i]
        prev_st = st_vals[-1]
        
        if curr_dir == 1:
            if c < prev_st:
                curr_dir = -1
                curr_st = ub
            else:
                curr_st = max(lb, prev_st) if c_prev > prev_st else lb
        else:
            if c > prev_st:
                curr_dir = 1
                curr_st = lb
            else:
                curr_st = min(ub, prev_st) if c_prev < prev_st else ub
                
        st_vals.append(curr_st)
        dir_vals.append(curr_dir)
        
    return pd.Series(st_vals, index=df.index), pd.Series(dir_vals, index=df.index)

def calculate_donchian_channel(df: pd.DataFrame, period: int = 20):
    """Donchian Breakout Channel (High of last N periods, Low of last N periods)"""
    upper = df['High'].shift(1).rolling(period).max()
    lower = df['Low'].shift(1).rolling(period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower

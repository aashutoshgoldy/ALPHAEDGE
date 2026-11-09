import os
import sys
# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_feed import fetch_market_data, POPULAR_SYMBOLS
from strategy_engine import AlphaBacktestEngine
from performance import PerformanceAnalyzer

# Page configuration
st.set_page_config(
    page_title="AlphaEdge | Quantitative Backtesting & Strategy Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00FFA3, #00B8D9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 4px solid #00FFA3;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-card-danger {
        background: #1E293B;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 4px solid #EF4444;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("## ⚙️ Strategy & Capital Controls")

# 1. Asset & Data
st.sidebar.subheader("1. Asset & Data Source")
selected_preset = st.sidebar.selectbox(
    "Choose Preset Symbol / Commodity:",
    list(POPULAR_SYMBOLS.keys()),
    index=0
)
custom_ticker = st.sidebar.text_input("Or Enter Custom Ticker (e.g. GC=F, CL=F, TATAPOWER.NS):", "")
symbol = custom_ticker.strip() if custom_ticker.strip() else POPULAR_SYMBOLS[selected_preset]

# Timeframe selector with full Intraday (1m to 1d)
timeframe_choice = st.sidebar.selectbox(
    "Candle Interval (Timeframe):",
    [
        "1m (Intraday Scalping)",
        "2m (Intraday)",
        "5m (Intraday Trend)",
        "15m (Intraday/BTST)",
        "30m (Half-Hourly)",
        "1h (Hourly Trend)",
        "1d (Daily Swing)"
    ],
    index=2  # default 5m
)
clean_interval = timeframe_choice.split()[0]  # Extracts '1m', '5m', '1d'

# Dynamic Lookback Period based on exchange limits
if clean_interval == "1m":
    period_options = ["1d", "3d", "5d", "7d"]
    default_period_idx = 2  # 5d
    st.sidebar.caption("⚡ *1-Min Scalping: Exchange provides up to last 7 days data.*")
elif clean_interval in ["2m", "5m", "15m", "30m"]:
    period_options = ["1d", "5d", "1mo", "60d"]
    default_period_idx = 2  # 1mo
    st.sidebar.caption("⚡ *Intraday: Exchange provides up to last 60 days data.*")
elif clean_interval == "1h":
    period_options = ["1mo", "3mo", "6mo", "1y", "2y"]
    default_period_idx = 2  # 6mo
else:
    period_options = ["3mo", "6mo", "1y", "2y", "5y"]
    default_period_idx = 2  # 1y

history_period = st.sidebar.selectbox("Historical Lookback Period:", period_options, index=default_period_idx)

# 2. Capital & Money Management
st.sidebar.subheader("2. Capital & Position Sizing")
initial_capital = st.sidebar.number_input(
    "Starting Capital (₹)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=25000
)

risk_pct = st.sidebar.slider(
    "Risk % Per Trade (Capital Risk)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.25,
    help="Each trade will automatically size its quantity so that if Stop-Loss hits, you lose exactly this % of capital."
)

target_rr = st.sidebar.slider(
    "Target Risk-to-Reward (R:R)",
    min_value=1.5,
    max_value=5.0,
    value=2.5,
    step=0.25,
    help="Target distance = Stop Loss distance * R:R multiplier."
)

trailing_sl = st.sidebar.checkbox("Enable Trailing Stop-Loss", value=True, help="Lock in gains once 1.2R profit is reached.")

# 3. Strategy Selection
st.sidebar.subheader("3. Strategy Model")
strategy_type = st.sidebar.radio(
    "Select Strategy Engine:",
    [
        "🚀 High-Beta Swing Breakout (Equity/Cash)",
        "⚡ Explosive Options Buyer (ATM Calls/Puts)",
        "🛡️ High-POP Credit Spread (Options Seller / Theta)"
    ],
    index=0
)

st.sidebar.markdown("---")
brokerage_fees = st.sidebar.checkbox("Include Brokerage & Slippage (₹20/order)", value=True)

# ----------------- DATA FETCHING -----------------
with st.spinner(f"Fetching historical market data for {symbol} ({clean_interval})..."):
    df = fetch_market_data(symbol, period=history_period, interval=clean_interval)

if df.empty or len(df) < 10:
    st.error(f"Could not load sufficient price data for symbol '{symbol}' on {clean_interval} timeframe. Try adjusting lookback period.")
    st.stop()

# ----------------- BACKTEST EXECUTION -----------------
engine = AlphaBacktestEngine(
    df=df,
    symbol=symbol,
    initial_capital=float(initial_capital),
    risk_per_trade_pct=float(risk_pct),
    target_rr_ratio=float(target_rr),
    trailing_sl_enabled=trailing_sl,
    slippage_pct=0.05 if brokerage_fees else 0.0,
    brokerage_per_order=20.0 if brokerage_fees else 0.0
)

if "High-Beta Swing Breakout" in strategy_type:
    trades, equity_curve = engine.run_momentum_breakout(execution_mode="EQUITY_SWING")
elif "Explosive Options Buyer" in strategy_type:
    trades, equity_curve = engine.run_momentum_breakout(execution_mode="OPTION_BUY")
else:
    trades, equity_curve = engine.run_supertrend_credit_spread()

metrics = PerformanceAnalyzer.compute_metrics(trades, equity_curve, float(initial_capital))
trades_df = PerformanceAnalyzer.trades_to_dataframe(trades)

# ----------------- HEADER -----------------
st.markdown('<div class="main-header">⚡ AlphaEdge Backtesting & Strategy Lab</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Backtesting <b>{symbol}</b> | Strategy: <b>{strategy_type}</b> | Capital: <b>₹{initial_capital:,.0f}</b> | Risk/Trade: <b>{risk_pct}% (₹{(initial_capital * risk_pct / 100):,.0f})</b></div>',
    unsafe_allow_html=True
)

# ----------------- KEY METRIC CARDS -----------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    pnl = metrics['net_profit_rs']
    roi = metrics['roi_pct']
    card_class = "metric-card" if pnl >= 0 else "metric-card-danger"
    color = "#00FFA3" if pnl >= 0 else "#EF4444"
    st.markdown(f"""
    <div class="{card_class}">
        <div class="metric-title">Total Net P&L</div>
        <div class="metric-value" style="color:{color}">₹{pnl:+,.0f}</div>
        <div class="metric-sub" style="color:{color}">ROI: {roi:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    win_rate = metrics['win_rate_pct']
    color = "#00FFA3" if win_rate >= 50 else "#F59E0B"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Win Rate (Empirical POP)</div>
        <div class="metric-value" style="color:{color}">{win_rate:.1f}%</div>
        <div class="metric-sub" style="color:#94A3B8">{metrics['win_trades']} Won / {metrics['loss_trades']} Lost</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pop = metrics['avg_pop_pct']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Entry POP</div>
        <div class="metric-value" style="color:#38BDF8">{pop:.1f}%</div>
        <div class="metric-sub" style="color:#94A3B8">Prob. of Profit at Entry</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    pf = metrics['profit_factor']
    pf_color = "#00FFA3" if pf >= 1.5 else ("#F59E0B" if pf >= 1.0 else "#EF4444")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Profit Factor</div>
        <div class="metric-value" style="color:{pf_color}">{pf:.2f}</div>
        <div class="metric-sub" style="color:#94A3B8">Gross Win / Gross Loss</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    mdd = metrics['max_drawdown_pct']
    st.markdown(f"""
    <div class="metric-card-danger">
        <div class="metric-title">Max Drawdown</div>
        <div class="metric-value" style="color:#EF4444">-{mdd:.1f}%</div>
        <div class="metric-sub" style="color:#EF4444">Worst Dip: -₹{metrics['max_drawdown_rs']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    exp_val = metrics['expectancy_rs']
    ev_color = "#00FFA3" if exp_val >= 0 else "#EF4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Expectancy / Trade</div>
        <div class="metric-value" style="color:{ev_color}">₹{exp_val:+,.0f}</div>
        <div class="metric-sub" style="color:#94A3B8">Realized R:R: 1:{metrics['avg_trade_rr']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Chart & Signals",
    "📈 Equity Curve vs Benchmark",
    "🌊 Drawdown Exposure",
    "📜 Detailed Trade Ledger",
    "🤖 Live Broker Automation Blueprint"
])

# TAB 1: Candlestick + Indicators
with tab1:
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    
    # Candlestick
    fig_candle.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color='#00FFA3',
            decreasing_line_color='#FF4B4B'
        ),
        row=1, col=1
    )

    # Indicator overlay
    if 'Donchian_High' in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['Donchian_High'], line=dict(color='rgba(56, 189, 248, 0.7)', width=1, dash='dash'), name='Donchian Upper (Breakout)'), row=1, col=1)
    if 'EMA_Trend' in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['EMA_Trend'], line=dict(color='#F59E0B', width=1.5), name='Trend Filter (EMA 50)'), row=1, col=1)
    if 'Supertrend' in df.columns:
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['Supertrend'], line=dict(color='#A855F7', width=1.5), name='Supertrend'), row=1, col=1)

    # Add Buy / Sell markers
    if not trades_df.empty:
        entries = trades_df[['Entry Date', 'Entry (₹)', 'Dir']].dropna()
        exits = trades_df[['Exit Date', 'Exit (₹)', 'Exit Reason']].dropna()
        
        fig_candle.add_trace(
            go.Scatter(
                x=pd.to_datetime(entries['Entry Date']),
                y=entries['Entry (₹)'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=11, color='#00FFA3'),
                name='Trade Entry'
            ),
            row=1, col=1
        )

        fig_candle.add_trace(
            go.Scatter(
                x=pd.to_datetime(exits['Exit Date']),
                y=exits['Exit (₹)'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=11, color='#EF4444'),
                name='Trade Exit'
            ),
            row=1, col=1
        )

    # Volume bar chart
    fig_candle.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color='#334155', name='Volume', opacity=0.7),
        row=2, col=1
    )

    fig_candle.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=20, b=20),
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_candle, use_container_width=True)

# TAB 2: Equity Curve vs Benchmark
with tab2:
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df['date'] = pd.to_datetime(eq_df['date'])

        # Benchmark calculation
        first_close = eq_df['close'].iloc[0]
        eq_df['benchmark_equity'] = (eq_df['close'] / first_close) * initial_capital

        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df['date'],
                y=eq_df['equity'],
                mode='lines',
                line=dict(color='#00FFA3', width=2.5),
                name=f'Alpha Strategy (₹{metrics["final_equity"]:,.0f})'
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=eq_df['date'],
                y=eq_df['benchmark_equity'],
                mode='lines',
                line=dict(color='#64748B', width=1.5, dash='dot'),
                name=f'Buy & Hold Benchmark ({metrics["benchmark_return_pct"]:+.1f}%)'
            )
        )

        fig_eq.update_layout(
            height=450,
            template='plotly_dark',
            title='Portfolio Equity Progression vs Benchmark',
            yaxis_title='Account Capital (₹)',
            xaxis_title='Date',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No trades executed within this timeframe.")

# TAB 3: Drawdown Exposure
with tab3:
    if equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_values = eq_df['equity'].values
        running_max = np.maximum.accumulate(eq_values)
        dd_pct = -1 * ((running_max - eq_values) / running_max) * 100.0
        eq_df['drawdown_pct'] = dd_pct

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=eq_df['date'],
                y=eq_df['drawdown_pct'],
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.25)',
                line=dict(color='#EF4444', width=1.5),
                name='Drawdown %'
            )
        )
        fig_dd.update_layout(
            height=400,
            template='plotly_dark',
            title=f'Drawdown Waterfall (Max Drawdown: -{metrics["max_drawdown_pct"]:.1f}%)',
            yaxis_title='Drawdown % from Peak',
            xaxis_title='Date',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

# TAB 4: Trade Ledger
with tab4:
    if not trades_df.empty:
        col_t1, col_t2 = st.columns([0.8, 0.2])
        with col_t1:
            st.markdown(f"**Total Executed Trades: {len(trades_df)}**")
        with col_t2:
            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Log",
                data=csv,
                file_name=f"{symbol}_trades_backtest.csv",
                mime='text/csv'
            )

        st.dataframe(
            trades_df.style.apply(
                lambda row: ['background-color: rgba(0, 255, 163, 0.12)' if row['P&L (₹)'] > 0 else 'background-color: rgba(239, 68, 68, 0.12)' for _ in row],
                axis=1
            ),
            use_container_width=True,
            height=500
        )
    else:
        st.warning("No trade signals triggered with the current parameters.")

# TAB 5: Live Automation Blueprint
with tab5:
    st.markdown("### 🔌 Direct Broker API Execution Blueprint")
    st.info("The logic running in this backtester is directly decoupled in `strategy_engine.py` and `live_executor.py`. Below is how you connect live order execution with Indian brokers:")

    broker_choice = st.selectbox("Select Your Broker:", ["ICICI Direct Breeze", "5Paisa", "Zerodha Kite Connect", "Dhan HQ API"])

    if broker_choice == "ICICI Direct Breeze":
        st.code("""
# ICICI Direct Breeze Live Order Dispatch
from breeze_connect import BreezeConnect
from alpha_backtester.live_executor import LiveTradingBridge

# 1. Initialize Breeze Session
breeze = BreezeConnect(api_key="YOUR_API_KEY")
breeze.generate_session(api_secret="YOUR_SECRET_KEY", session_token="YOUR_SESSION_TOKEN")

# 2. When AlphaEdge generates a Buy signal:
def on_signal_generated(symbol, qty, stop_loss, target):
    order = breeze.place_order(
        stock_code=symbol,
        exchange_code="NSE",
        product="options",   # or "cash"
        action="buy",
        order_type="limit",
        stoploss=stop_loss,
        quantity=qty,
        price=current_market_price,
        validity="day"
    )
    print("Live Breeze Order Placed:", order)
""", language="python")

    elif broker_choice == "5Paisa":
        st.code("""
# 5Paisa Live Order Dispatch
from py5paisa import FivePaisaClient

client = FivePaisaClient(email="...", passwd="...", dob="...", cred={"APP_NAME": "...", "APP_SOURCE": "...", "USER_KEY": "...", "ENCRYPTION_KEY": "..."})
client.login_2fa(totp="...")

# Place order directly from strategy signal
client.place_order(
    OrderType='B',
    Exchange='N',
    ExchangeType='D',  # Derivative / Options
    ScripCode=12345,
    Qty=qty,
    Price=price,
    IsIntraday=False
)
""", language="python")

    elif broker_choice == "Zerodha Kite Connect":
        st.code("""
# Zerodha Kite Connect
from kiteconnect import KiteConnect

kite = KiteConnect(api_key="your_api_key")
kite.set_access_token("your_access_token")

# Place Bracket / Stoploss Market order
order_id = kite.place_order(
    variety=kite.VARIETY_REGULAR,
    exchange=kite.EXCHANGE_NFO,
    tradingsymbol=symbol,
    transaction_type=kite.TRANSACTION_TYPE_BUY,
    quantity=qty,
    order_type=kite.ORDER_TYPE_LIMIT,
    price=price,
    product=kite.PRODUCT_NRML
)
""", language="python")

    else:
        st.code("""
# Dhan HQ Python SDK
from dhanhq import dhanhq

dhan = dhanhq("client_id", "access_token")

dhan.place_order(
    security_id='1333',
    exchange_segment=dhan.NSE_FNO,
    transaction_type=dhan.BUY,
    quantity=qty,
    order_type=dhan.LIMIT,
    product_type=dhan.CNC,
    price=price
)
""", language="python")

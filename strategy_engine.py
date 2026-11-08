import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

try:
    from .indicators import (
        calculate_ema, calculate_atr, calculate_supertrend, 
        calculate_donchian_channel, calculate_rsi
    )
    from .options_model import OptionsPricingModel
except (ImportError, ValueError):
    from indicators import (
        calculate_ema, calculate_atr, calculate_supertrend, 
        calculate_donchian_channel, calculate_rsi
    )
    from options_model import OptionsPricingModel

@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    entry_date: str
    exit_date: str
    direction: str  # "LONG" or "SHORT"
    trade_type: str  # "EQUITY_SWING", "OPTION_BUY", "OPTION_SPREAD"
    strike_info: str  # e.g. "ATM CE 24500" or "SPOT"
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    quantity: int
    risk_rupees: float  # Capital * Risk %
    capital_deployed: float
    pop_percent: float  # Probability of Profit at entry
    pnl_rupees: float
    pnl_percent: float
    exit_reason: str  # "TARGET", "STOP_LOSS", "TRAILING_STOP", "END_OF_DATA"
    realized_rr: float
    portfolio_equity: float

class AlphaBacktestEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "NIFTY 50",
        initial_capital: float = 100000.0,
        risk_per_trade_pct: float = 2.0,  # 2% risk of capital per trade
        target_rr_ratio: float = 2.5,     # 1:2.5 Risk to Reward
        trailing_sl_enabled: bool = True,
        slippage_pct: float = 0.05,       # 0.05% slippage
        brokerage_per_order: float = 20.0 # Flat ₹20 per executed order (Zerodha/Angel/Dhan standard)
    ):
        self.df = df.copy()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_pct = risk_per_trade_pct / 100.0
        self.target_rr = target_rr_ratio
        self.trailing_sl_enabled = trailing_sl_enabled
        self.slippage = slippage_pct / 100.0
        self.brokerage = brokerage_per_order
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def run_momentum_breakout(
        self,
        donchian_period: int = 20,
        ema_trend_period: int = 50,
        atr_period: int = 14,
        atr_multiplier_sl: float = 1.5,
        execution_mode: str = "EQUITY_SWING"  # "EQUITY_SWING" or "OPTION_BUY"
    ):
        """
        High Risk / High Reward Momentum Breakout Strategy.
        - Entry: Price breaks Donchian High + Price > EMA 50
        - SL: ATR * atr_multiplier_sl below entry
        - Target: Entry + (Risk * target_rr)
        - Trailing SL: Trails profit at ATR distance once 1R is achieved
        """
        df = self.df
        upper, mid, lower = calculate_donchian_channel(df, donchian_period)
        ema = calculate_ema(df['Close'], ema_trend_period)
        atr = calculate_atr(df, atr_period)
        rsi = calculate_rsi(df['Close'], 14)

        df['Donchian_High'] = upper
        df['Donchian_Low'] = lower
        df['EMA_Trend'] = ema
        df['ATR'] = atr
        df['RSI'] = rsi

        in_position = False
        current_trade = {}
        trade_counter = 0

        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

        dates = df.index
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        dh = df['Donchian_High'].values
        emas = df['EMA_Trend'].values
        atrs = df['ATR'].values

        lot_size = 25 if "NIFTY" in self.symbol.upper() else (15 if "BANK" in self.symbol.upper() else 1)

        for i in range(max(donchian_period, ema_trend_period) + 2, len(df)):
            date_str = str(dates[i].date())
            price = closes[i]

            # Track daily equity
            self.equity_curve.append({
                'date': date_str,
                'equity': self.current_capital,
                'close': price
            })

            # Check existing position exit logic
            if in_position:
                entry_p = current_trade['entry_price']
                sl_p = current_trade['sl']
                target_p = current_trade['target']
                trail_sl = current_trade['trail_sl']

                # Check if high hit target or low hit SL
                day_high = highs[i]
                day_low = lows[i]
                exit_price = None
                exit_reason = ""

                # 1. Target hit
                if day_high >= target_p:
                    exit_price = target_p
                    exit_reason = "TARGET"
                # 2. Stop loss / Trailing SL hit
                elif day_low <= trail_sl:
                    exit_price = trail_sl
                    exit_reason = "TRAILING_STOP" if trail_sl > sl_p else "STOP_LOSS"
                # 3. Update trailing SL if price advanced
                elif self.trailing_sl_enabled:
                    # If profit has reached 1.2R, trail SL to Entry, and higher
                    risk_pts = current_trade['risk_pts']
                    if price > entry_p + (1.2 * risk_pts):
                        new_trail = price - (atrs[i] * 1.2)
                        if new_trail > current_trade['trail_sl']:
                            current_trade['trail_sl'] = max(current_trade['trail_sl'], new_trail)

                if exit_price is not None:
                    # Calculate P&L
                    qty = current_trade['qty']
                    risk_rs = current_trade['risk_rupees']
                    deployed = current_trade['deployed']

                    if execution_mode == "OPTION_BUY":
                        # Option delta estimation (ATM Call delta ~ 0.55)
                        delta = 0.55
                        underlying_move = exit_price - entry_p
                        opt_entry_prem = current_trade['opt_entry_prem']
                        # Theta loss per day (approx 3% of premium)
                        days_held = max(1, i - current_trade['entry_bar'])
                        theta_decay = days_held * (opt_entry_prem * 0.025)
                        opt_exit_prem = max(0.05, opt_entry_prem + (underlying_move * delta) - theta_decay)

                        gross_pnl = (opt_exit_prem - opt_entry_prem) * qty
                        pnl_rs = gross_pnl - (2 * self.brokerage)
                        pnl_pct = (pnl_rs / deployed) * 100.0 if deployed > 0 else 0
                        actual_rr = pnl_rs / risk_rs if risk_rs > 0 else 0
                    else:
                        # Equity / Futures swing
                        gross_pnl = (exit_price - entry_p) * qty
                        # Slippage cost
                        slippage_cost = (entry_p + exit_price) * qty * self.slippage
                        pnl_rs = gross_pnl - slippage_cost - (2 * self.brokerage)
                        pnl_pct = (pnl_rs / deployed) * 100.0 if deployed > 0 else 0
                        actual_rr = pnl_rs / risk_rs if risk_rs > 0 else 0

                    self.current_capital += pnl_rs
                    trade_counter += 1

                    self.trades.append(TradeRecord(
                        trade_id=trade_counter,
                        symbol=self.symbol,
                        entry_date=current_trade['entry_date'],
                        exit_date=date_str,
                        direction="LONG",
                        trade_type=execution_mode,
                        strike_info=current_trade['strike_info'],
                        entry_price=round(entry_p, 2),
                        exit_price=round(exit_price, 2),
                        stop_loss=round(sl_p, 2),
                        target=round(target_p, 2),
                        quantity=qty,
                        risk_rupees=round(risk_rs, 2),
                        capital_deployed=round(deployed, 2),
                        pop_percent=round(current_trade['pop'], 2),
                        pnl_rupees=round(pnl_rs, 2),
                        pnl_percent=round(pnl_pct, 2),
                        exit_reason=exit_reason,
                        realized_rr=round(actual_rr, 2),
                        portfolio_equity=round(self.current_capital, 2)
                    ))
                    in_position = False
                    current_trade = {}

            # Check new entry logic (Only if not already in position)
            if not in_position and not np.isnan(dh[i-1]) and not np.isnan(emas[i-1]):
                # Condition: Yesterday closed below Donchian High, today breaks above Donchian High, and above EMA
                prev_c = closes[i-1]
                prev_dh = dh[i-1]
                curr_c = closes[i]
                curr_ema = emas[i]
                curr_atr = atrs[i]

                if curr_c > prev_dh and curr_c > curr_ema and not np.isnan(curr_atr) and curr_atr > 0:
                    entry_price = curr_c
                    risk_points = curr_atr * atr_multiplier_sl
                    sl_price = entry_price - risk_points
                    target_price = entry_price + (risk_points * self.target_rr)

                    # Position Sizing: Risk per trade = Capital * Risk%
                    risk_rupees = self.current_capital * self.risk_pct
                    
                    if execution_mode == "OPTION_BUY":
                        # Simulate ATM Strike
                        strike = round(entry_price / 50.0) * 50 if "NIFTY" in self.symbol.upper() else round(entry_price / 100.0) * 100
                        days_to_expiry = 7  # weekly expiry proxy
                        opt_prem = OptionsPricingModel.calculate_price(
                            S=entry_price, K=strike, T=days_to_expiry/365.0, sigma=0.15, option_type='CE'
                        )
                        # Option SL point is estimated ~ 40% of option premium or delta * risk_points
                        opt_sl_pts = min(opt_prem * 0.40, risk_points * 0.50)
                        raw_qty = int(risk_rupees / opt_sl_pts) if opt_sl_pts > 0 else lot_size
                        qty = max(lot_size, (raw_qty // lot_size) * lot_size)
                        deployed = qty * opt_prem
                        strike_info = f"ATM {strike} CE (₹{round(opt_prem, 1)})"

                        # Probability of profit for buyer: probability spot reaches breakeven (Strike + Prem)
                        pop = OptionsPricingModel.calculate_pop(
                            S=entry_price, breakeven=strike + opt_prem, T=days_to_expiry/365.0, sigma=0.15, trade_type='BUY_CALL'
                        )
                    else:
                        # Equity / Cash position sizing
                        raw_qty = int(risk_rupees / risk_points) if risk_points > 0 else 1
                        qty = max(1, raw_qty)
                        deployed = min(qty * entry_price, self.current_capital)
                        strike_info = "SPOT CASH / FUT"
                        # Risk-neutral theoretical POP based on SL and Target distances
                        # POP = Risk / (Risk + Reward)
                        pop = (risk_points / (risk_points + (target_price - entry_price))) * 100.0

                    in_position = True
                    current_trade = {
                        'entry_date': date_str,
                        'entry_bar': i,
                        'entry_price': entry_price,
                        'sl': sl_price,
                        'trail_sl': sl_price,
                        'target': target_price,
                        'risk_pts': risk_points,
                        'risk_rupees': risk_rupees,
                        'qty': qty,
                        'deployed': deployed,
                        'strike_info': strike_info,
                        'pop': pop,
                        'opt_entry_prem': opt_prem if execution_mode == "OPTION_BUY" else 0.0
                    }

        return self.trades, self.equity_curve

    def run_supertrend_credit_spread(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        spread_width_pts: float = 300.0,
        otm_buffer_pts: float = 200.0
    ):
        """
        High POP (Probability of Profit) Option Selling / Credit Spread Strategy.
        - When Supertrend flips Bullish: Sell OTM Bull Put Spread (Collect Theta)
        - When Supertrend flips Bearish: Sell OTM Bear Call Spread (Collect Theta)
        """
        df = self.df
        st_val, st_dir = calculate_supertrend(df, period=st_period, multiplier=st_multiplier)
        df['Supertrend'] = st_val
        df['ST_Dir'] = st_dir

        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        trade_counter = 0

        dates = df.index
        closes = df['Close'].values
        dirs = df['ST_Dir'].values

        lot_size = 25 if "NIFTY" in self.symbol.upper() else (15 if "BANK" in self.symbol.upper() else 1)

        in_pos = False
        current_trade = {}

        for i in range(st_period + 2, len(df)):
            date_str = str(dates[i].date())
            price = closes[i]

            self.equity_curve.append({
                'date': date_str,
                'equity': self.current_capital,
                'close': price
            })

            # Check trade management (Exit if ST flips or after 5 bars / expiry)
            if in_pos:
                bars_held = i - current_trade['entry_bar']
                st_reversed = (dirs[i] != current_trade['dir'])
                
                # Option spread expires after 5-7 days or if trend reverses
                if st_reversed or bars_held >= 6:
                    entry_net_credit = current_trade['net_credit']
                    max_loss_pts = current_trade['max_loss_pts']
                    qty = current_trade['qty']
                    risk_rs = current_trade['risk_rupees']

                    if st_reversed:
                        # Adverse move - lost part or full spread
                        exit_cost = entry_net_credit * 2.2  # Stop loss triggered on spread
                        pnl_pts = entry_net_credit - exit_cost
                        exit_reason = "STOP_LOSS (Trend Reversal)"
                    else:
                        # Full theta decay collected! (Max Profit)
                        pnl_pts = entry_net_credit * 0.90  # 90% decay harvested
                        exit_reason = "EXPIRY (Theta Harvested)"

                    pnl_rs = (pnl_pts * qty) - (4 * self.brokerage) # 4 legs total
                    pnl_pct = (pnl_rs / current_trade['deployed']) * 100.0
                    actual_rr = pnl_rs / risk_rs if risk_rs > 0 else 0

                    self.current_capital += pnl_rs
                    trade_counter += 1

                    self.trades.append(TradeRecord(
                        trade_id=trade_counter,
                        symbol=self.symbol,
                        entry_date=current_trade['entry_date'],
                        exit_date=date_str,
                        direction="BULLISH_SPREAD" if current_trade['dir'] == 1 else "BEARISH_SPREAD",
                        trade_type="OPTION_CREDIT_SPREAD",
                        strike_info=current_trade['strike_info'],
                        entry_price=round(current_trade['entry_price'], 2),
                        exit_price=round(price, 2),
                        stop_loss=round(current_trade['sl'], 2),
                        target=round(current_trade['target'], 2),
                        quantity=qty,
                        risk_rupees=round(risk_rs, 2),
                        capital_deployed=round(current_trade['deployed'], 2),
                        pop_percent=round(current_trade['pop'], 2),
                        pnl_rupees=round(pnl_rs, 2),
                        pnl_percent=round(pnl_pct, 2),
                        exit_reason=exit_reason,
                        realized_rr=round(actual_rr, 2),
                        portfolio_equity=round(self.current_capital, 2)
                    ))
                    in_pos = False
                    current_trade = {}

            # New signal entry
            if not in_pos and dirs[i] != dirs[i-1]:
                curr_dir = dirs[i]
                entry_price = price
                risk_rupees = self.current_capital * self.risk_pct

                if curr_dir == 1:
                    # Bullish -> Sell OTM Put Spread
                    short_strike = round((entry_price - otm_buffer_pts) / 50.0) * 50
                    long_strike = short_strike - spread_width_pts
                    opt_type = 'PE'
                    strike_desc = f"Bull Put Spread (Sell {short_strike}PE / Buy {long_strike}PE)"
                    trade_pop_type = 'SELL_PUT'
                else:
                    # Bearish -> Sell OTM Call Spread
                    short_strike = round((entry_price + otm_buffer_pts) / 50.0) * 50
                    long_strike = short_strike + spread_width_pts
                    opt_type = 'CE'
                    strike_desc = f"Bear Call Spread (Sell {short_strike}CE / Buy {long_strike}CE)"
                    trade_pop_type = 'SELL_CALL'

                prem_short = OptionsPricingModel.calculate_price(entry_price, short_strike, 7/365.0, sigma=0.15, option_type=opt_type)
                prem_long = OptionsPricingModel.calculate_price(entry_price, long_strike, 7/365.0, sigma=0.15, option_type=opt_type)
                net_credit = max(5.0, prem_short - prem_long)
                max_loss_pts = max(10.0, spread_width_pts - net_credit)

                raw_qty = int(risk_rupees / max_loss_pts) if max_loss_pts > 0 else lot_size
                qty = max(lot_size, (raw_qty // lot_size) * lot_size)
                margin_required = spread_width_pts * qty # Defined risk spread margin

                # POP calculation for credit spread (Usually high: 70% to 85%)
                pop = OptionsPricingModel.calculate_pop(
                    S=entry_price, breakeven=short_strike, T=7/365.0, sigma=0.15, trade_type=trade_pop_type
                )

                in_pos = True
                current_trade = {
                    'entry_date': date_str,
                    'entry_bar': i,
                    'dir': curr_dir,
                    'entry_price': entry_price,
                    'sl': entry_price - (curr_dir * otm_buffer_pts),
                    'target': entry_price + (curr_dir * otm_buffer_pts),
                    'net_credit': net_credit,
                    'max_loss_pts': max_loss_pts,
                    'qty': qty,
                    'risk_rupees': risk_rupees,
                    'deployed': margin_required,
                    'strike_info': strike_desc,
                    'pop': pop
                }

        return self.trades, self.equity_curve

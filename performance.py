import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    from .strategy_engine import TradeRecord
except (ImportError, ValueError):
    from strategy_engine import TradeRecord

class PerformanceAnalyzer:
    @staticmethod
    def compute_metrics(
        trades: List[TradeRecord],
        equity_curve: List[Dict[str, Any]],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Compute comprehensive performance teardown metrics"""
        if not trades:
            return {
                "total_trades": 0,
                "net_profit_rs": 0.0,
                "roi_pct": 0.0,
                "win_rate_pct": 0.0,
                "avg_pop_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_rs": 0.0,
                "expectancy_rs": 0.0,
                "sharpe_ratio": 0.0,
                "final_equity": initial_capital,
                "benchmark_return_pct": 0.0,
                "avg_trade_rr": 0.0,
                "consecutive_losses": 0
            }

        pnl_series = [t.pnl_rupees for t in trades]
        wins = [p for p in pnl_series if p > 0]
        losses = [p for p in pnl_series if p <= 0]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = abs(float(np.mean(losses))) if losses else 0.0

        profit_factor = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else (99.0 if sum(wins) > 0 else 0.0)

        # Expectancy per trade: (Win% * Avg Win) - (Loss% * Avg Loss)
        win_prob = win_rate / 100.0
        loss_prob = 1.0 - win_prob
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)

        # Average POP predicted at entry
        avg_pop = float(np.mean([t.pop_percent for t in trades]))

        # Max Drawdown calculation from equity curve
        eq_df = pd.DataFrame(equity_curve)
        if not eq_df.empty and 'equity' in eq_df.columns:
            eq_values = eq_df['equity'].values
            running_max = np.maximum.accumulate(eq_values)
            drawdowns = (running_max - eq_values)
            drawdown_pcts = (drawdowns / running_max) * 100.0
            max_dd_rs = float(np.max(drawdowns))
            max_dd_pct = float(np.max(drawdown_pcts))
            final_equity = float(eq_values[-1])
        else:
            max_dd_rs = 0.0
            max_dd_pct = 0.0
            final_equity = initial_capital + sum(pnl_series)

        net_profit_rs = final_equity - initial_capital
        roi_pct = (net_profit_rs / initial_capital) * 100.0

        # Benchmark return (Buy & Hold)
        if not eq_df.empty and 'close' in eq_df.columns and len(eq_df['close']) > 1:
            first_p = eq_df['close'].iloc[0]
            last_p = eq_df['close'].iloc[-1]
            benchmark_return = ((last_p - first_p) / first_p) * 100.0
        else:
            benchmark_return = 0.0

        # Sharpe Ratio (Approx based on trade return variance)
        trade_returns = [t.pnl_percent for t in trades]
        std_returns = np.std(trade_returns) if len(trade_returns) > 1 else 1.0
        mean_return = np.mean(trade_returns) if trade_returns else 0.0
        sharpe = (mean_return / std_returns) * np.sqrt(min(total_trades, 50)) if std_returns > 0 else 0.0

        # Consecutive losses
        max_cons_losses = 0
        curr_cons = 0
        for p in pnl_series:
            if p <= 0:
                curr_cons += 1
                max_cons_losses = max(max_cons_losses, curr_cons)
            else:
                curr_cons = 0

        avg_rr = float(np.mean([t.realized_rr for t in trades]))

        return {
            "total_trades": total_trades,
            "win_trades": win_count,
            "loss_trades": loss_count,
            "win_rate_pct": round(win_rate, 2),
            "avg_pop_pct": round(avg_pop, 2),
            "avg_win_rs": round(avg_win, 2),
            "avg_loss_rs": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy_rs": round(expectancy, 2),
            "net_profit_rs": round(net_profit_rs, 2),
            "roi_pct": round(roi_pct, 2),
            "final_equity": round(final_equity, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "max_drawdown_rs": round(max_dd_rs, 2),
            "sharpe_ratio": round(sharpe, 2),
            "benchmark_return_pct": round(benchmark_return, 2),
            "avg_trade_rr": round(avg_rr, 2),
            "consecutive_losses": max_cons_losses
        }

    @staticmethod
    def trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
        """Convert list of TradeRecord to formatted DataFrame"""
        if not trades:
            return pd.DataFrame()
        
        data = [t.__dict__ for t in trades]
        df = pd.DataFrame(data)
        
        # Rename columns for presentation
        col_rename = {
            'trade_id': '#',
            'symbol': 'Symbol',
            'entry_date': 'Entry Date',
            'exit_date': 'Exit Date',
            'direction': 'Dir',
            'trade_type': 'Type',
            'strike_info': 'Strike / Setup',
            'entry_price': 'Entry (₹)',
            'exit_price': 'Exit (₹)',
            'stop_loss': 'SL (₹)',
            'target': 'Target (₹)',
            'quantity': 'Qty',
            'risk_rupees': 'Risk (₹)',
            'capital_deployed': 'Deployed (₹)',
            'pop_percent': 'POP %',
            'pnl_rupees': 'P&L (₹)',
            'pnl_percent': 'Return %',
            'exit_reason': 'Exit Reason',
            'realized_rr': 'Realized R:R',
            'portfolio_equity': 'Equity (₹)'
        }
        return df.rename(columns=col_rename)

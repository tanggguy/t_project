from __future__ import annotations

# --- 1. Bibliothèques natives ---
from typing import Dict, Any, Optional, List

# --- 2. Bibliothèques tierces ---
import numpy as np
import pandas as pd

try:
    import backtrader as bt
except Exception:  # pragma: no cover - backtrader peut ne pas être dispo dans certains contextes
    bt = None  # type: ignore


def _as_series(x: pd.Series | Dict[Any, float]) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    s = pd.Series(x)
    # Essayer de convertir l'index en datetime si possible
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s.sort_index()


def compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    total_return = float(equity.iloc[-1]) / float(equity.iloc[0])
    if total_return <= 0:
        return 0.0
    years = max(len(equity) / periods_per_year, 1e-9)
    return total_return ** (1.0 / years) - 1.0


def compute_sharpe(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    # Log returns attendus
    rf_period = risk_free_rate_annual / periods_per_year
    excess = r - rf_period
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std) * np.sqrt(periods_per_year)


def compute_sortino(
    returns: pd.Series,
    periods_per_year: int = 252,
    mar_annual: float = 0.0,
) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    mar_period = mar_annual / periods_per_year
    downside = r[r < mar_period] - mar_period
    dd = downside.pow(2).mean()
    dd_std = np.sqrt(dd) if dd is not None and not np.isnan(dd) else 0.0
    if dd_std == 0:
        return 0.0
    return float((r.mean() - mar_period) / dd_std) * np.sqrt(periods_per_year)


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    md = abs(float(max_drawdown))
    if md == 0:
        return 0.0
    return float(cagr) / md


def compute_trade_stats(trades: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "total_trades": 0,
            "won_trades": 0,
            "lost_trades": 0,
            "win_rate": 0.0,
            "profit_factor": np.inf,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
            "max_consecutive_losses": 0,
            "avg_trade_size": 0.0,
            "avg_trade_duration_days": 0.0,
        }

    df = trades.copy()
    # Essayer de trier chronologiquement pour les statistiques de séries
    if "entry_dt" in df.columns:
        try:
            df["entry_dt"] = pd.to_datetime(df["entry_dt"], errors="coerce")
            df = df.sort_values("entry_dt")
        except Exception:
            df = df.copy()
    elif "exit_dt" in df.columns:
        try:
            df["exit_dt"] = pd.to_datetime(df["exit_dt"], errors="coerce")
            df = df.sort_values("exit_dt")
        except Exception:
            df = df.copy()
    # Supposer colonne 'net_pnl' (fallback sur 'pnl')
    if "net_pnl" in df.columns:
        pnl_col = "net_pnl"
    else:
        pnl_col = "pnl" if "pnl" in df.columns else None
    if pnl_col is None:
        return {
            "total_trades": len(df),
            "won_trades": 0,
            "lost_trades": 0,
            "win_rate": 0.0,
            "profit_factor": np.inf,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
            "max_consecutive_losses": 0,
            "avg_trade_size": 0.0,
            "avg_trade_duration_days": 0.0,
        }

    total = len(df)
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] < 0]
    won = len(wins)
    lost = len(losses)
    win_rate = (won / total) * 100.0 if total > 0 else 0.0

    total_won = float(wins[pnl_col].sum()) if won > 0 else 0.0
    total_lost_abs = abs(float(losses[pnl_col].sum())) if lost > 0 else 0.0
    profit_factor = (total_won / total_lost_abs) if total_lost_abs > 0 else np.inf

    avg_win = float(wins[pnl_col].mean()) if won > 0 else 0.0
    avg_loss = float(losses[pnl_col].mean()) if lost > 0 else 0.0
    best_trade = float(df[pnl_col].max()) if total > 0 else 0.0
    worst_trade = float(df[pnl_col].min()) if total > 0 else 0.0

    # Taille moyenne de trade (absolue si disponible)
    if "size" in df.columns:
        try:
            avg_trade_size = float(
                pd.to_numeric(df["size"], errors="coerce").abs().mean()
            )
        except Exception:
            avg_trade_size = 0.0
    else:
        avg_trade_size = 0.0

    # Durée moyenne des trades en jours (si disponible)
    if "duration_days" in df.columns:
        try:
            avg_trade_duration_days = float(
                pd.to_numeric(df["duration_days"], errors="coerce")
                .dropna()
                .mean()
            )
        except Exception:
            avg_trade_duration_days = 0.0
    else:
        avg_trade_duration_days = 0.0

    # Plus grande série de pertes consécutives
    max_consecutive_losses = 0
    current_streak = 0
    try:
        loss_flags = df[pnl_col] < 0
        for is_loss in loss_flags:
            if bool(is_loss):
                current_streak += 1
                if current_streak > max_consecutive_losses:
                    max_consecutive_losses = current_streak
            else:
                current_streak = 0
    except Exception:
        max_consecutive_losses = 0

    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else (avg_win / 1.0 if avg_loss == 0 else 0.0)

    expectancy = ((won / total) * avg_win + (lost / total) * avg_loss) if total > 0 else 0.0

    return {
        "total_trades": total,
        "won_trades": won,
        "lost_trades": lost,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "max_consecutive_losses": int(max_consecutive_losses),
        "avg_trade_size": avg_trade_size,
        "avg_trade_duration_days": avg_trade_duration_days,
    }


def compute(
    equity: Optional[pd.Series],
    returns: pd.Series | Dict[Any, float],
    trades: Optional[pd.DataFrame] = None,
    *,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
    mar_annual: float = 0.0,
) -> Dict[str, Any]:
    """
    Calcule les métriques de performance avancées.

    Args:
        equity: Courbe d'equity (si None, recalculable depuis returns et capital de départ en amont)
        returns: Série de log-returns par période (ou dict {date: ret})
        trades: Table des trades (optionnelle)
        periods_per_year: Nombre de périodes par an (252 pour daily)
        risk_free_rate_annual: Taux sans risque (annuel)
        mar_annual: Minimum Acceptable Return (annuel)
    """
    r = _as_series(returns).astype(float).dropna()

    # Ratios (log-returns)
    sharpe = compute_sharpe(r, periods_per_year, risk_free_rate_annual)
    sortino = compute_sortino(r, periods_per_year, mar_annual)

    # VolatilitÃ© annuelle simple (log-returns)
    if not r.empty:
        ann_vol = float(r.std(ddof=0) * np.sqrt(periods_per_year))
    else:
        ann_vol = 0.0

    # Equity / CAGR
    cagr = compute_cagr(equity, periods_per_year) if equity is not None else 0.0

    # Stats de trades
    tstats = compute_trade_stats(trades)

    metrics = {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "cagr": cagr,
        "ann_vol": ann_vol,
        **tstats,
    }
    return metrics


# --- Analyzer pour historiser les trades détaillés ---
if bt is not None:

    class TradeListAnalyzer(bt.Analyzer):  # type: ignore
        def __init__(self):
            self.trades: List[Dict[str, Any]] = []
            self._open_trades: Dict[int, Dict[str, Any]] = {}

        def _to_datetime(self, value):
            if value is None:
                return None
            try:
                if hasattr(bt, "num2date"):
                    return bt.num2date(value)
            except Exception:
                return value
            return value

        def notify_trade(self, trade):
            # Stocker les infos à l'ouverture
            if trade.justopened:
                entry_dt = self._to_datetime(trade.dtopen)
                self._open_trades[trade.ref] = {
                    "entry_dt": entry_dt,
                    "entry_price": trade.price,
                    "size": trade.size,
                    "baropen": trade.baropen,
                }

            if not trade.isclosed:
                return

            try:
                entry_info = self._open_trades.pop(trade.ref, {})
                entry_price = entry_info.get("entry_price", trade.price)
                entry_size = entry_info.get("size", None)

                exit_dt = self._to_datetime(trade.dtclose)
                entry_dt = entry_info.get("entry_dt", self._to_datetime(trade.dtopen))

                exit_price = entry_price
                if entry_price is not None and entry_size not in (0, None):
                    try:
                        exit_price = entry_price + (trade.pnlcomm / entry_size)
                    except Exception:
                        exit_price = entry_price

                duration_bars = None
                baropen = entry_info.get("baropen", trade.baropen)
                if baropen is not None and trade.barclose is not None:
                    try:
                        duration_bars = int(trade.barclose) - int(baropen)
                    except Exception:
                        duration_bars = None

                duration_days = None
                if entry_dt is not None and exit_dt is not None:
                    try:
                        duration_days = (exit_dt - entry_dt).days
                    except Exception:
                        duration_days = None

                ret_pct = None
                if entry_price not in (0, None) and entry_size not in (0, None) and trade.pnlcomm is not None:
                    try:
                        ret_pct = (trade.pnlcomm / (abs(entry_size) * entry_price)) * 100.0
                    except Exception:
                        ret_pct = None

                fees = None
                if trade.pnl is not None and trade.pnlcomm is not None:
                    fees = trade.pnl - trade.pnlcomm

                info = {
                    "ref": getattr(trade, "ref", None),
                    "data_name": getattr(self.datas[0], "_name", "data0") if self.datas else "data0",
                    "size": entry_size,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_dt": entry_dt,
                    "exit_dt": exit_dt,
                    "duration_bars": duration_bars,
                    "duration_days": duration_days,
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "net_pnl": trade.pnlcomm,
                    "fees": fees,
                    "ret_pct": ret_pct,
                    "is_long": trade.long,
                }

                self.trades.append(info)
            except Exception:
                pass

        def get_analysis(self):  # type: ignore
            return self.trades

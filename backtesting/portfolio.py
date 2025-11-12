# --- 1. Bibliothèques natives ---
import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

# --- 2. Bibliothèques tierces ---
import numpy as np
import pandas as pd

# --- 3. Imports locaux ---
from backtesting.analyzers import drawdown as dd_analyzer
from backtesting.analyzers import performance as perf_analyzer

logger = logging.getLogger(__name__)

WeightConfig = Optional[Union[Sequence[float], Dict[str, float]]]


def normalize_weights(
    tickers: Sequence[str],
    weights_cfg: WeightConfig = None,
) -> pd.Series:
    """
    Retourne une série de poids normalisés alignée sur les tickers.

    Args:
        tickers: Séquence de tickers disponibles (ordre = pondération).
        weights_cfg: Liste/tuple (même ordre) ou dict {ticker: poids}.

    Raises:
        ValueError: Si les poids sont invalides ou ne correspondent pas aux tickers.
    """
    unique_tickers = [t for t in tickers if t]
    if not unique_tickers:
        raise ValueError("Impossible de normaliser les poids sans tickers valides.")

    if weights_cfg is None:
        weight = 1.0 / len(unique_tickers)
        return pd.Series([weight] * len(unique_tickers), index=unique_tickers)

    weights_series: pd.Series
    if isinstance(weights_cfg, dict):
        missing = [t for t in unique_tickers if t not in weights_cfg]
        if missing:
            raise ValueError(
                f"Poids manquants pour les tickers: {', '.join(missing)}."
            )
        weights_series = pd.Series(
            [float(weights_cfg[t]) for t in unique_tickers],
            index=unique_tickers,
        )
    else:
        cfg_list = list(weights_cfg)
        if len(cfg_list) != len(unique_tickers):
            raise ValueError(
                "Le nombre de poids doit correspondre au nombre de tickers."
            )
        weights_series = pd.Series(
            [float(value) for value in cfg_list],
            index=unique_tickers,
        )

    total = weights_series.sum()
    if total <= 0:
        raise ValueError("La somme des poids doit être strictement positive.")

    return weights_series / total


def aggregate_weighted_returns(
    returns_map: Dict[str, pd.Series],
    weights: pd.Series,
    alignment: str = "intersection",
) -> pd.Series:
    """
    Agrège des séries de rendements pondérées par ticker.

    Args:
        returns_map: Dictionnaire {ticker: pd.Series de rendements}.
        weights: Série pandas contenant les poids normalisés par ticker.
        alignment: 'intersection' (dates communes) ou 'union' (remplit les vides à 0).
    """
    if not returns_map:
        return pd.Series(dtype=float)

    alignment_mode = alignment.lower()
    join_method = "inner" if alignment_mode == "intersection" else "outer"

    aligned_series = []
    valid_tickers = []
    for ticker, series in returns_map.items():
        if ticker not in weights.index:
            logger.warning("Ticker '%s' absent des poids. Ignoré.", ticker)
            continue
        if series is None or series.empty:
            logger.warning("Série de rendements vide pour '%s'. Ignorée.", ticker)
            continue
        renamed = series.rename(ticker)
        aligned_series.append(renamed)
        valid_tickers.append(ticker)

    if not aligned_series:
        return pd.Series(dtype=float)

    returns_df = pd.concat(aligned_series, axis=1, join=join_method).sort_index()
    if returns_df.empty:
        return pd.Series(dtype=float)

    if join_method == "outer":
        returns_df = returns_df.fillna(0.0)

    weight_vector = weights.reindex(valid_tickers)
    weight_vector = weight_vector.fillna(0.0)
    weight_vector = weight_vector / weight_vector.sum()

    weighted_returns = returns_df.mul(weight_vector, axis=1).sum(axis=1)
    return weighted_returns


def compute_portfolio_metrics(
    returns: pd.Series,
    initial_capital: float,
    analytics_settings: Optional[Dict[str, Union[int, float, str]]] = None,
) -> Tuple[Dict[str, float], pd.Series, pd.Series, pd.Series]:
    """
    Calcule les métriques de portefeuille à partir d'une série de rendements.

    Args:
        returns: Série de rendements simples alignés dans le temps.
        initial_capital: Capital initial (utilisé pour reconstruire l'équity).
        analytics_settings: Paramètres (periods_per_year, risk_free_rate, mar, returns).

    Returns:
        Tuple (metrics, equity_curve, log_or_simple_returns, underwater_series)
    """
    if returns is None or returns.empty:
        raise ValueError("La série de rendements du portefeuille est vide.")

    analytics = analytics_settings or {}
    periods_per_year = int(analytics.get("periods_per_year", 252))
    risk_free = float(analytics.get("risk_free_rate", 0.0))
    mar = float(analytics.get("mar", 0.0))
    returns_mode = str(analytics.get("returns", "log")).lower()

    equity = (1.0 + returns).cumprod() * float(initial_capital)
    working_returns = (
        np.log1p(returns) if returns_mode == "log" else returns.copy()
    )

    dd_metrics, underwater = dd_analyzer.analyze(equity)
    perf_metrics = perf_analyzer.compute(
        equity=equity,
        returns=working_returns,
        trades=None,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=risk_free,
        mar_annual=mar,
    )

    max_dd = dd_metrics.get("max_drawdown", 0.0)
    cagr = perf_metrics.get("cagr", 0.0)
    try:
        calmar = perf_analyzer.compute_calmar(cagr, max_dd)
    except Exception:  # pragma: no cover - sécurité
        calmar = 0.0

    final_value = float(equity.iloc[-1])
    pnl_value = final_value - float(initial_capital)
    pnl_pct = (pnl_value / float(initial_capital)) * 100 if initial_capital else 0.0

    perf_metrics["calmar_ratio"] = calmar
    perf_metrics["max_drawdown"] = max_dd
    perf_metrics["ulcer_index"] = dd_metrics.get("ulcer_index", 0.0)
    perf_metrics["final_value"] = final_value
    perf_metrics["pnl"] = pnl_value
    perf_metrics["pnl_pct"] = pnl_pct
    perf_metrics.setdefault("expectancy", perf_metrics.get("expectancy", 0.0))

    return perf_metrics, equity, working_returns, underwater


__all__ = [
    "aggregate_weighted_returns",
    "compute_portfolio_metrics",
    "normalize_weights",
]

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


def analyze(equity: pd.Series) -> Tuple[Dict[str, Any], pd.Series]:
    """
    Analyse des drawdowns à partir d'une courbe d'equity.

    Returns:
        (metrics, underwater)
        - metrics: dict avec max_drawdown, start, trough, recovery, duration_bars, recovery_bars
        - underwater: série de drawdown (en proportion, négative ou 0)
    """
    if equity is None or len(equity) == 0:
        return (
            {
                "max_drawdown": 0.0,
                "max_drawdown_start": None,
                "max_drawdown_trough": None,
                "max_drawdown_recovery": None,
                "max_drawdown_duration": 0,
                "recovery_bars": None,
                "ulcer_index": 0.0,
            },
            pd.Series(dtype=float),
        )

    eq = equity.astype(float).copy().dropna()
    running_max = eq.cummax()
    underwater = eq / running_max - 1.0

    # Max drawdown
    trough_idx = underwater.idxmin()
    max_dd = float(underwater.loc[trough_idx]) if len(underwater) else 0.0

    # Début du DD (dernier sommet avant le trough)
    start_idx = running_max.loc[:trough_idx].idxmax() if len(underwater) else None

    # Recherche de la récupération (retour à 0 ou au-dessus)
    recovery_idx = None
    if start_idx is not None:
        after_trough = underwater.loc[trough_idx:]
        rec_candidates = after_trough[after_trough >= 0]
        if len(rec_candidates) > 0:
            recovery_idx = rec_candidates.index[0]

    # Durées
    duration_bars = 0
    recovery_bars = None
    if start_idx is not None and trough_idx is not None:
        try:
            duration_bars = int(eq.index.get_loc(trough_idx) - eq.index.get_loc(start_idx))
        except Exception:
            duration_bars = 0
    if recovery_idx is not None and start_idx is not None:
        try:
            recovery_bars = int(eq.index.get_loc(recovery_idx) - eq.index.get_loc(trough_idx))
        except Exception:
            recovery_bars = None

    # Ulcer index (racine de la moyenne des DD^2 > 0)
    dd_sq = underwater.copy()
    dd_sq = dd_sq[dd_sq < 0].pow(2)
    ulcer_index = float(np.sqrt(dd_sq.mean())) if len(dd_sq) else 0.0

    metrics = {
        "max_drawdown": abs(max_dd),  # en valeur positive pour lecture facile
        "max_drawdown_start": start_idx,
        "max_drawdown_trough": trough_idx,
        "max_drawdown_recovery": recovery_idx,
        "max_drawdown_duration": duration_bars,
        "recovery_bars": recovery_bars,
        "ulcer_index": ulcer_index,
    }

    return metrics, underwater

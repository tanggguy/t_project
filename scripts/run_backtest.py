#!/usr/bin/env python3
# --- 1. Bibliothèques natives ---
import sys
from pathlib import Path
import logging
import argparse
import importlib
import inspect
from typing import Any, Dict, Optional, Sequence, Tuple, Type

# --- 2. Bibliothèques tierces ---
import pandas as pd
import yaml
import numpy as np

# --- Configuration du Chemin ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except NameError:
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger
from utils.data_manager import DataManager
from utils.config_loader import get_settings
from backtesting.engine import BacktestEngine
from backtesting.portfolio import (
    aggregate_weighted_returns,
    compute_portfolio_metrics,
    normalize_weights,
)
from strategies.base_strategy import BaseStrategy
from risk_management.position_sizing import (
    FixedSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer,
)

# Initialisation du logger
logger = setup_logger(__name__, log_file="logs/backtest/run_backtest.log")


def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
    """
    Auto-découvre toutes les stratégies dans strategies/implementations/.

    Returns:
        Dict[str, Type[BaseStrategy]]: Dictionnaire {nom_stratégie: classe_stratégie}
    """
    strategies = {}
    strategies_dir = PROJECT_ROOT / "strategies" / "implementations"

    if not strategies_dir.exists():
        logger.error(f"Le dossier {strategies_dir} n'existe pas")
        return strategies

    # Parcourir tous les fichiers .py dans implementations/
    for py_file in strategies_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem

        try:
            # Import dynamique du module
            module = importlib.import_module(
                f"strategies.implementations.{module_name}"
            )

            # Chercher les classes qui héritent de BaseStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseStrategy)
                    and obj is not BaseStrategy
                    and name.endswith("Strategy")
                ):

                    # Extraire le nom court (sans "Strategy")
                    short_name = name.replace("Strategy", "")
                    strategies[short_name] = obj
                    logger.debug(f"Stratégie découverte: {short_name} ({name})")

        except Exception as e:
            logger.warning(f"Impossible d'importer {module_name}: {e}")

    return strategies


def _extract_price_series(
    df: pd.DataFrame,
    price_candidates: Sequence[str],
) -> Optional[pd.Series]:
    """Retourne une série de prix alignée sur un index date si possible."""
    price_col = next((col for col in price_candidates if col in df.columns), None)
    if price_col is None:
        return None

    series = pd.to_numeric(df[price_col], errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    else:
        date_col = next(
            (
                col
                for col in ("date", "Date", "datetime", "Datetime")
                if col in df.columns
            ),
            None,
        )
        if date_col is not None:
            index = pd.to_datetime(df[date_col], errors="coerce")
        else:
            index = pd.RangeIndex(len(series))

    index = pd.to_datetime(index, errors="coerce")
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if isinstance(index, pd.DatetimeIndex):
        index = index.normalize()

    price_series = pd.Series(series.values, index=index)
    price_series = price_series.dropna()
    return price_series if not price_series.empty else None


def _align_prices_to_equity(
    prices: pd.Series,
    equity_index: Optional[pd.Index],
) -> pd.Series:
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC").tz_localize(None)
    if isinstance(prices.index, pd.DatetimeIndex):
        prices.index = prices.index.normalize()

    target_index = None
    if equity_index is not None and len(equity_index) > 0:
        target_index = pd.to_datetime(equity_index, errors="coerce")
        if isinstance(target_index, pd.DatetimeIndex) and target_index.tz is not None:
            target_index = target_index.tz_convert("UTC").tz_localize(None)
        if isinstance(target_index, pd.DatetimeIndex):
            target_index = target_index.normalize()

    if target_index is not None and len(target_index) > 0:
        prices = prices.reindex(target_index).ffill().bfill()
    return prices.dropna()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path: Chemin vers le fichier YAML

    Returns:
        Dict contenant la configuration
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Fichier de configuration introuvable: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration chargée depuis: {config_path}")
    return config


def get_strategy_defaults(strategy_class: Type[BaseStrategy]) -> Dict[str, Any]:
    """
    Extrait les paramètres par défaut d'une stratégie.

    Args:
        strategy_class: Classe de la stratégie

    Returns:
        Dict des paramètres par défaut
    """
    defaults = {}

    if hasattr(strategy_class, "params"):
        # Backtrader stocke les params dans un objet spécial
        # La méthode la plus simple est d'itérer sur les attributs publics
        for param_name in dir(strategy_class.params):
            # Ignorer les attributs privés/magiques et les méthodes
            if not param_name.startswith("_") and not callable(
                getattr(strategy_class.params, param_name)
            ):
                try:
                    defaults[param_name] = getattr(strategy_class.params, param_name)
                except (TypeError, AttributeError):
                    pass

    return defaults


def merge_params(
    defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Fusionne les paramètres par défaut avec les overrides du YAML.

    Args:
        defaults: Paramètres par défaut de la stratégie
        overrides: Paramètres spécifiés dans le YAML (peut être None)

    Returns:
        Dict des paramètres finaux
    """
    final_params = defaults.copy()

    if overrides:
        final_params.update(overrides)

    return final_params


def print_results(results: list, initial_capital: float, data_df: pd.DataFrame) -> None:
    """
    Affiche les résultats de base du backtest.

    Args:
        results: Liste des stratégies exécutées par Cerebro
        initial_capital: Capital de départ
        data_df: DataFrame des données
    """
    if not results:
        logger.error("Aucun résultat de stratégie à analyser")
        return

    strat = results[0]

    # --- Analyseurs ---
    try:
        trades_analyzer = strat.analyzers.trades.get_analysis()
        sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
        returns_analyzer = strat.analyzers.returns.get_analysis()

    except KeyError as e:
        logger.error(f"Erreur: Analyseur manquant - {e}")
        return

    # --- Calculs ---
    final_value = strat.broker.getvalue()
    pnl = final_value - initial_capital
    pnl_pct = (pnl / initial_capital) * 100

    total_trades = trades_analyzer.get("total", {}).get("total", 0)
    won_trades = trades_analyzer.get("won", {}).get("total", 0)
    lost_trades = trades_analyzer.get("lost", {}).get("total", 0)

    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    sharpe_ratio = sharpe_analyzer.get("sharperatio")
    if sharpe_ratio is None:
        sharpe_ratio = "N/A"

    max_dd = drawdown_analyzer.get("max", {}).get("drawdown", 0)

    total_return = returns_analyzer.get("rtot", 0) * 100
    # Calculer un retour annualisé (CAGR) à partir des dates et des valeurs
    years = 0.0
    try:
        start_dt = data_df.index.min()
        end_dt = data_df.index.max()
        # gérer timezone-aware
        if hasattr(start_dt, "to_pydatetime"):
            start_dt = start_dt.to_pydatetime()
        if hasattr(end_dt, "to_pydatetime"):
            end_dt = end_dt.to_pydatetime()
        days = max((end_dt - start_dt).days, 0)
        years = days / 365.25 if days > 0 else 0.0
    except Exception:
        years = 0.0
    cagr = (
        ((final_value / initial_capital) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    )

    # --- Affichage ---
    print("\n" + "=" * 70)
    print("RÉSULTATS DU BACKTEST")
    print("=" * 70)

    print(f"\n📊 Période: {data_df.index.min().date()} à {data_df.index.max().date()}")
    print(f"📊 Nombre de bougies: {len(data_df)}")

    print(f"\n💰 PERFORMANCE")
    print(f"   Capital Initial:        {initial_capital:>15,.2f} €")
    print(f"   Capital Final:          {final_value:>15,.2f} €")
    print(f"   P&L:                    {pnl:>15,.2f} € ({pnl_pct:+.2f}%)")
    print(f"   Retour Total:           {total_return:>15.2f}%")
    print(f"   Retour annualisé (CAGR): {cagr * 100:>14.2f}%")

    print(f"\n📈 TRADES")
    print(f"   Nombre Total:           {total_trades:>15}")
    print(f"   Trades Gagnants:        {won_trades:>15}")
    print(f"   Trades Perdants:        {lost_trades:>15}")
    print(f"   Win Rate:               {win_rate:>14.2f}%")

    if total_trades > 0:
        avg_win = trades_analyzer.get("won", {}).get("pnl", {}).get("average", 0)
        avg_loss = trades_analyzer.get("lost", {}).get("pnl", {}).get("average", 0)

        print(f"   Gain Moyen:             {avg_win:>15,.2f} €")
        print(f"   Perte Moyenne:          {avg_loss:>15,.2f} €")

    print(f"\n📉 RISQUE")
    print(f"   Sharpe Ratio:           {str(sharpe_ratio):>15}")
    print(f"   Max Drawdown:           {max_dd:>14.2f}%")

    print("\n" + "=" * 70)


def save_results(
    results: list,
    config: Dict[str, Any],
    output_dir: str,
    data_df: pd.DataFrame,
    ticker_override: Optional[str] = None,
) -> None:
    """
    Sauvegarde les résultats dans un fichier.

    Args:
        results: Résultats du backtest
        config: Configuration utilisée
        output_dir: Répertoire de sortie
        data_df: DataFrame des données
    """
    if not results:
        return

    # Créer le répertoire si nécessaire
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strat = results[0]
    strategy_name = config["backtest"]["strategy"]
    ticker = ticker_override or config["backtest"]["data"]["ticker"]

    # Nom du fichier
    filename = (
        f"{strategy_name}_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    filepath = output_path / filename

    # Écrire les résultats
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RÉSULTATS DU BACKTEST\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Stratégie: {strategy_name}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(
            f"Période: {data_df.index.min().date()} à {data_df.index.max().date()}\n"
        )
        f.write(
            f"Capital Initial: {config['backtest']['broker']['initial_capital']:,.2f} €\n"
        )
        f.write(f"Capital Final: {strat.broker.getvalue():,.2f} €\n\n")

        f.write("Configuration:\n")
        f.write(yaml.dump(config, default_flow_style=False, sort_keys=False))

    logger.info(f"Résultats sauvegardés dans: {filepath}")


def configure_position_sizing(engine: BacktestEngine, config: Dict[str, Any]) -> None:
    """
    Configure le position sizing à partir de la configuration.

    Args:
        engine (BacktestEngine): L'instance du moteur de backtest.
        config (Dict[str, Any]): La configuration complète chargée du YAML.
    """
    bt_config = config.get("backtest", {})
    ps_config = bt_config.get("position_sizing", {})

    if not ps_config.get("enabled", False):
        logger.info("Position sizing désactivé - utilisation du sizing par défaut")
        return

    method = ps_config.get("method", "fixed")
    logger.info(f"Configuration du position sizing: {method}")

    try:
        if method == "fixed":
            # Position sizing fixe
            fixed_config = ps_config.get("fixed", {})
            stake = fixed_config.get("stake", None)
            pct_size = fixed_config.get("pct_size", 1.0)

            engine.add_sizer(FixedSizer, stake=stake, pct_size=pct_size)

            if stake:
                logger.info(f"Sizer fixe configuré: {stake} unités par trade")
            else:
                logger.info(f"Sizer fixe configuré: {pct_size:.1%} du capital")

        elif method == "fixed_fractional":
            # Position sizing basé sur le risque fixe
            ff_config = ps_config.get("fixed_fractional", {})
            risk_pct = ff_config.get("risk_pct", 0.02)
            stop_distance = ff_config.get("stop_distance", 0.03)

            engine.add_sizer(
                FixedFractionalSizer, risk_pct=risk_pct, stop_distance=stop_distance
            )

            logger.info(
                f"Sizer Fixed Fractional configuré: "
                f"Risque {risk_pct:.1%}, Stop {stop_distance:.1%}"
            )

        elif method == "volatility_based":
            # Position sizing basé sur la volatilité (ATR)
            vb_config = ps_config.get("volatility_based", {})
            risk_pct = vb_config.get("risk_pct", 0.02)
            atr_period = vb_config.get("atr_period", 14)
            atr_multiplier = vb_config.get("atr_multiplier", 2.0)

            engine.add_sizer(
                VolatilityBasedSizer,
                risk_pct=risk_pct,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
            )

            logger.info(
                f"Sizer Volatility-Based configuré: "
                f"Risque {risk_pct:.1%}, ATR {atr_period} x {atr_multiplier}"
            )

        else:
            logger.warning(
                f"Méthode de position sizing inconnue: {method}. "
                "Position sizing désactivé."
            )

    except Exception as e:
        logger.error(f"Erreur lors de la configuration du position sizing: {e}")
        logger.warning("Le backtest continuera sans position sizing configuré.")


def _load_analytics_settings() -> Dict[str, Any]:
    try:
        settings = get_settings()
        return settings.get("analytics", {})
    except Exception:
        logger.warning("Impossible de charger settings.yaml pour analytics.")
        return {}


def _execute_backtest_for_ticker(
    ticker: str,
    data_df: pd.DataFrame,
    strategy_class: Type[BaseStrategy],
    final_params: Dict[str, Any],
    broker_config: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    engine = BacktestEngine()
    initial_capital = float(broker_config.get("initial_capital", 10000.0))

    if "initial_capital" in broker_config:
        engine.cerebro.broker.setcash(initial_capital)
        logger.info(f"Capital initial ({ticker}): {initial_capital:,.2f}")

    if "commission_pct" in broker_config:
        comm_pct = float(broker_config["commission_pct"])
        engine.cerebro.broker.setcommission(commission=comm_pct)
        logger.info(f"Commission ({ticker}): {comm_pct:.4%}")
    elif "commission_fixed" in broker_config:
        comm_val = float(broker_config["commission_fixed"])
        engine.cerebro.broker.setcommission(commission=comm_val)
        logger.info(f"Commission fixe ({ticker}): {comm_val:.4f}")

    if "slippage_pct" in broker_config:
        slippage = float(broker_config["slippage_pct"])
        if slippage > 0:
            engine.cerebro.broker.set_slippage_perc(perc=slippage)
            logger.info(f"Slippage ({ticker}): {slippage:.4%}")

    engine.add_data(data_df)
    configure_position_sizing(engine, config)
    engine.add_strategy(strategy_class, **final_params)

    logger.info("Lancement du backtest pour %s...", ticker)
    results = engine.run()
    return {
        "ticker": ticker,
        "engine": engine,
        "results": results,
        "data": data_df,
        "initial_capital": initial_capital,
    }


def _extract_time_returns(strat: Any) -> pd.Series:
    try:
        returns_dict = strat.analyzers.timereturns.get_analysis()
        series = pd.Series(returns_dict)
        series.index = pd.to_datetime(series.index)
        return series.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _extract_trade_stats(strat: Any) -> Dict[str, float]:
    try:
        trades = strat.analyzers.trades.get_analysis()
    except Exception:
        trades = {}

    total = trades.get("total", {}).get("total", 0)
    won = trades.get("won", {}).get("total", 0)
    lost = trades.get("lost", {}).get("total", 0)
    avg_win = trades.get("won", {}).get("pnl", {}).get("average", 0.0) if won else 0.0
    avg_loss = (
        trades.get("lost", {}).get("pnl", {}).get("average", 0.0) if lost else 0.0
    )

    return {
        "total": total,
        "won": won,
        "lost": lost,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def _aggregate_trade_stats(
    stats_list: Sequence[Dict[str, float]],
    tickers: Sequence[str],
    weights: pd.Series,
) -> Dict[str, float]:
    """Agrège les stats de trades en tenant compte des poids de portefeuille.

    - total/won/lost: sommes brutes (pour lisibilité)
    - avg_win/avg_loss: moyennes pondérées par (nombre de trades) ET par poids
    - win_rate: won / total (non pondéré, pour rester intuitif)
    """
    total = sum(stat.get("total", 0) for stat in stats_list)
    won = sum(stat.get("won", 0) for stat in stats_list)
    lost = sum(stat.get("lost", 0) for stat in stats_list)

    # Construire vecteurs alignés ticker -> poids
    ticker_weights = [float(weights.get(t, 0.0)) for t in tickers]

    # Moyennes pondérées par (nb_trades * poids_ticker)
    won_den = 0.0
    won_num = 0.0
    lost_den = 0.0
    lost_num = 0.0
    for stat, w in zip(stats_list, ticker_weights):
        w = max(w, 0.0)
        won_i = stat.get("won", 0)
        lost_i = stat.get("lost", 0)
        avg_win_i = float(stat.get("avg_win", 0.0))
        avg_loss_i = float(stat.get("avg_loss", 0.0))
        if won_i:
            won_num += avg_win_i * won_i * w
            won_den += won_i * w
        if lost_i:
            lost_num += avg_loss_i * lost_i * w
            lost_den += lost_i * w

    avg_win = (won_num / won_den) if won_den > 0 else 0.0
    avg_loss = (lost_num / lost_den) if lost_den > 0 else 0.0
    win_rate = (won / total * 100) if total else 0.0

    return {
        "total": total,
        "won": won,
        "lost": lost,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_rate": win_rate,
    }


def _print_portfolio_summary(
    strategy_name: str,
    metrics: Dict[str, float],
    trade_stats: Dict[str, float],
    returns_count: int,
    tickers: Sequence[str],
    weights: pd.Series,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> None:
    print("" + "=" * 70)
    print("RESULTATS PORTEFEUILLE (Multi-Ticker)")
    print("=" * 70)
    print(f"Strategie: {strategy_name}")
    print(f"Tickers: {', '.join(tickers)}")
    weight_str = ", ".join(f"{t}: {weights.get(t, 0.0):.2%}" for t in tickers)
    print(f"Poids: {weight_str}")
    if start_date and end_date:
        print(
            f"Periode combinee: {start_date.date()} -> {end_date.date()} ({returns_count} points)"
        )

    print("PERFORMANCE")
    print(f"   Capital Initial: {metrics.get('initial_capital', 0):>15,.2f}")
    print(f"   Capital Final:   {metrics.get('final_value', 0):>15,.2f}")
    print(
        f"   P&L:             {metrics.get('pnl', 0):>15,.2f} ({metrics.get('pnl_pct', 0):+.2f}%)"
    )
    print(f"   Sharpe Ratio:    {metrics.get('sharpe_ratio', float('nan')):>15.2f}")
    # metrics['max_drawdown'] est une proportion (0.0-1.0). Afficher en %.
    print(f"   Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:>15.2f}%")

    print("TRADES AGREGERES")
    print(f"   Nombre Total:    {trade_stats.get('total', 0):>15}")
    print(f"   Gagnants:        {trade_stats.get('won', 0):>15}")
    print(f"   Perdants:        {trade_stats.get('lost', 0):>15}")
    print(f"   Win Rate:        {trade_stats.get('win_rate', 0):>14.2f}%")
    if trade_stats.get("won", 0):
        print(f"   Gain Moyen:      {trade_stats.get('avg_win', 0):>15,.2f}")
    if trade_stats.get("lost", 0):
        print(f"   Perte Moyenne:   {trade_stats.get('avg_loss', 0):>15,.2f}")

    print("" + "=" * 70)


def _save_portfolio_summary(
    output_dir: str,
    strategy_name: str,
    tickers: Sequence[str],
    weights: pd.Series,
    metrics: Dict[str, float],
    trade_stats: Dict[str, float],
    period: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{strategy_name}_Portfolio_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    filepath = output_path / filename
    start, end = period

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "")
        f.write("RAPPORT PORTFOLIO MULTI-TICKER")
        f.write("=" * 70 + "")
        f.write(f"Strategie: {strategy_name}")
        f.write(f"Tickers: {', '.join(tickers)}")
        f.write(
            "Poids: " + ", ".join(f"{t}={weights.get(t, 0):.2%}" for t in tickers) + ""
        )
        if start and end:
            f.write(f"Periode: {start.date()} -> {end.date()}")

        f.write("Performance:")
        f.write(f"  Capital initial: {metrics.get('initial_capital', 0):,.2f}")
        f.write(f"  Capital final:   {metrics.get('final_value', 0):,.2f}")
        f.write(f"  P&L:             {metrics.get('pnl', 0):,.2f}")
        f.write(f"  P&L %:           {metrics.get('pnl_pct', 0):.2f}%")
        f.write(f"  Sharpe:          {metrics.get('sharpe_ratio', float('nan')):.4f}")
        # Enregistrer le MDD en pourcentage (valeur stockée = proportion)
        f.write(f"  Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:.2f}%")

        f.write("Trades agreges:")
        f.write(f"  Total:    {trade_stats.get('total', 0)}")
        f.write(f"  Gagnants: {trade_stats.get('won', 0)}")
        f.write(f"  Perdants: {trade_stats.get('lost', 0)}")
        f.write(f"  Win Rate: {trade_stats.get('win_rate', 0):.2f}%")

    logger.info("Resume portefeuille sauvegarde dans: %s", filepath)


def _generate_strategy_report(
    strat: Any,
    ticker: str,
    initial_capital: float,
    data_df: pd.DataFrame,
    report_cfg: Dict[str, Any],
    strategy_name: str,
    analytics_settings: Dict[str, Any],
) -> None:
    try:
        from reports.report_generator import generate_report
    except Exception as exc:
        logger.error("Impossible d'importer generate_report: %s", exc)
        return

    returns = _extract_time_returns(strat)
    if returns.empty:
        logger.warning(
            "Impossible de generer un rapport HTML pour %s: retours indisponibles.",
            ticker,
        )
        return

    log_returns = np.log1p(returns)
    equity = (1.0 + returns).cumprod() * float(initial_capital)

    try:
        from backtesting.analyzers import drawdown as dd_an
        from backtesting.analyzers import performance as perf
    except Exception as exc:
        logger.error("Imports analyzers indisponibles: %s", exc)
        return

    dd_metrics, underwater = dd_an.analyze(equity)
    trades_df = None
    try:
        trade_list = strat.analyzers.tradelist.get_analysis()
        trades_df = pd.DataFrame(trade_list)
        if not trades_df.empty and "entry_dt" in trades_df.columns:
            trades_df = trades_df.sort_values(by="entry_dt").reset_index(drop=True)
    except Exception:
        trades_df = None

    periods_per_year = analytics_settings.get("periods_per_year", 252)
    risk_free = analytics_settings.get("risk_free_rate", 0.0)
    mar = analytics_settings.get("mar", 0.0)

    perf_metrics = perf.compute(
        equity=equity,
        returns=log_returns,
        trades=trades_df,
        periods_per_year=periods_per_year,
        risk_free_rate_annual=risk_free,
        mar_annual=mar,
    )

    max_dd = dd_metrics.get("max_drawdown", 0.0)
    cagr = perf_metrics.get("cagr", 0.0)
    try:
        calmar = perf.compute_calmar(cagr, max_dd)
    except Exception:
        calmar = 0.0

    perf_metrics["calmar_ratio"] = calmar
    perf_metrics["max_drawdown"] = max_dd
    perf_metrics["ulcer_index"] = dd_metrics.get("ulcer_index", 0.0)
    final_value = strat.broker.getvalue()
    pnl_value = final_value - initial_capital
    perf_metrics["final_value"] = final_value
    perf_metrics["pnl"] = pnl_value
    perf_metrics["pnl_pct"] = (
        (pnl_value / initial_capital) * 100 if initial_capital else 0.0
    )
    perf_metrics.setdefault("expectancy", perf_metrics.get("expectancy", 0.0))

    benchmark_payload: Optional[Dict[str, Any]] = None
    price_columns = ["adj_close", "Adj Close", "AdjClose", "close", "Close"]
    price_series = _extract_price_series(data_df, price_columns)
    if price_series is not None:
        equity_index = equity.index if equity is not None else price_series.index
        aligned_prices = _align_prices_to_equity(price_series, equity_index)
        if not aligned_prices.empty and float(aligned_prices.iloc[0]) != 0.0:
            shares = initial_capital / float(aligned_prices.iloc[0])
            bh_equity = aligned_prices * shares
            bh_returns = bh_equity.pct_change().fillna(0.0)
            bh_log_returns = np.log1p(bh_returns)
            bh_dd_metrics, _ = dd_an.analyze(bh_equity)
            bh_perf = perf.compute(
                equity=bh_equity,
                returns=bh_log_returns,
                trades=None,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=risk_free,
                mar_annual=mar,
            )
            bh_max_dd = bh_dd_metrics.get("max_drawdown", 0.0)
            bh_cagr = bh_perf.get("cagr", 0.0)
            try:
                bh_calmar = perf.compute_calmar(bh_cagr, bh_max_dd)
            except Exception:
                bh_calmar = 0.0

            bh_final_value = float(bh_equity.iloc[-1])
            bh_pnl = bh_final_value - initial_capital
            bh_perf.update(
                {
                    "calmar_ratio": bh_calmar,
                    "max_drawdown": bh_max_dd,
                    "ulcer_index": bh_dd_metrics.get("ulcer_index", 0.0),
                    "final_value": bh_final_value,
                    "pnl": bh_pnl,
                    "pnl_pct": (
                        (bh_pnl / initial_capital) * 100 if initial_capital else 0.0
                    ),
                }
            )
            benchmark_payload = {
                "name": "Buy & Hold",
                "metrics": bh_perf,
                "equity": bh_equity,
            }

    out_dir = report_cfg.get("out_dir", "reports/generated")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = (
        f"{strategy_name}_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    out_path = str(Path(out_dir) / out_file)
    template = report_cfg.get("template", "default.html")
    logger.info("Benchmark payload for %s: %s", ticker, bool(benchmark_payload))

    generate_report(
        meta={
            "strategy_name": strategy_name,
            "ticker": ticker,
            "start_date": data_df.index.min().date() if not data_df.empty else None,
            "end_date": data_df.index.max().date() if not data_df.empty else None,
        },
        metrics=perf_metrics,
        equity=equity,
        underwater=underwater,
        trades=trades_df,
        out_path=out_path,
        template=template,
        returns=returns,
        log_returns=log_returns,
        analytics_config={
            "periods_per_year": periods_per_year,
            "risk_free_rate": risk_free,
            "rolling_window": analytics_settings.get("rolling_window", 63),
        },
        benchmark=benchmark_payload,
    )


def _generate_portfolio_report(
    strategy_name: str,
    tickers: Sequence[str],
    weights: pd.Series,
    metrics: Dict[str, float],
    equity: pd.Series,
    underwater: pd.Series,
    portfolio_returns: pd.Series,
    working_returns: pd.Series,
    report_cfg: Dict[str, Any],
    analytics_settings: Dict[str, Any],
    *,
    data_frames: Optional[Dict[str, pd.DataFrame]] = None,
    initial_capital: float = 0.0,
) -> None:
    try:
        from reports.report_generator import generate_report
    except Exception as exc:
        logger.error("Impossible d'importer generate_report: %s", exc)
        return

    out_dir = report_cfg.get("out_dir", "reports/generated")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = (
        f"{strategy_name}_Portfolio_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    template = report_cfg.get("template", "default.html")

    meta = {
        "strategy_name": strategy_name,
        "ticker": "Portfolio",
        "tickers": ", ".join(tickers),
        "weights": {t: weights.get(t, 0.0) for t in tickers},
        "start_date": (
            portfolio_returns.index.min().date()
            if not portfolio_returns.empty
            else None
        ),
        "end_date": (
            portfolio_returns.index.max().date()
            if not portfolio_returns.empty
            else None
        ),
    }

    benchmark_payload: Optional[Dict[str, Any]] = None
    if data_frames:
        try:
            from backtesting.analyzers import drawdown as dd_an
            from backtesting.analyzers import performance as perf
        except Exception as exc:
            logger.error("Imports analyzers indisponibles pour benchmark: %s", exc)
        else:
            price_columns = ["adj_close", "Adj Close", "AdjClose", "close", "Close"]
            price_series: Dict[str, pd.Series] = {}
            for ticker in tickers:
                df = data_frames.get(ticker)
                if df is None or df.empty:
                    continue
                series = _extract_price_series(df, price_columns)
                if series is None:
                    continue
                price_series[ticker] = series

            if price_series and initial_capital > 0:
                try:
                    price_df = pd.concat(
                        price_series, axis=1, join="outer"
                    ).sort_index()
                except Exception:
                    price_df = pd.DataFrame(price_series).sort_index()
                price_df = price_df.ffill().dropna(how="all")
                if not price_df.empty:
                    aligned_weights = weights.reindex(price_df.columns).fillna(0.0)
                    total_weight = aligned_weights.sum()
                    if total_weight > 0:
                        aligned_weights = aligned_weights / total_weight
                        start_prices = price_df.iloc[0].replace(0, np.nan)
                        shares = (
                            (initial_capital * aligned_weights)
                            .divide(start_prices)
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                        )
                        bh_equity = (price_df * shares).sum(axis=1)
                        if equity is not None and not equity.empty:
                            bh_equity = _align_prices_to_equity(bh_equity, equity.index)
                        else:
                            bh_equity = bh_equity.dropna()

                        if not bh_equity.empty:
                            bh_returns = bh_equity.pct_change().fillna(0.0)
                            bh_log_returns = np.log1p(bh_returns)
                            bh_dd_metrics, _ = dd_an.analyze(bh_equity)
                            bh_perf = perf.compute(
                                equity=bh_equity,
                                returns=bh_log_returns,
                                trades=None,
                                periods_per_year=analytics_settings.get(
                                    "periods_per_year", 252
                                ),
                                risk_free_rate_annual=analytics_settings.get(
                                    "risk_free_rate", 0.0
                                ),
                                mar_annual=analytics_settings.get("mar", 0.0),
                            )
                            bh_max_dd = bh_dd_metrics.get("max_drawdown", 0.0)
                            bh_cagr = bh_perf.get("cagr", 0.0)
                            try:
                                bh_calmar = perf.compute_calmar(bh_cagr, bh_max_dd)
                            except Exception:
                                bh_calmar = 0.0

                            bh_final_value = float(bh_equity.iloc[-1])
                            bh_pnl = bh_final_value - initial_capital
                            bh_perf.update(
                                {
                                    "calmar_ratio": bh_calmar,
                                    "max_drawdown": bh_max_dd,
                                    "ulcer_index": bh_dd_metrics.get(
                                        "ulcer_index", 0.0
                                    ),
                                    "final_value": bh_final_value,
                                    "pnl": bh_pnl,
                                    "pnl_pct": (
                                        (bh_pnl / initial_capital) * 100
                                        if initial_capital
                                        else 0.0
                                    ),
                                }
                            )
                            benchmark_payload = {
                                "name": "Buy & Hold",
                                "metrics": bh_perf,
                                "equity": bh_equity,
                            }

    generate_report(
        meta=meta,
        metrics=metrics,
        equity=equity,
        underwater=underwater,
        trades=None,
        out_path=str(Path(out_dir) / out_file),
        template=template,
        returns=portfolio_returns,
        log_returns=working_returns,
        analytics_config={
            "periods_per_year": analytics_settings.get("periods_per_year", 252),
            "risk_free_rate": analytics_settings.get("risk_free_rate", 0.0),
            "rolling_window": analytics_settings.get("rolling_window", 63),
        },
        benchmark=benchmark_payload,
    )


def _handle_single_run_outputs(
    run_info: Dict[str, Any],
    config: Dict[str, Any],
    output_config: Dict[str, Any],
    report_cfg: Dict[str, Any],
    report_enabled: bool,
    analytics_settings: Dict[str, Any],
    verbose: bool,
    save_results_enabled: bool,
    per_ticker_reports: bool,
) -> None:
    ticker = run_info["ticker"]
    results = run_info["results"]
    data_df = run_info["data"]
    initial_capital = run_info["initial_capital"]

    if verbose:
        print_results(results, initial_capital, data_df)

    if save_results_enabled:
        save_results(
            results,
            config,
            output_config.get("results_dir", "results/backtests"),
            data_df,
            ticker_override=ticker,
        )

    if report_enabled and per_ticker_reports:
        strat = results[0]
        _generate_strategy_report(
            strat,
            ticker,
            initial_capital,
            data_df,
            report_cfg,
            config["backtest"]["strategy"],
            analytics_settings,
        )


def _run_multi_ticker_backtest(
    tickers: Sequence[str],
    data_manager: DataManager,
    data_config: Dict[str, Any],
    bt_config: Dict[str, Any],
    strategy_class: Type[BaseStrategy],
    final_params: Dict[str, Any],
    broker_config: Dict[str, Any],
    config: Dict[str, Any],
    output_config: Dict[str, Any],
    strategy_name: str,
) -> None:
    portfolio_cfg = bt_config.get("portfolio", {})
    alignment = str(portfolio_cfg.get("alignment", "intersection")).lower()
    per_ticker_reports = bool(portfolio_cfg.get("per_ticker_reports", True))

    report_cfg = output_config.get("report", {})
    report_enabled = report_cfg.get("enable", output_config.get("report_enable", False))
    verbose = output_config.get("verbose", True)
    save_results_enabled = output_config.get("save_results", False)
    analytics_settings = _load_analytics_settings()

    data_frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = data_manager.get_data(
            ticker=ticker,
            start_date=data_config.get("start_date"),
            end_date=data_config.get("end_date"),
            interval=data_config.get("interval"),
            use_cache=data_config.get("use_cache", True),
        )
        if df is None or df.empty:
            logger.error("Impossible de charger les donnees pour %s", ticker)
            continue
        data_frames[ticker] = df

    if not data_frames:
        logger.error("Aucun ticker valide pour le mode multi-ticker.")
        return

    try:
        weights = normalize_weights(
            list(data_frames.keys()), data_config.get("weights")
        )
    except ValueError as exc:
        logger.error("Configuration des poids invalide: %s", exc)
        return

    run_entries = []
    for ticker, df in data_frames.items():
        run_info = _execute_backtest_for_ticker(
            ticker,
            df,
            strategy_class,
            final_params,
            broker_config,
            config,
        )
        if not run_info.get("results"):
            logger.error("Backtest vide pour %s", ticker)
            continue

        strat = run_info["results"][0]
        run_info["returns"] = _extract_time_returns(strat)
        run_info["trade_stats"] = _extract_trade_stats(strat)
        run_entries.append(run_info)

        _handle_single_run_outputs(
            run_info,
            config,
            output_config,
            report_cfg,
            report_enabled,
            analytics_settings,
            verbose,
            save_results_enabled,
            per_ticker_reports,
        )

    if not run_entries:
        logger.error("Aucun backtest valide n'a ete execute pour les tickers fournis.")
        return

    returns_map = {
        entry["ticker"]: entry.get("returns", pd.Series(dtype=float))
        for entry in run_entries
    }
    portfolio_returns = aggregate_weighted_returns(
        returns_map,
        weights,
        alignment=alignment,
    )

    if portfolio_returns.empty:
        logger.error("Impossible de calculer la serie de rendements du portefeuille.")
        return

    initial_capital = float(broker_config.get("initial_capital", 10000.0))
    metrics, equity, working_returns, underwater = compute_portfolio_metrics(
        portfolio_returns,
        initial_capital,
        analytics_settings,
    )
    metrics["initial_capital"] = initial_capital

    trade_stats = _aggregate_trade_stats(
        [entry["trade_stats"] for entry in run_entries],
        [entry["ticker"] for entry in run_entries],
        weights,
    )
    start_date = min(
        entry["data"].index.min() for entry in run_entries if not entry["data"].empty
    )
    end_date = max(
        entry["data"].index.max() for entry in run_entries if not entry["data"].empty
    )

    _print_portfolio_summary(
        strategy_name,
        metrics,
        trade_stats,
        len(portfolio_returns),
        list(data_frames.keys()),
        weights,
        start_date,
        end_date,
    )

    if save_results_enabled:
        _save_portfolio_summary(
            output_config.get("results_dir", "results/backtests"),
            strategy_name,
            list(data_frames.keys()),
            weights,
            metrics,
            trade_stats,
            (start_date, end_date),
        )

    if report_enabled:
        _generate_portfolio_report(
            strategy_name,
            list(data_frames.keys()),
            weights,
            metrics,
            equity,
            underwater,
            portfolio_returns,
            working_returns,
            report_cfg,
            analytics_settings,
            data_frames=data_frames,
            initial_capital=initial_capital,
        )

    if output_config.get("plot", False):
        logger.warning("Plot indisponible en mode multi-ticker (non supporte).")


def main() -> None:
    """
    Fonction principale du script.
    """
    # --- Arguments CLI ---
    parser = argparse.ArgumentParser(
        description="Lance un backtest avec configuration YAML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/backtest_config.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Affiche la liste des stratégies disponibles",
    )

    args = parser.parse_args()

    # --- Découverte des stratégies ---
    available_strategies = discover_strategies()

    if args.list_strategies:
        print("\n📋 Stratégies disponibles:")
        for name, cls in available_strategies.items():
            defaults = get_strategy_defaults(cls)
            print(f"\n  • {name} ({cls.__name__})")
            if defaults:
                print(f"    Paramètres par défaut:")
                for param, value in defaults.items():
                    print(f"      - {param}: {value}")
        return

    if not available_strategies:
        logger.error("Aucune stratégie trouvée dans strategies/implementations/")
        return

    logger.info(f"Stratégies découvertes: {', '.join(available_strategies.keys())}")

    # --- Chargement de la configuration ---
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la config: {e}")
        return

    # --- Extraction des paramètres ---
    bt_config = config.get("backtest", {})
    strategy_name = bt_config.get("strategy")

    if not strategy_name:
        logger.error("Aucune stratégie spécifiée dans le fichier de configuration")
        return

    if strategy_name not in available_strategies:
        logger.error(
            f"Stratégie '{strategy_name}' introuvable. "
            f"Disponibles: {', '.join(available_strategies.keys())}"
        )
        return

    strategy_class = available_strategies[strategy_name]
    logger.info(f"Stratégie sélectionnée: {strategy_name} ({strategy_class.__name__})")

    # --- Paramètres de la stratégie ---
    default_params = get_strategy_defaults(strategy_class)
    config_params = bt_config.get("strategy_params", {})
    final_params = merge_params(default_params, config_params)

    logger.info(f"Paramètres de la stratégie: {final_params}")

    # --- Chargement des données ---
    data_config = bt_config.get("data", {})
    ticker = data_config.get("ticker", "AAPL")
    tickers_cfg = data_config.get("tickers")
    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    data_manager = DataManager()

    broker_config = bt_config.get("broker", {})
    output_config = bt_config.get("output", {})
    initial_capital = broker_config.get("initial_capital", 10000.0)

    ticker_list: Sequence[str] = []
    if tickers_cfg:
        if isinstance(tickers_cfg, str):
            candidates = [tickers_cfg]
        else:
            candidates = list(tickers_cfg)
        ticker_list = [str(t).strip() for t in candidates if str(t).strip()]

    if ticker_list:
        if not (start_date and end_date):
            logger.error(
                "Le mode multi-ticker requiert 'start_date' et 'end_date' définis."
            )
            return

        _run_multi_ticker_backtest(
            tickers=ticker_list,
            data_manager=data_manager,
            data_config=data_config,
            bt_config=bt_config,
            strategy_class=strategy_class,
            final_params=final_params,
            broker_config=broker_config,
            config=config,
            output_config=output_config,
            strategy_name=strategy_name,
        )
        return

    logger.info(f"Chargement des données pour {ticker}...")

    if start_date and end_date:
        df = data_manager.get_data(
            ticker=ticker, start_date=start_date, end_date=end_date
        )
    else:
        logger.error("Spécifiez 'period' OU 'start_date' + 'end_date' dans la config")
        return

    if df is None or df.empty:
        logger.error(f"Impossible de charger les données pour {ticker}")
        return

    logger.info(
        f"Données chargées: {len(df)} bougies "
        f"({df.index.min().date()} à {df.index.max().date()})"
    )

    # --- Initialisation et lancement du backtest ---
    engine = BacktestEngine()

    # Override du capital si spécifié dans config
    if "initial_capital" in broker_config:
        engine.cerebro.broker.setcash(initial_capital)
        logger.info(f"Capital initial (config): {initial_capital:,.2f} €")

    # Override des commissions si spécifié
    if "commission_pct" in broker_config:
        comm_pct = broker_config["commission_pct"]
        engine.cerebro.broker.setcommission(commission=comm_pct)
        logger.info(f"Commission (config): {comm_pct:.4%}")

    # Override du slippage si spécifié
    if "slippage_pct" in broker_config:
        slippage = broker_config["slippage_pct"]
        if slippage > 0:
            engine.cerebro.broker.set_slippage_perc(perc=slippage)
            logger.info(f"Slippage (config): {slippage:.4%}")

    # Ajouter les données
    engine.add_data(df)
    # Configurer le position sizing
    configure_position_sizing(engine, config)
    # Ajouter la stratégie
    engine.add_strategy(strategy_class, **final_params)

    logger.info("Lancement du backtest...")
    results = engine.run()

    # --- Affichage des résultats ---
    verbose = output_config.get("verbose", True)

    if verbose:
        print_results(results, initial_capital, df)

    # --- Sauvegarde ---
    if output_config.get("save_results", False):
        results_dir = output_config.get("results_dir", "results/backtests")
        save_results(results, config, results_dir, df)

    # --- Reporting HTML (via YAML) ---
    try:
        report_cfg = output_config.get("report", {})
        report_enabled = report_cfg.get(
            "enable", output_config.get("report_enable", False)
        )
        if report_enabled:
            from backtesting.analyzers import drawdown as dd_an
            from backtesting.analyzers import performance as perf
            from reports.report_generator import generate_report

            strat = results[0]

            # Série de rendements (TimeReturn)
            try:
                rt_dict = strat.analyzers.timereturns.get_analysis()
                returns = pd.Series(rt_dict)
                # Assurer ordre chronologique & datetime index
                try:
                    returns.index = pd.to_datetime(returns.index)
                except Exception:
                    pass
                returns = returns.sort_index()
            except Exception:
                returns = pd.Series(dtype=float)

            # Equity
            equity = None
            if not returns.empty:
                equity = pd.Series(
                    (1.0 + returns).cumprod() * initial_capital, index=returns.index
                )

            # Underwater / Drawdown
            dd_metrics, underwater = (
                dd_an.analyze(equity)
                if equity is not None
                else ({}, pd.Series(dtype=float))
            )

            # Trade list (custom analyzer si dispo)
            trades_df = None
            try:
                trade_list = strat.analyzers.tradelist.get_analysis()
                trades_df = pd.DataFrame(trade_list)
                if not trades_df.empty:
                    if "entry_dt" in trades_df.columns:
                        trades_df = trades_df.sort_values(by="entry_dt")
                    trades_df = trades_df.reset_index(drop=True)
                    trades_df = trades_df.replace({np.nan: None})
            except Exception:
                trades_df = None

            # Paramètres analytics (defaults si absents)
            from utils.config_loader import get_settings

            settings = get_settings()
            analytics = settings.get("analytics", {})
            periods_per_year = analytics.get("periods_per_year", 252)
            rf = analytics.get("risk_free_rate", 0.0)
            mar = analytics.get("mar", 0.0)
            rolling_window = analytics.get("rolling_window", 63)

            # Log-returns pour ratios
            log_returns = None
            if not returns.empty:
                log_returns = np.log1p(returns)
                log_returns = pd.Series(log_returns, index=returns.index)

            # Performance metrics
            perf_metrics = perf.compute(
                equity=equity,
                returns=log_returns if log_returns is not None else returns,
                trades=trades_df,
                periods_per_year=periods_per_year,
                risk_free_rate_annual=rf,
                mar_annual=mar,
            )

            # Combiner avec drawdown pour Calmar
            max_dd = dd_metrics.get("max_drawdown", 0.0)
            cagr = perf_metrics.get("cagr", 0.0)
            calmar = 0.0
            try:
                calmar = perf.compute_calmar(cagr, max_dd)
            except Exception:
                pass
        perf_metrics["calmar_ratio"] = calmar
        perf_metrics["max_drawdown"] = max_dd
        perf_metrics["ulcer_index"] = dd_metrics.get("ulcer_index", 0.0)

        # Assurer compatibilité Jinja2 (pas d'inf)
        try:
            import math as _math

            if _math.isinf(perf_metrics.get("profit_factor", 0.0)):
                perf_metrics["profit_factor"] = float("nan")
        except Exception:
            pass

        final_value = strat.broker.getvalue()
        pnl_value = final_value - initial_capital
        pnl_pct_value = (pnl_value / initial_capital) * 100 if initial_capital else 0.0
        perf_metrics["final_value"] = final_value
        perf_metrics["pnl"] = pnl_value
        perf_metrics["pnl_pct"] = pnl_pct_value
        perf_metrics.setdefault("expectancy", perf_metrics.get("expectancy", 0.0))

        # Meta
        meta = {
            "strategy_name": strategy_name,
            "ticker": ticker,
            "start_date": df.index.min().date() if not df.empty else None,
            "end_date": df.index.max().date() if not df.empty else None,
        }

        out_dir = report_cfg.get("out_dir", "reports/generated")
        out_file = f"{strategy_name}_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        out_path = str(Path(out_dir) / out_file)
        template = report_cfg.get("template", "default.html")

        generate_report(
            meta=meta,
            metrics=perf_metrics,
            equity=equity if equity is not None else pd.Series(dtype=float),
            underwater=underwater if underwater is not None else pd.Series(dtype=float),
            trades=trades_df,
            out_path=out_path,
            template=template,
            returns=returns,
            log_returns=log_returns,
            analytics_config={
                "periods_per_year": periods_per_year,
                "risk_free_rate": rf,
                "rolling_window": rolling_window,
            },
        )
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport HTML: {e}")

    # --- Plot ---
    if output_config.get("plot", False):
        logger.info("Affichage des graphiques...")
        engine.cerebro.plot(style="candlestick")


if __name__ == "__main__":
    main()

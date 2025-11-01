#!/usr/bin/env python3
# --- 1. Bibliothèques natives ---
import sys
from pathlib import Path
import logging
import argparse
import importlib
import inspect
from typing import Dict, Any, Type, Optional

# --- 2. Bibliothèques tierces ---
import pandas as pd
import yaml

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
from backtesting.engine import BacktestEngine
from strategies.base_strategy import BaseStrategy
from strategies.implementations.ma_crossover import MaCrossoverStrategy
from strategies.implementations.rsi_oversold import RsiOversoldStrategy
from strategies.implementations.macd_momentum import MacdMomentumStrategy

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
        for param in strategy_class.params:
            if isinstance(param, tuple) and len(param) == 2:
                param_name, default_value = param
                defaults[param_name] = default_value

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
    avg_return = returns_analyzer.get("ravg", 0) * 100

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
    print(f"   Retour Moyen (annuel):  {avg_return:>15.2f}%")

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
    results: list, config: Dict[str, Any], output_dir: str, data_df: pd.DataFrame
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
    ticker = config["backtest"]["data"]["ticker"]

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
    period = data_config.get("period")
    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")

    logger.info(f"Chargement des données pour {ticker}...")
    data_manager = DataManager()

    if period:
        df = data_manager.get_data(ticker=ticker, period=period)
    elif start_date and end_date:
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

    # --- Configuration du broker ---
    broker_config = bt_config.get("broker", {})
    initial_capital = broker_config.get("initial_capital", 10000.0)

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

    engine.add_data(df)
    engine.add_strategy(strategy_class, **final_params)

    logger.info("Lancement du backtest...")
    results = engine.run()

    # --- Affichage des résultats ---
    output_config = bt_config.get("output", {})
    verbose = output_config.get("verbose", True)

    if verbose:
        print_results(results, initial_capital, df)

    # --- Sauvegarde ---
    if output_config.get("save_results", False):
        results_dir = output_config.get("results_dir", "results/backtests")
        save_results(results, config, results_dir, df)

    # --- Plot ---
    if output_config.get("plot", False):
        logger.info("Affichage des graphiques...")
        engine.cerebro.plot(style="candlestick")


if __name__ == "__main__":
    main()

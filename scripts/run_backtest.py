# --- 1. Bibliothèques natives ---
import sys
from pathlib import Path
import logging

# --- Configuration du Chemin (Important pour les scripts) ---
# Ajoute la racine du projet au PYTHONPATH pour que les imports (utils, etc.) fonctionnent
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except NameError:
    # Si __file__ n'est pas défini (ex: dans un notebook interactif)
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger
from utils.data_manager import DataManager
from backtesting.engine import BacktestEngine
from strategies.implementations.ma_crossover import MaCrossoverStrategy
from utils.config_loader import get_settings  # Pour lire le capital initial

# Initialisation du logger pour ce script
logger = setup_logger(__name__, log_file="logs/backtest/run_backtest.log")


def print_results(results: list, initial_capital: float) -> None:
    """
    Affiche les résultats de base du backtest.
    """

    # --- CORRECTION ---
    # Le traceback montre que le simple fait de vérifier la "truthiness"
    # de results[0] (if not results[0]) déclenche une erreur interne
    # de backtrader (AttributeError: 'NoneType' object has no attribute 'addindicator').
    # Si results est une liste vide, nous le capterons.
    # Si engine.run() a réussi, nous supposons que results[0] est un
    # objet stratégie valide.
    if not results:
        logger.error(
            "Aucun résultat de stratégie à analyser (liste de résultats vide)."
        )
        return

    # Nous accédons directement à l'objet sans le tester.
    strat = results[0]
    # --- FIN CORRECTION ---

    # --- Analyseurs ---
    try:
        trades_analyzer = strat.analyzers.trades.get_analysis()
        sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
        drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
        returns_analyzer = strat.analyzers.returns.get_analysis()

    except KeyError as e:
        logger.error(f"Erreur: Analyseur manquant - {e}. Le backtest a-t-il échoué ?")
        return

    # --- Métriques de base ---
    final_value = strat.broker.getvalue()
    pnl = final_value - initial_capital
    pnl_pct = (pnl / initial_capital) * 100
    total_trades = trades_analyzer.get("total", {}).get("total", 0)
    sharpe_ratio = sharpe_analyzer.get("sharperatio", None)
    max_drawdown = drawdown_analyzer.get("max", {}).get("drawdown", 0) * 100  # en %

    # --- Affichage (style 'print' pour un script CLI, 'logger.info' pour les logs) ---
    report = f"""
    ==================================================
    🏁 RÉSULTATS DU BACKTEST - {strat.strategy_name}
    ==================================================
    Période analysée: {strat.data.datetime.date(0).isoformat()} à {strat.data.datetime.date(-1).isoformat()}
    
    📈 Performance:
    --------------------------------------------------
    Capital initial:    {initial_capital:,.2f} €
    Portefeuille final: {final_value:,.2f} €
    P&L Net:            {pnl:,.2f} € ({pnl_pct:,.2f} %)
    
    📊 Statistiques:
    --------------------------------------------------
    Nombre total de trades: {total_trades}
    Ratio de Sharpe:        {sharpe_ratio if sharpe_ratio else 'N/A' :.4f}
    Drawdown Max:           {max_drawdown:.2f} %
    
    (Note: Les logs détaillés des trades sont dans le fichier .log)
    ==================================================
    """
    print(report)
    logger.info(f"Rapport de backtest généré. Portefeuille final: {final_value:,.2f}")


def main() -> None:
    """
    Fonction principale pour exécuter le backtest de bout en bout.
    """
    logger.info("--- Démarrage du script run_backtest.py ---")

    # --- Paramètres du Test (Tâche 3.4) ---
    TICKER: str = "AAPL"
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2023-12-31"  # 2 ans
    STRATEGY_PARAMS: dict = {
        "fast_period": 10,
        "slow_period": 30,
    }

    try:
        # 1. Charger les données
        logger.info(
            f"Chargement des données pour {TICKER} de {START_DATE} à {END_DATE}"
        )
        dm = DataManager()
        data_df = dm.get_data(
            ticker=TICKER,
            start_date=START_DATE,
            end_date=END_DATE,
            add_indicators=True,  # Important pour nettoyer les NaNs de chauffe
            use_cache=True,
        )

        if data_df.empty:
            logger.error(f"Aucune donnée chargée pour {TICKER}. Arrêt.")
            return

        # 2. Initialiser le moteur de Backtest
        engine = BacktestEngine()

        # 3. Ajouter les données au moteur
        engine.add_data(data_df, name=TICKER)

        # 4. Ajouter la stratégie au moteur
        engine.add_strategy(MaCrossoverStrategy, **STRATEGY_PARAMS)

        # 5. Lancer le backtest
        results = engine.run()

        # 6. Afficher les résultats
        initial_capital = (
            get_settings().get("backtest", {}).get("initial_capital", 10000.0)
        )
        print_results(results, initial_capital)

        # 7. Afficher le graphique
        logger.info("Affichage du graphique (fermez la fenêtre pour quitter)...")
        engine.plot()

        logger.info("--- Fin du script run_backtest.py ---")

    except Exception as e:
        logger.critical(f"Une erreur fatale est survenue: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

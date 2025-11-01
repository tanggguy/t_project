# --- 1. Bibliothèques natives ---
from typing import Dict, Any

# --- 2. Bibliothèques tierces ---
import pandas as pd
import numpy as np

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger

# Initialisation du logger pour ce module
logger = setup_logger(__name__)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes de rendements au DataFrame.

    - 'pct_return' : Rendement en pourcentage (ex: 0.01 pour 1%)
    - 'log_return' : Rendement logarithmique (utile pour les additions)

    Args:
        df (pd.DataFrame): DataFrame d'entrée (doit avoir 'close').

    Returns:
        pd.DataFrame: DataFrame avec les colonnes de rendements.
    """
    if "close" not in df.columns:
        logger.error("La colonne 'close' est manquante pour calculer les rendements.")
        return df

    df["pct_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Remplir le premier NaN (pas de rendement pour la première bougie)
    df.fillna(
        {"pct_return": 0.0, "log_return": 0.0},
        inplace=True,
    )

    logger.debug("Colonnes 'pct_return' et 'log_return' ajoutées.")
    return df


def handle_outliers(
    df: pd.DataFrame,
    method: str = "clip_returns",
    quantile: float = 0.01,
) -> pd.DataFrame:
    """
    Gère les valeurs aberrantes (outliers) dans le DataFrame.

    Méthodes supportées :
    - 'clip_returns': Écrête les rendements extrêmes (pct_return)
      basé sur les quantiles.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        method (str): Méthode de gestion (défaut: 'clip_returns').
        quantile (float): Le quantile à utiliser (ex: 0.01 pour 1% et 99%).

    Returns:
        pd.DataFrame: DataFrame traité.
    """
    if method == "clip_returns":
        if "pct_return" not in df.columns:
            logger.warning(
                "La méthode 'clip_returns' nécessite 'pct_return'. "
                "Appel de add_returns() implicitement."
            )
            df = add_returns(df)

        # Exclude the first row from outlier detection as its return is always 0 and not a market value.
        if df.shape[0] > 1:
            returns_to_check = df["pct_return"].iloc[1:]

            lower_bound = returns_to_check.quantile(quantile)
            upper_bound = returns_to_check.quantile(1 - quantile)

            clipped_mask = (returns_to_check < lower_bound) | (returns_to_check > upper_bound)
            initial_clipped_count = clipped_mask.sum()

            if initial_clipped_count > 0:
                # Note : Ceci modifie 'pct_return', mais pas 'close'.
                # C'est généralement ce qu'on veut pour l'analyse,
                # car le prix 'close' est (généralement) la vérité du marché.
                df.loc[df.index[1:], "pct_return"] = returns_to_check.clip(
                    lower=lower_bound, upper=upper_bound
                )
                logger.info(
                    f"{initial_clipped_count} rendements extrêmes ont été "
                    f"écrêtés aux quantiles {quantile*100}% et {(1-quantile)*100}%."
                )
            else:
                logger.debug("Aucun outlier de rendement détecté (basé sur les quantiles).")
        else:
            logger.debug("DataFrame trop petit pour la détection d'outliers de rendement.")

    else:
        logger.warning(f"Méthode d'outlier '{method}' non reconnue. Aucune action.")

    return df


def resample_data(df: pd.DataFrame, rule: str = "1W") -> pd.DataFrame:
    """
    Ré-échantillonne (resample) les données OHLCV à une fréquence
    temporelle supérieure (ex: 1 jour -> 1 semaine).

    Args:
        df (pd.DataFrame): DataFrame en entrée (index doit être DatetimeIndex).
        rule (str): La règle de resampling (ex: '1W' pour semaine,
                    '1M' pour mois).

    Returns:
        pd.DataFrame: Le DataFrame ré-échantillonné.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("L'index n'est pas un DatetimeIndex. Resampling impossible.")
        return df

    # Définition de la logique d'agrégation
    agg_logic: Dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Conserver uniquement les colonnes qui existent
    cols_to_agg = {col: agg_logic[col] for col in agg_logic if col in df.columns}

    try:
        resampled_df = df.resample(rule).apply(cols_to_agg)

        # Supprimer les lignes créées pour des périodes sans données
        resampled_df.dropna(subset=["close"], inplace=True)

        logger.info(
            f"DataFrame ré-échantillonné avec la règle '{rule}'. "
            f"Taille: {df.shape} -> {resampled_df.shape}"
        )
        return resampled_df

    except Exception as e:
        logger.error(f"Échec du resampling avec la règle '{rule}': {e}")
        return df


# --- Bloc de test ---
if __name__ == "__main__":
    """
    Ce bloc s'exécute uniquement lorsque vous lancez ce script directement
    (ex: `python utils/data_processor.py`) pour tester son fonctionnement.
    """
    logger.info("--- Début du test de DataProcessor ---")

    # 1. Créer un faux DataFrame (similaire à ce que DataManager produirait)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1D")
    data = {
        "open": np.random.uniform(95, 105, 100),
        "high": np.random.uniform(105, 110, 100),
        "low": np.random.uniform(90, 95, 100),
        "close": np.random.uniform(100, 108, 100),
        "volume": np.random.randint(1_000_000, 5_000_000, 100),
    }
    test_df = pd.DataFrame(data, index=dates)
    test_df.index.name = "Date"

    # Ajouter un outlier
    test_df.loc[test_df.index[10], "close"] = 500.0

    logger.info(
        f"DataFrame de test créé. Taille: {test_df.shape}. "
        f"Outlier 'close' à 500.0 ajouté."
    )

    # 2. Test add_returns
    df_with_returns = add_returns(test_df.copy())
    logger.info(f"Test 'add_returns'. Colonnes: {list(df_with_returns.columns)}")
    logger.info(
        f"Rendement outlier (non-traité): {df_with_returns['pct_return'].max():.4f}"
    )

    # 3. Test handle_outliers
    df_handled = handle_outliers(df_with_returns.copy(), quantile=0.01)
    logger.info(
        f"Test 'handle_outliers'. Rendement max après clipping: "
        f"{df_handled['pct_return'].max():.4f}"
    )

    # 4. Test resample_data
    df_weekly = resample_data(test_df.copy(), rule="1W")
    logger.info(
        f"Test 'resample_data' (1W). Taille originale: {test_df.shape}, "
        f"Taille resampled: {df_weekly.shape}"
    )
    logger.info("Premières 5 lignes du DataFrame '1W':")
    print(df_weekly.head())

    logger.info("--- Fin du test de DataProcessor ---")

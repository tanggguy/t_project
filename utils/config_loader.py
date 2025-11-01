# utils/config_loader.py

# --- 1. Bibliothèques natives ---
import os
import logging
from typing import Dict, Any, Optional

# --- 2. Bibliothèques tierces ---
import yaml

# --- 3. Imports locaux du projet ---
# AUCUN IMPORT de logger ici pour éviter la dépendance circulaire

# Logger simple pour ce module (sera configuré plus tard)
logger = logging.getLogger(__name__)

# --- Constantes de chemin ---
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

# Variable globale pour stocker la configuration (modèle Singleton)
_global_settings: Optional[Dict[str, Any]] = None


def _load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Fonction d'aide interne pour charger un fichier YAML spécifique.

    Args:
        file_path (str): Le chemin absolu vers le fichier .yaml.

    Returns:
        Dict[str, Any]: Le contenu du fichier YAML sous forme de dictionnaire.

    Raises:
        FileNotFoundError: Si le fichier de configuration n'est pas trouvé.
        yaml.YAMLError: Si le fichier YAML est mal formaté.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                logger.warning(f"Le fichier de configuration {file_path} est vide.")
                return {}
            return data
    except FileNotFoundError:
        logger.error(
            f"ERREUR CRITIQUE: Fichier de configuration non trouvé à : {file_path}"
        )
        raise
    except yaml.YAMLError as e:
        logger.error(
            f"ERREUR CRITIQUE: Erreur de syntaxe dans le fichier YAML {file_path}: {e}"
        )
        raise


def get_settings(force_reload: bool = False) -> Dict[str, Any]:
    """
    Charge les paramètres globaux du projet depuis 'config/settings.yaml'.

    Implémente un modèle singleton pour éviter de relire le fichier
    inutilement. La configuration est chargée une fois et mise en cache.

    Args:
        force_reload (bool): Si True, force une relecture du fichier
                             de configuration. Par défaut à False.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les paramètres globaux.
    """
    global _global_settings

    if _global_settings is not None and not force_reload:
        logger.debug("Retour des paramètres globaux depuis le cache.")
        return _global_settings

    logger.info(f"Chargement des paramètres globaux depuis : {DEFAULT_CONFIG_PATH}")
    _global_settings = _load_yaml_file(DEFAULT_CONFIG_PATH)

    return _global_settings


def get_log_level(context: str = "logging") -> int:
    """
    Récupère le niveau de logging depuis settings.yaml.

    Args:
        context (str): Contexte ('logging' pour normal, 'optimization' pour Optuna)

    Returns:
        int: Niveau de logging (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> from utils.config_loader import get_log_level
        >>> level = get_log_level()  # Normal
        >>> level = get_log_level('optimization')  # Optuna
    """
    settings = get_settings()
    level_str = settings.get(context, {}).get("log_level", "INFO")

    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return levels.get(level_str.upper(), logging.INFO)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Récupère une valeur de configuration avec notation pointée.

    Args:
        key_path (str): Chemin vers la clé (ex: 'logging.level')
        default (Any): Valeur par défaut si clé introuvable

    Returns:
        Any: Valeur de configuration

    Example:
        >>> from utils.config_loader import get_config_value
        >>> log_file = get_config_value('logging.file')
        >>> initial_cash = get_config_value('backtrader.initial_cash', 10000)
    """
    settings = get_settings()
    keys = key_path.split(".")
    value = settings

    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default

    return value


# --- Bloc de test ---
if __name__ == "__main__":  # pragma: no cover
    """
    Test du module config_loader.
    """
    # Setup logging basique pour les tests
    logging.basicConfig(level=logging.DEBUG)

    logger.info("--- Début du test de config_loader.py ---")

    # 1. Premier chargement
    settings = get_settings()

    # 2. Test d'accès
    if settings:
        logger.info(
            f"Capital initial (lu): {settings.get('backtest', {}).get('initial_capital')}"
        )
        logger.info(f"Timezone (lue): {settings.get('project', {}).get('timezone')}")

    # 3. Test du singleton
    logger.info("--- Test du Singleton (devrait utiliser le cache) ---")
    settings_cached = get_settings()

    # 4. Test du rechargement forcé
    logger.info("--- Test du rechargement forcé ---")
    settings_reloaded = get_settings(force_reload=True)

    # 5. Test des nouvelles fonctions helpers
    logger.info("--- Test des helpers ---")
    log_level = get_log_level()
    logger.info(f"Log level: {log_level}")

    log_file = get_config_value("logging.file", "logs/default.log")
    logger.info(f"Log file: {log_file}")

    logger.info("--- Fin du test de config_loader.py ---")

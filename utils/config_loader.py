# --- 1. Bibliothèques natives ---
import os
from typing import Dict, Any, Optional

# --- 2. Bibliothèques tierces ---
import yaml

# --- 3. Imports locaux du projet ---
from utils.logger import setup_logger

# Initialisation du logger pour ce module
logger = setup_logger(__name__)

# --- Constantes de chemin ---
# Détermine le chemin racine du projet (en supposant que ce fichier est dans t_project/utils/)
# __file__ -> /chemin/vers/t_project/utils/config_loader.py
# os.path.dirname(__file__) -> /chemin/vers/t_project/utils
# os.path.dirname(os.path.dirname(__file__)) -> /chemin/vers/t_project
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

    # Si la configuration est déjà chargée et qu'on ne force pas la recharge,
    # on retourne la version en cache.
    if _global_settings is not None and not force_reload:
        logger.debug("Retour des paramètres globaux depuis le cache.")
        return _global_settings

    # Sinon, on charge (ou recharge) le fichier
    logger.info(f"Chargement des paramètres globaux depuis : {DEFAULT_CONFIG_PATH}")
    _global_settings = _load_yaml_file(DEFAULT_CONFIG_PATH)

    return _global_settings


# --- Bloc de test ---
if __name__ == "__main__":
    """
    Ce bloc s'exécute uniquement lorsque vous lancez ce script directement
    (ex: `python utils/config_loader.py`) pour tester son fonctionnement.
    """
    logger.info("--- Début du test de config_loader.py ---")

    # 1. Premier chargement
    settings = get_settings()

    # 2. Test d'accès (conformément au settings.yaml créé précédemment)
    if settings:
        logger.info(
            f"Capital initial (lu): {settings.get('backtest', {}).get('initial_capital')}"
        )
        logger.info(f"Timezone (lue): {settings.get('project', {}).get('timezone')}")

    # 3. Test du singleton (devrait utiliser le cache)
    logger.info("--- Test du Singleton (devrait utiliser le cache) ---")
    settings_cached = get_settings()
    # Vous devriez voir un message 'DEBUG: Retour des paramètres globaux...'

    # 4. Test du rechargement forcé
    logger.info("--- Test du rechargement forcé ---")
    settings_reloaded = get_settings(force_reload=True)
    # Vous devriez voir un message 'INFO: Chargement des paramètres...'

    logger.info("--- Fin du test de config_loader.py ---")

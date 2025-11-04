# utils/logger.py

# --- 1. Bibliothèques natives ---
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict

# --- 2. Bibliothèques tierces ---
import coloredlogs

# --- 3. Imports locaux du projet ---
# Import tardif pour éviter dépendance circulaire


# Handlers de fichiers partagés par chemin absolu pour éviter
# d'ouvrir plusieurs fois le même fichier sur Windows (verrouillage).
_FILE_HANDLERS: Dict[str, RotatingFileHandler] = {}


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[int] = None,
    use_config: bool = True,
) -> logging.Logger:
    """
    Configure un logger avec rotation de fichiers et sortie console colorée.

    Args:
        name (str): Nom du logger (généralement __name__)
        log_file (Optional[str]): Chemin du fichier de log (si None, utilise config)
        level (Optional[int]): Niveau de log console (si None, utilise config)
        use_config (bool): Si True, utilise settings.yaml (défaut: True)

    Returns:
        logging.Logger: Logger configuré

    Example:
        >>> from utils.logger import setup_logger
        >>> logger = setup_logger(__name__)  # Utilise settings.yaml
    """
    # Import tardif pour éviter dépendance circulaire
    if use_config:
        try:
            from utils.config_loader import get_config_value, get_log_level

            if log_file is None:
                log_file = get_config_value("logging.file", "logs/trading_project.log")

            if level is None:
                level = get_log_level("logging")

            max_bytes = get_config_value("logging.max_bytes", 5242880)
            backup_count = get_config_value("logging.backup_count", 5)
        except Exception:
            # Fallback si config non disponible
            log_file = log_file or "logs/trading_project.log"
            level = level or logging.INFO
            max_bytes = 5242880
            backup_count = 5
    else:
        # Valeurs par défaut si use_config=False
        log_file = log_file or "logs/trading_project.log"
        level = level or logging.INFO
        max_bytes = 5242880
        backup_count = 5

    # Create logs directory if it doesn't exist
    # Utiliser un chemin absolu pour uniformiser la clé de partage
    log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Éviter d'ajouter des handlers plusieurs fois
    if logger.hasHandlers():
        return logger

    # File handler (sans couleurs) - TOUJOURS DEBUG
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Réutiliser un unique handler par fichier pour éviter
    # les conflits de rotation sur Windows (WinError 32)
    file_handler = _FILE_HANDLERS.get(log_file)
    if file_handler is None:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=int(max_bytes),
            backupCount=int(backup_count),
            encoding="utf-8",
            delay=True,  # n'ouvre le fichier qu'au premier emit
        )
        _FILE_HANDLERS[log_file] = file_handler
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Console handler (AVEC COULEURS)
    console_formatter = coloredlogs.ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level_styles=coloredlogs.DEFAULT_LEVEL_STYLES,
        field_styles=coloredlogs.DEFAULT_FIELD_STYLES,
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger


# Silencer les logs verbeux de bibliothèques tierces
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)

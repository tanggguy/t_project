# --- 1. Bibliothèques natives ---
import logging
import os
from logging.handlers import RotatingFileHandler

# --- 2. Bibliothèques tierces ---
import coloredlogs  # <--- NOUVEL IMPORT

# --- 3. Imports locaux du projet ---
# (Aucun dans ce fichier)


def setup_logger(name, log_file="logs/trading_project.log", level=logging.INFO):
    """Function to setup as many loggers as you want"""

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # --- Logger ---
    # On récupère le logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture tous les messages

    # Éviter les logs en double vers le logger root (souvent configuré par d'autres libs)
    logger.propagate = False

    # Éviter d'ajouter des handlers plusieurs fois si le script est ré-importé
    if logger.hasHandlers():
        return logger

    # --- File handler (sans couleurs) ---
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1024 * 1024 * 5, backupCount=5
    )  # 5 MB
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Écrit tout dans le fichier
    logger.addHandler(file_handler)

    # --- Console handler (AVEC COULEURS) ---
    # Format personnalisé pour les logs colorés
    console_formatter = coloredlogs.ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level_styles=coloredlogs.DEFAULT_LEVEL_STYLES,
        field_styles=coloredlogs.DEFAULT_FIELD_STYLES,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)  # Utilise le level (INFO, DEBUG)
    logger.addHandler(console_handler)

    return logger


# Example of how to use it:
# from utils.logger import setup_logger
# logger = setup_logger(__name__)
# logger.info("This is an info message")
# logger.debug("This is a debug message")

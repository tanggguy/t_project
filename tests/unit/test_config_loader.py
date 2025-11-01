# tests/unit/test_config_loader.py

import os
import pytest
import yaml
import logging
from typing import Dict, Any

from utils.config_loader import (
    get_settings,
    get_log_level,
    get_config_value,
    _load_yaml_file,
    DEFAULT_CONFIG_PATH,
)


# --- Fixtures ---
@pytest.fixture
def sample_config() -> Dict[str, Any]:
    return {
        "project": {"name": "trading_bot", "timezone": "Europe/Paris"},
        "backtest": {"initial_capital": 10000},
        "logging": {"level": "DEBUG", "file": "logs/test.log"},
        "optimization": {"log_level": "WARNING"},
    }


@pytest.fixture
def mock_yaml_file(tmp_path, sample_config):
    """Crée un fichier YAML temporaire avec la configuration de test"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config, f)
    return str(config_path)


# --- Tests pour _load_yaml_file ---
def test_load_yaml_file_success(mock_yaml_file, sample_config, caplog):
    """Test le chargement réussi d'un fichier YAML"""
    with caplog.at_level(logging.WARNING):
        result = _load_yaml_file(mock_yaml_file)
        assert result == sample_config
        assert len(caplog.records) == 0


def test_load_yaml_file_none_content(tmp_path, caplog):
    """Test le chargement d'un fichier YAML qui retourne None"""
    test_file = tmp_path / "none.yaml"
    test_file.write_text("~")  # YAML qui sera lu comme None

    with caplog.at_level(logging.WARNING):
        result = _load_yaml_file(str(test_file))
        assert result == {}
        assert "est vide" in caplog.text


def test_load_yaml_file_empty(tmp_path):
    """Test le chargement d'un fichier YAML vide"""
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    result = _load_yaml_file(str(empty_file))
    assert result == {}


def test_load_yaml_file_not_found():
    """Test le chargement d'un fichier YAML inexistant"""
    with pytest.raises(FileNotFoundError):
        _load_yaml_file("non_existent.yaml")


def test_load_yaml_file_invalid(tmp_path):
    """Test le chargement d'un fichier YAML mal formaté"""
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("invalid: yaml: content:")
    with pytest.raises(yaml.YAMLError):
        _load_yaml_file(str(invalid_file))


# --- Tests pour get_settings ---
@pytest.fixture(autouse=True)
def clear_global_settings():
    """Réinitialise la configuration globale avant chaque test"""
    import utils.config_loader

    utils.config_loader._global_settings = None
    yield
    utils.config_loader._global_settings = None


def test_get_settings_first_load(mocker, sample_config):
    """Test le premier chargement des paramètres"""
    mock_load = mocker.patch("utils.config_loader._load_yaml_file")
    mock_load.return_value = sample_config.copy()
    result = get_settings()
    assert result == sample_config
    mock_load.assert_called_once_with(DEFAULT_CONFIG_PATH)


def test_get_settings_cached(mocker, sample_config):
    """Test l'utilisation du cache pour les paramètres"""
    mock_load = mocker.patch("utils.config_loader._load_yaml_file")
    mock_load.return_value = sample_config.copy()

    # Premier appel pour initialiser le cache
    first_result = get_settings()
    # Deuxième appel qui devrait utiliser le cache
    second_result = get_settings()

    # Vérifie que _load_yaml_file n'a été appelé qu'une seule fois
    mock_load.assert_called_once_with(DEFAULT_CONFIG_PATH)
    # Vérifie que les deux résultats sont identiques
    assert first_result == second_result == sample_config


def test_get_settings_force_reload(mocker, sample_config):
    """Test le rechargement forcé des paramètres"""
    mock_load = mocker.patch("utils.config_loader._load_yaml_file")
    mock_load.return_value = sample_config.copy()

    # Premier appel normal
    first_result = get_settings()
    # Appel avec force_reload=True
    second_result = get_settings(force_reload=True)

    # Vérifie que _load_yaml_file a été appelé deux fois
    assert mock_load.call_count == 2
    mock_load.assert_has_calls(
        [mocker.call(DEFAULT_CONFIG_PATH), mocker.call(DEFAULT_CONFIG_PATH)]
    )
    # Vérifie que les deux résultats sont identiques
    assert first_result == second_result == sample_config


# --- Tests pour get_log_level ---
def test_get_log_level_all_levels(mocker):
    """Test tous les niveaux de log possibles"""
    test_cases = [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("debug", logging.DEBUG),  # Test de la casse
        ("warning", logging.WARNING),  # Test de la casse
    ]

    for level_str, expected_level in test_cases:
        mock_settings = {"logging": {"log_level": level_str}}
        mocker.patch("utils.config_loader.get_settings", return_value=mock_settings)
        assert get_log_level() == expected_level


def test_get_log_level_default(mocker):
    """Test la récupération du niveau de log par défaut"""
    mock_settings = {"logging": {"log_level": "INFO"}}
    mocker.patch("utils.config_loader.get_settings", return_value=mock_settings)

    level = get_log_level()
    assert level == logging.INFO


def test_get_log_level_custom_context(mocker):
    """Test la récupération du niveau de log pour un contexte spécifique"""
    mock_settings = {"optimization": {"log_level": "WARNING"}}
    mocker.patch("utils.config_loader.get_settings", return_value=mock_settings)

    level = get_log_level("optimization")
    assert level == logging.WARNING


def test_get_log_level_invalid(mocker):
    """Test la récupération d'un niveau de log invalide"""
    mock_settings = {"logging": {"log_level": "INVALID_LEVEL"}}
    mocker.patch("utils.config_loader.get_settings", return_value=mock_settings)

    level = get_log_level()
    assert level == logging.INFO  # Valeur par défaut


def test_get_log_level_missing_context(mocker):
    """Test la récupération du niveau de log pour un contexte manquant"""
    mock_settings = {}
    mocker.patch("utils.config_loader.get_settings", return_value=mock_settings)

    level = get_log_level("missing_context")
    assert level == logging.INFO


# --- Tests pour get_config_value ---
def test_get_config_value_existing(mocker, sample_config):
    """Test la récupération d'une valeur existante"""
    mocker.patch("utils.config_loader.get_settings", return_value=sample_config)

    value = get_config_value("logging.file")
    assert value == "logs/test.log"


def test_get_config_value_missing(mocker):
    """Test la récupération d'une valeur manquante"""
    mocker.patch("utils.config_loader.get_settings", return_value={})

    value = get_config_value("missing.key", default="default_value")
    assert value == "default_value"


def test_get_config_value_nested_deep(mocker):
    """Test la récupération d'une valeur profondément imbriquée"""
    deep_config = {"level1": {"level2": {"level3": {"value": "deep_value"}}}}
    mocker.patch("utils.config_loader.get_settings", return_value=deep_config)

    # Test chemin profond
    assert get_config_value("level1.level2.level3.value") == "deep_value"

    # Test chemin intermédiaire
    assert get_config_value("level1.level2.level3") == {"value": "deep_value"}


def test_get_config_value_none_intermediate(mocker):
    """Test la récupération quand une valeur intermédiaire est None"""
    config = {"parent": {"child": None}}
    mocker.patch("utils.config_loader.get_settings", return_value=config)

    assert get_config_value("parent.child.subkey", "default") == "default"


def test_get_config_value_nested(mocker, sample_config):
    """Test la récupération d'une valeur imbriquée"""
    mocker.patch("utils.config_loader.get_settings", return_value=sample_config)

    value = get_config_value("project.timezone")
    assert value == "Europe/Paris"


def test_get_config_value_invalid_path(mocker, sample_config):
    """Test la récupération avec un chemin invalide"""
    mocker.patch("utils.config_loader.get_settings", return_value=sample_config)

    value = get_config_value("invalid.path.to.value", default="default")
    assert value == "default"


def test_get_config_value_non_dict(mocker):
    """Test la récupération quand la valeur intermédiaire n'est pas un dictionnaire"""
    mock_settings = {"key": "value"}  # 'key' pointe vers une chaîne, pas un dict
    mocker.patch("utils.config_loader.get_settings", return_value=mock_settings)

    value = get_config_value("key.subkey", default="default")
    assert value == "default"

# Fixtures globales
# tests/conftest.py
"""
Configuration globale pour tous les tests pytest.
Couvre Phases 1-13 (Backtest → Live Trading).
"""

# --- 1. Bibliothèques natives ---
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock

# --- 2. Bibliothèques tierces ---
import pytest
import pandas as pd
import yaml

# --- 3. Imports locaux du projet ---
# (À adapter selon les modules créés)


# ========================================
# FIXTURES GÉNÉRALES
# ========================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Retourne le chemin vers tests/fixtures/"""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Configuration minimale pour tests."""
    return {
        "data": {"default_period": "1mo", "cache_enabled": False, "interval": "1d"},
        "backtest": {"initial_cash": 10000, "commission": 0.001, "slippage": 0.0005},
        "risk": {"max_position_size": 0.2, "stop_loss_pct": 0.05},
    }


# ========================================
# FIXTURES DATA
# ========================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """DataFrame OHLCV minimal (5 jours)."""
    data = {
        "Open": [100, 102, 101, 103, 105],
        "High": [102, 103, 104, 105, 106],
        "Low": [99, 101, 100, 102, 104],
        "Close": [101, 102, 103, 104, 105],
        "Volume": [1000000, 1100000, 1050000, 1200000, 1150000],
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=5, freq="D")
    df.index.name = "Date"
    return df


@pytest.fixture
def long_ohlcv_data() -> pd.DataFrame:
    """DataFrame OHLCV longue période (252 jours = 1 an de trading)."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")  # Business days
    data = {
        "Open": [100 + i * 0.1 for i in range(252)],
        "High": [101 + i * 0.1 for i in range(252)],
        "Low": [99 + i * 0.1 for i in range(252)],
        "Close": [100.5 + i * 0.1 for i in range(252)],
        "Volume": [1000000 + i * 1000 for i in range(252)],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def corrupted_data() -> pd.DataFrame:
    """DataFrame avec NaN, valeurs négatives, trous."""
    data = {
        "Open": [100, None, 101, 103, 105],
        "High": [102, 103, None, 105, 106],
        "Low": [99, 101, 100, 102, 104],
        "Close": [101, 102, 103, None, 105],
        "Volume": [1000000, 0, 1050000, 1200000, -100],
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=5, freq="D")
    df.index.name = "Date"
    return df


# ========================================
# 🆕 FIXTURES LIVE TRADING (Phase 12-13)
# ========================================


@pytest.fixture
def mock_broker() -> MagicMock:
    """Mock d'un broker pour paper trading."""
    broker = MagicMock()
    broker.get_cash.return_value = 10000.0
    broker.get_positions.return_value = []
    broker.submit_order.return_value = {"order_id": "12345", "status": "filled"}
    broker.cancel_order.return_value = {"status": "cancelled"}
    return broker


@pytest.fixture
def mock_websocket() -> Generator[MagicMock, None, None]:
    """Mock d'un flux WebSocket temps réel."""
    ws = MagicMock()
    ws.is_connected = True
    ws.subscribe.return_value = True

    # Simuler des données de marché
    ws.get_next_tick.return_value = {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 1000,
        "timestamp": "2024-10-31T10:30:00Z",
    }

    yield ws
    ws.close()


@pytest.fixture
def mock_alert_system() -> MagicMock:
    """Mock du système d'alertes (email/telegram)."""
    alerts = MagicMock()
    alerts.send_email.return_value = True
    alerts.send_telegram.return_value = True
    return alerts


@pytest.fixture
def live_market_snapshot(fixtures_dir: Path) -> pd.DataFrame:
    """Snapshot de données temps réel (mockées)."""
    file_path = fixtures_dir / "market_data" / "historical_snapshot.csv"

    # Si le fichier n'existe pas, créer des données factices
    if not file_path.exists():
        data = {
            "Symbol": ["AAPL", "MSFT", "GOOGL"],
            "Price": [150.25, 300.50, 2800.75],
            "Volume": [50000000, 30000000, 20000000],
            "Change": [1.5, -0.8, 2.3],
        }
        return pd.DataFrame(data)

    return pd.read_csv(file_path)


# ========================================
# 🆕 FIXTURES OPTIMIZATION (Phase 6+9)
# ========================================


@pytest.fixture
def sample_parameter_space() -> Dict[str, tuple]:
    """Espace de paramètres pour optimisation."""
    return {
        "fast_period": (5, 20, 1),  # (min, max, step)
        "slow_period": (25, 50, 5),
        "rsi_threshold": (20, 40, 5),
        "stop_loss_pct": (0.01, 0.10, 0.01),
    }


# ========================================
# 🆕 FIXTURES PERFORMANCE TESTS
# ========================================


@pytest.fixture
def performance_test_config() -> Dict[str, Any]:
    """Configuration pour tests de performance."""
    return {
        "max_execution_time_seconds": 30,
        "max_memory_mb": 500,
        "parallel_workers": 4,
    }

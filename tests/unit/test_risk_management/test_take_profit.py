"""
Tests unitaires pour les modules de Take Profit.

Ce fichier teste les classes :
- FixedTakeProfit
- ATRTakeProfit
- SupportResistanceTakeProfit
"""

# --- 1. Bibliothèques natives ---
from typing import List

# --- 2. Bibliothèques tierces ---
import pytest
import backtrader as bt

# --- 3. Imports locaux du projet ---
# (Assurez-vous que le PYTHONPATH est configuré pour que 'src' ou le parent soit visible)
from risk_management.take_profit import (
    FixedTakeProfit,
    ATRTakeProfit,
    SupportResistanceTakeProfit,
)

# --- Fixtures et Mocks ---


class MockData:
    """Mock pour simuler self.data.low/high dans Backtrader."""

    def __init__(self, high_data: List[float], low_data: List[float]):
        self._high_data = high_data
        self._low_data = low_data
        self.buflen = len(high_data)

    def __len__(self) -> int:
        return self.buflen

    @property
    def high(self):
        return self.MockLine(self._high_data)

    @property
    def low(self):
        return self.MockLine(self._low_data)

    class MockLine:
        def __init__(self, data: List[float]):
            self._data = data

        def __getitem__(self, index: int) -> float:
            # Simule l'accès inversé de backtrader (-1 est la bougie précédente)
            if index < 0:
                pos = len(self._data) + index
                if 0 <= pos < len(self._data):
                    return self._data[pos]
            raise IndexError("MockData index out of range for relative access")


# --- CORRECTION ---
# Retrait de l'héritage de bt.Strategy pour éviter l'erreur
# d'initialisation de Cerebro (AttributeError: 'NoneType')
class MockStrategy:
    """
    Mock (simulation) d'une stratégie Backtrader.
    Ne doit PAS hériter de bt.Strategy pour éviter l'initialisation
    de Cerebro qui n'est pas nécessaire pour ce test (Duck Typing).
    """

    def __init__(self, high_data: List[float], low_data: List[float]):
        # super().__init__() # <--- Supprimé
        self.data = MockData(high_data, low_data)


# --- FIN DE LA CORRECTION ---


@pytest.fixture
def mock_strategy() -> MockStrategy:
    """Fixture fournissant une stratégie mock avec des données de test."""
    # Données:          -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1 (indices backtrader)
    high_data = [100.0, 102.0, 105.0, 104.0, 106.0, 103.0, 101.0, 108.0]  # -1 = 108.0
    low_data = [99.0, 98.0, 95.0, 96.0, 94.0, 97.0, 99.0, 101.0]  # -1 = 101.0

    # Pivots Hauts attendus (High[-i] > High[-i-1] et High[-i] > High[-i+1]):
    # 105.0 (à -6) (car > 102 et > 104)
    # 106.0 (à -4) (car > 104 et > 103)

    # Pivots Bas attendus (Low[-i] < Low[-i-1] et Low[-i] < Low[-i+1]):
    # 95.0 (à -6) (car < 98 et < 96)
    # 94.0 (à -4) (car < 96 et < 97)

    # Note: J'ai modifié high_data[-1] à 108.0 pour que 101.0 (à -2) ne soit pas
    # un pivot haut, afin de rendre les tests plus clairs.

    return MockStrategy(high_data, low_data)


# --- Tests pour FixedTakeProfit ---


class TestFixedTakeProfit:

    def test_init_valid(self):
        """Teste l'initialisation valide."""
        tp = FixedTakeProfit(tp_pct=0.05)
        assert tp.tp_pct == 0.05

    @pytest.mark.parametrize("invalid_pct", [0, -0.01, -1.0])
    def test_init_invalid(self, invalid_pct: float):
        """Teste que l'initialisation échoue avec un pct négatif ou nul."""
        with pytest.raises(ValueError, match="tp_pct doit être positif"):
            FixedTakeProfit(tp_pct=invalid_pct)

    @pytest.mark.parametrize(
        "entry, pct, p_type, expected",
        [
            (100.0, 0.05, "long", 105.0),  # 100 * (1 + 0.05)
            (100.0, 0.05, "short", 95.0),  # 100 * (1 - 0.05)
            (50.0, 0.1, "long", 55.0),  # 50 * (1 + 0.1)
            (50.0, 0.1, "short", 45.0),  # 50 * (1 - 0.1)
        ],
    )
    def test_calculate_target(
        self, entry: float, pct: float, p_type: str, expected: float
    ):
        """Teste le calcul du niveau de TP."""
        tp = FixedTakeProfit(tp_pct=pct)
        target = tp.calculate_target(entry_price=entry, position_type=p_type)
        assert target == pytest.approx(expected)

    @pytest.mark.parametrize(
        "current, target, p_type, expected_trigger",
        [
            (106.0, 105.0, "long", True),  # Prix > Cible (Long) -> Déclenché
            (105.0, 105.0, "long", True),  # Prix == Cible (Long) -> Déclenché
            (104.0, 105.0, "long", False),  # Prix < Cible (Long) -> Non déclenché
            (94.0, 95.0, "short", True),  # Prix < Cible (Short) -> Déclenché
            (95.0, 95.0, "short", True),  # Prix == Cible (Short) -> Déclenché
            (96.0, 95.0, "short", False),  # Prix > Cible (Short) -> Non déclenché
        ],
    )
    def test_should_trigger(
        self, current: float, target: float, p_type: str, expected_trigger: bool
    ):
        """Teste la logique de déclenchement."""
        tp = FixedTakeProfit(tp_pct=0.01)  # pct non pertinent ici
        triggered = tp.should_trigger(current, target, p_type)
        assert triggered == expected_trigger


# --- Tests pour ATRTakeProfit ---


class TestATRTakeProfit:

    def test_init_valid(self):
        """Teste l'initialisation valide."""
        tp = ATRTakeProfit(atr_multiplier=3.0, atr_period=14)
        assert tp.atr_multiplier == 3.0
        assert tp.atr_period == 14

    @pytest.mark.parametrize("mult, period", [(0, 14), (-1, 14), (2, 0), (2, -1)])
    def test_init_invalid(self, mult: float, period: int):
        """Teste les multiplicateurs ou périodes invalides."""
        with pytest.raises(ValueError):
            ATRTakeProfit(atr_multiplier=mult, atr_period=period)

    @pytest.mark.parametrize(
        "entry, atr, mult, p_type, expected",
        [
            (100.0, 2.0, 3.0, "long", 106.0),  # 100 + (2.0 * 3.0)
            (100.0, 2.0, 3.0, "short", 94.0),  # 100 - (2.0 * 3.0)
            (50.0, 1.5, 2.0, "long", 53.0),  # 50 + (1.5 * 2.0)
            (50.0, 1.5, 2.0, "short", 47.0),  # 50 - (1.5 * 2.0)
        ],
    )
    def test_calculate_target(
        self, entry: float, atr: float, mult: float, p_type: str, expected: float
    ):
        """Teste le calcul du TP basé sur l'ATR."""
        tp = ATRTakeProfit(atr_multiplier=mult, atr_period=14)
        target = tp.calculate_target(
            entry_price=entry, atr_value=atr, position_type=p_type
        )
        assert target == pytest.approx(expected)

    def test_should_trigger(self):
        """La logique de déclenchement est identique à FixedTakeProfit."""
        tp = ATRTakeProfit(atr_multiplier=1.0)
        # Long (Prix >= Cible)
        assert tp.should_trigger(110, 109, "long") is True
        assert tp.should_trigger(109, 109, "long") is True
        assert tp.should_trigger(108, 109, "long") is False
        # Short (Prix <= Cible)
        assert tp.should_trigger(89, 90, "short") is True
        assert tp.should_trigger(90, 90, "short") is True
        assert tp.should_trigger(91, 90, "short") is False


# --- Tests pour SupportResistanceTakeProfit ---


class TestSupportResistanceTakeProfit:

    def test_init_valid(self):
        """Teste l'initialisation valide."""
        tp = SupportResistanceTakeProfit(lookback_period=20, buffer_pct=0.01)
        assert tp.lookback_period == 20
        assert tp.buffer_pct == 0.01

    @pytest.mark.parametrize("period, pct", [(0, 0.01), (-1, 0.01), (20, -0.01)])
    def test_init_invalid(self, period: int, pct: float):
        """Teste les paramètres d'initialisation invalides."""
        with pytest.raises(ValueError):
            SupportResistanceTakeProfit(lookback_period=period, buffer_pct=pct)

    def test_find_resistance(self, mock_strategy: MockStrategy):
        """Teste la détection des pivots hauts (résistances)."""
        tp = SupportResistanceTakeProfit(lookback_period=10, buffer_pct=0.0)
        # Niveaux attendus (triés du plus bas au plus haut)
        # Données High: [..., 105.0(-6), 104.0(-5), 106.0(-4), 103.0(-3), ...]
        # Pivot à 105.0 (-6) (car > 102 et > 104)
        # Pivot à 106.0 (-4) (car > 104 et > 103)
        resistances = tp.find_resistance(mock_strategy, num_levels=2)
        assert resistances == [105.0, 106.0]
        resistances_1 = tp.find_resistance(mock_strategy, num_levels=1)
        assert resistances_1 == [105.0]

    def test_find_support(self, mock_strategy: MockStrategy):
        """Teste la détection des pivots bas (supports)."""
        tp = SupportResistanceTakeProfit(lookback_period=10, buffer_pct=0.0)
        # Niveaux attendus (triés du plus haut au plus bas)
        # Données Low: [..., 95.0(-6), 96.0(-5), 94.0(-4), 97.0(-3), ...]
        # Pivot à 95.0 (-6) (car < 98 et < 96)
        # Pivot à 94.0 (-4) (car < 96 et < 97)
        supports = tp.find_support(mock_strategy, num_levels=2)
        assert supports == [95.0, 94.0]
        supports_1 = tp.find_support(mock_strategy, num_levels=1)
        assert supports_1 == [95.0]

    def test_calculate_target(self):
        """Teste le calcul du TP avec buffer."""
        tp = SupportResistanceTakeProfit(buffer_pct=0.01)  # 1% buffer

        # Long: TP juste sous la résistance (1 - buffer)
        target_long = tp.calculate_target(
            entry_price=100, resistance_level=110.0, position_type="long"
        )
        assert target_long == pytest.approx(110.0 * (1 - 0.01))  # 108.9

        # Short: TP juste au-dessus du support (1 + buffer)
        target_short = tp.calculate_target(
            entry_price=100, support_level=90.0, position_type="short"
        )
        assert target_short == pytest.approx(90.0 * (1 + 0.01))  # 90.9

    def test_calculate_target_missing_levels(self):
        """Teste que les calculs échouent si les niveaux sont absents."""
        tp = SupportResistanceTakeProfit()

        # Long a besoin de 'resistance_level'
        with pytest.raises(ValueError, match="resistance_level requis"):
            tp.calculate_target(entry_price=100, position_type="long")

        # Short a besoin de 'support_level'
        with pytest.raises(ValueError, match="support_level requis"):
            tp.calculate_target(entry_price=100, position_type="short")

    def test_should_trigger(self):
        """La logique de déclenchement est identique aux autres classes."""
        tp = SupportResistanceTakeProfit()
        # Long (Prix >= Cible)
        assert tp.should_trigger(110, 109, "long") is True
        # Short (Prix <= Cible)
        assert tp.should_trigger(89, 90, "short") is True

"""
Tests unitaires pour le module risk_management.stop_loss.

Execute avec : python -m pytest test_stop_loss.py -v
"""

import pytest
from risk_management import (
    FixedStopLoss,
    TrailingStopLoss,
    ATRStopLoss,
    SupportResistanceStop,
)


class TestFixedStopLoss:
    """Tests pour la classe FixedStopLoss."""

    def test_init_valid(self):
        """Test d'initialisation valide."""
        stop = FixedStopLoss(stop_pct=0.02)
        assert stop.stop_pct == 0.02

    def test_init_invalid_negative(self):
        """Test avec pourcentage négatif."""
        with pytest.raises(ValueError):
            FixedStopLoss(stop_pct=-0.02)

    def test_init_invalid_zero(self):
        """Test avec pourcentage nul."""
        with pytest.raises(ValueError):
            FixedStopLoss(stop_pct=0)

    def test_calculate_stop_long(self):
        """Test calcul stop pour position long."""
        stop = FixedStopLoss(stop_pct=0.02)
        stop_level = stop.calculate_stop(entry_price=100, position_type="long")
        assert stop_level == 98.0

    def test_calculate_stop_short(self):
        """Test calcul stop pour position short."""
        stop = FixedStopLoss(stop_pct=0.02)
        stop_level = stop.calculate_stop(entry_price=100, position_type="short")
        assert stop_level == 102.0

    def test_calculate_stop_invalid_type(self):
        """Test avec type de position invalide."""
        stop = FixedStopLoss(stop_pct=0.02)
        with pytest.raises(ValueError):
            stop.calculate_stop(entry_price=100, position_type="invalid")

    def test_should_trigger_long_yes(self):
        """Test déclenchement stop pour long (doit déclencher)."""
        stop = FixedStopLoss(stop_pct=0.02)
        assert (
            stop.should_trigger(
                current_price=97.5, stop_level=98.0, position_type="long"
            )
            is True
        )

    def test_should_trigger_long_no(self):
        """Test déclenchement stop pour long (ne doit pas déclencher)."""
        stop = FixedStopLoss(stop_pct=0.02)
        assert (
            stop.should_trigger(
                current_price=99.0, stop_level=98.0, position_type="long"
            )
            is False
        )

    def test_should_trigger_short_yes(self):
        """Test déclenchement stop pour short (doit déclencher)."""
        stop = FixedStopLoss(stop_pct=0.02)
        assert (
            stop.should_trigger(
                current_price=102.5, stop_level=102.0, position_type="short"
            )
            is True
        )

    def test_should_trigger_short_no(self):
        """Test déclenchement stop pour short (ne doit pas déclencher)."""
        stop = FixedStopLoss(stop_pct=0.02)
        assert (
            stop.should_trigger(
                current_price=101.0, stop_level=102.0, position_type="short"
            )
            is False
        )


class TestTrailingStopLoss:
    """Tests pour la classe TrailingStopLoss."""

    def test_init_valid(self):
        """Test d'initialisation valide."""
        stop = TrailingStopLoss(trail_pct=0.03)
        assert stop.trail_pct == 0.03
        assert stop.highest_price is None
        assert stop.lowest_price is None

    def test_reset(self):
        """Test de la réinitialisation."""
        stop = TrailingStopLoss(trail_pct=0.03)
        stop.highest_price = 110.0
        stop.lowest_price = 90.0
        stop.reset()
        assert stop.highest_price is None
        assert stop.lowest_price is None

    def test_calculate_stop_long_first_call(self):
        """Test premier calcul pour position long."""
        stop = TrailingStopLoss(trail_pct=0.03)
        stop_level = stop.calculate_stop(
            entry_price=100, current_price=105, position_type="long"
        )
        assert stop.highest_price == 105
        assert stop_level == pytest.approx(105 * 0.97, rel=1e-6)

    def test_calculate_stop_long_update_higher(self):
        """Test mise à jour avec prix plus haut."""
        stop = TrailingStopLoss(trail_pct=0.03)
        stop.calculate_stop(entry_price=100, current_price=105, position_type="long")
        stop_level = stop.calculate_stop(
            entry_price=100, current_price=110, position_type="long"
        )
        assert stop.highest_price == 110
        assert stop_level == pytest.approx(110 * 0.97, rel=1e-6)

    def test_calculate_stop_long_no_update_lower(self):
        """Test que le stop ne recule pas."""
        stop = TrailingStopLoss(trail_pct=0.03)
        stop.calculate_stop(entry_price=100, current_price=110, position_type="long")
        stop_level = stop.calculate_stop(
            entry_price=100, current_price=105, position_type="long"
        )
        assert stop.highest_price == 110  # Ne recule pas
        assert stop_level == pytest.approx(110 * 0.97, rel=1e-6)

    def test_calculate_stop_short(self):
        """Test calcul pour position short."""
        stop = TrailingStopLoss(trail_pct=0.03)
        stop_level = stop.calculate_stop(
            entry_price=100, current_price=95, position_type="short"
        )
        assert stop.lowest_price == 95
        assert stop_level == pytest.approx(95 * 1.03, rel=1e-6)


class TestATRStopLoss:
    """Tests pour la classe ATRStopLoss."""

    def test_init_valid(self):
        """Test d'initialisation valide."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_period=14)
        assert stop.atr_multiplier == 2.0
        assert stop.atr_period == 14

    def test_init_invalid_multiplier(self):
        """Test avec multiplicateur invalide."""
        with pytest.raises(ValueError):
            ATRStopLoss(atr_multiplier=-1.0)

    def test_init_invalid_period(self):
        """Test avec période invalide."""
        with pytest.raises(ValueError):
            ATRStopLoss(atr_period=-5)

    def test_calculate_stop_long(self):
        """Test calcul stop pour position long."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_period=14)
        stop_level = stop.calculate_stop(
            entry_price=100, atr_value=2.5, position_type="long"
        )
        # Stop = 100 - (2.5 * 2.0) = 95.0
        assert stop_level == 95.0

    def test_calculate_stop_short(self):
        """Test calcul stop pour position short."""
        stop = ATRStopLoss(atr_multiplier=2.0, atr_period=14)
        stop_level = stop.calculate_stop(
            entry_price=100, atr_value=2.5, position_type="short"
        )
        # Stop = 100 + (2.5 * 2.0) = 105.0
        assert stop_level == 105.0

    def test_calculate_stop_different_atr(self):
        """Test avec différentes valeurs d'ATR."""
        stop = ATRStopLoss(atr_multiplier=1.5, atr_period=14)

        # ATR faible = stop serré
        stop_level_1 = stop.calculate_stop(
            entry_price=100, atr_value=1.0, position_type="long"
        )
        assert stop_level_1 == 98.5

        # ATR élevé = stop large
        stop_level_2 = stop.calculate_stop(
            entry_price=100, atr_value=5.0, position_type="long"
        )
        assert stop_level_2 == 92.5


class TestSupportResistanceStop:
    """Tests pour la classe SupportResistanceStop."""

    def test_init_valid(self):
        """Test d'initialisation valide."""
        stop = SupportResistanceStop(lookback_period=20, buffer_pct=0.005)
        assert stop.lookback_period == 20
        assert stop.buffer_pct == 0.005

    def test_init_invalid_lookback(self):
        """Test avec lookback invalide."""
        with pytest.raises(ValueError):
            SupportResistanceStop(lookback_period=-10)

    def test_init_invalid_buffer(self):
        """Test avec buffer négatif."""
        with pytest.raises(ValueError):
            SupportResistanceStop(buffer_pct=-0.01)

    def test_calculate_stop_long(self):
        """Test calcul stop pour position long."""
        stop = SupportResistanceStop(lookback_period=20, buffer_pct=0.005)
        stop_level = stop.calculate_stop(
            entry_price=100, support_level=95, position_type="long"
        )
        # Stop = 95 * (1 - 0.005) = 94.525
        assert stop_level == pytest.approx(94.525, rel=1e-6)

    def test_calculate_stop_short(self):
        """Test calcul stop pour position short."""
        stop = SupportResistanceStop(lookback_period=20, buffer_pct=0.005)
        stop_level = stop.calculate_stop(
            entry_price=100, resistance_level=105, position_type="short"
        )
        # Stop = 105 * (1 + 0.005) = 105.525
        assert stop_level == pytest.approx(105.525, rel=1e-6)

    def test_calculate_stop_long_missing_support(self):
        """Test erreur si support manquant pour long."""
        stop = SupportResistanceStop(lookback_period=20, buffer_pct=0.005)
        with pytest.raises(ValueError, match="support_level requis"):
            stop.calculate_stop(entry_price=100, position_type="long")

    def test_calculate_stop_short_missing_resistance(self):
        """Test erreur si résistance manquante pour short."""
        stop = SupportResistanceStop(lookback_period=20, buffer_pct=0.005)
        with pytest.raises(ValueError, match="resistance_level requis"):
            stop.calculate_stop(entry_price=100, position_type="short")


# Tests d'intégration
class TestIntegration:
    """Tests d'intégration pour vérifier la cohérence entre les classes."""

    def test_all_stops_same_interface(self):
        """Vérifie que toutes les classes ont la même interface."""
        stops = [
            FixedStopLoss(stop_pct=0.02),
            TrailingStopLoss(trail_pct=0.03),
            ATRStopLoss(atr_multiplier=2.0),
            SupportResistanceStop(lookback_period=20),
        ]

        for stop in stops:
            # Toutes doivent avoir should_trigger
            assert hasattr(stop, "should_trigger")
            assert callable(stop.should_trigger)

            # Toutes doivent avoir calculate_stop
            assert hasattr(stop, "calculate_stop")
            assert callable(stop.calculate_stop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

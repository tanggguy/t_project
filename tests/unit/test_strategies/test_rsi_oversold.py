# test_rsi_oversold.py
import logging
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Importer la classe à tester
# (Supposant qu'elle se trouve dans strategies/implementations/rsi_oversold.py)
from strategies.implementations.rsi_oversold import RsiOversoldStrategy


@pytest.fixture
def strategy_fixture(mocker):
    """
    Fournit une instance "vide" de RsiOversoldStrategy pour tester `next()`.

    Nous utilisons `__new__` pour créer l'instance sans appeler `__init__` ou
    la métaclasse de `backtrader` (ce qui cause l'erreur 'cerebro').
    Cela nous permet de tester la logique de `next` en isolation totale.
    """

    # 1. Créer une instance "vide" en contournant l'instanciation de backtrader
    strategy = RsiOversoldStrategy.__new__(RsiOversoldStrategy)

    # 2. Attacher manuellement tous les mocks nécessaires pour `next`
    #    CORRECTION : L'accès [0] doit être mocké via __getitem__
    strategy.rsi = MagicMock(name="rsi_indicator")
    strategy.data_close = MagicMock(name="data_close")

    # 3. Mocker les attributs et méthodes de BaseStrategy
    strategy.log = mocker.MagicMock(name="log_call")
    strategy.order = None  # Pas d'ordre en cours par défaut
    buy_order_mock = MagicMock(name="buy_order")
    sell_order_mock = MagicMock(name="sell_order")
    strategy.buy = mocker.MagicMock(name="buy_call", return_value=buy_order_mock)
    strategy.sell = mocker.MagicMock(name="sell_call", return_value=sell_order_mock)

    # 4. Mocker `len(self)`
    #    La méthode __len__ de backtrader appelle `len(self.lines)`.
    #    Nous mockons donc `self.lines` et sa propre méthode `__len__`.
    strategy.lines = MagicMock(name="lines_object")
    strategy.lines.__len__ = mocker.MagicMock(return_value=100)

    # 5. Mocker les paramètres 'p' que Backtrader génère
    #    Utilise les valeurs par défaut du fichier source.
    strategy.p = MagicMock()
    strategy.p.rsi_period = 14
    strategy.p.oversold_level = 30.0
    strategy.p.overbought_level = 70.0

    # 6. Mocker la propriété 'position' (qui est read-only)
    #    Nous patchons la CLASSE avec un PropertyMock.
    mock_position = MagicMock(name="position_object")
    mock_position.__bool__.return_value = False  # Défaut: pas de position

    mocker.patch.object(
        RsiOversoldStrategy,
        "position",
        new_callable=PropertyMock,
        return_value=mock_position,
    )
    # Exposer le mock de position pour le 'Arrange' des tests
    strategy.mock_position = mock_position

    return strategy


def test_init(mocker):
    """
    Teste la logique interne de la méthode `__init__`.
    """
    # 1. Mocker les dépendances externes appelées par __init__
    mock_super_init = mocker.patch("strategies.base_strategy.BaseStrategy.__init__")
    mock_rsi_class = mocker.patch("backtrader.indicators.RSI")
    mock_rsi_instance = MagicMock(name="rsi_instance")
    mock_rsi_class.return_value = mock_rsi_instance

    # 2. Créer une instance "vide" (pour éviter l'erreur de métaclasse)
    strategy = RsiOversoldStrategy.__new__(RsiOversoldStrategy)

    # 3. Attacher les attributs dont `__init__` a besoin pour s'exécuter
    strategy.log = mocker.MagicMock(name="log_call_init")
    strategy.data_close = MagicMock(name="data_close_init")

    # 4. Simuler l'objet 'p' que backtrader crée à partir de `params`
    #    CORRECTION pour `TypeError: 'type' object is not iterable`
    #    La métaclasse de backtrader transforme `params` en un objet de type
    #    auquel on peut accéder par attribut.
    strategy.p = MagicMock()
    strategy.p.rsi_period = RsiOversoldStrategy.params.rsi_period
    strategy.p.oversold_level = RsiOversoldStrategy.params.oversold_level
    strategy.p.overbought_level = RsiOversoldStrategy.params.overbought_level

    # 5. Appeler manuellement __init__ pour tester sa logique
    strategy.__init__()

    # 6. Assertions
    mock_super_init.assert_called_once()
    mock_rsi_class.assert_called_once_with(
        strategy.data_close,
        period=strategy.p.rsi_period,
    )
    assert strategy.rsi == mock_rsi_instance

    strategy.log.assert_called_with(mocker.ANY, level=logging.INFO)
    log_message = strategy.log.call_args[0][0]
    assert "Initialisation RsiOversoldStrategy" in log_message
    assert f"Période RSI: {strategy.p.rsi_period}" in log_message
    assert f"Seuil Oversold: {strategy.p.oversold_level}" in log_message
    assert f"Seuil Overbought: {strategy.p.overbought_level}" in log_message


class TestRsiOversoldStrategyNext:
    """
    Teste tous les scénarios logiques de la méthode `next()`.
    Utilise `strategy_fixture` pour chaque test.
    """

    def test_next_order_pending(self, strategy_fixture):
        """
        Scénario: Un ordre est déjà en cours.
        Attendu: La méthode `next` doit retourner immédiatement (guard clause).
        """
        # Arrange
        strategy_fixture.order = MagicMock(name="pending_order")

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_not_called()
        strategy_fixture.sell.assert_not_called()
        strategy_fixture.log.assert_not_called()

    def test_next_insufficient_data(self, strategy_fixture):
        """
        Scénario: Pas assez de barres de données pour chauffer le RSI.
        Attendu: La méthode `next` doit retourner (guard clause).
        """
        # Arrange
        strategy_fixture.lines.__len__.return_value = 10
        strategy_fixture.p.rsi_period = 14

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_not_called()
        strategy_fixture.sell.assert_not_called()
        strategy_fixture.log.assert_not_called()

    def test_next_buy_signal_triggered(self, strategy_fixture, mocker):
        """
        Scénario: Pas de position, pas d'ordre, RSI en survente.
        Attendu: Un ordre d'achat est créé.
        """
        # Arrange
        # CORRECTION pour `TypeError: '<' not supported...`
        # Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 25.0
        strategy_fixture.data_close.__getitem__.return_value = 100.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_called_once()
        strategy_fixture.sell.assert_not_called()
        assert strategy_fixture.order == strategy_fixture.buy.return_value
        strategy_fixture.log.assert_called_with(mocker.ANY, level=logging.INFO)
        assert "SIGNAL ACHAT" in strategy_fixture.log.call_args[0][0]

    def test_next_no_buy_signal_not_oversold(self, strategy_fixture):
        """
        Scénario: Pas de position, mais le RSI n'est pas en survente.
        Attendu: Aucun ordre n'est créé.
        """
        # Arrange
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 35.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_not_called()
        strategy_fixture.sell.assert_not_called()
        strategy_fixture.log.assert_not_called()

    def test_next_sell_signal_triggered(self, strategy_fixture, mocker):
        """
        Scénario: En position, pas d'ordre, RSI en surachat.
        Attendu: Un ordre de vente (clôture) est créé.
        """
        # Arrange
        strategy_fixture.mock_position.__bool__.return_value = True
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 75.0
        strategy_fixture.data_close.__getitem__.return_value = 150.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.sell.assert_called_once()
        strategy_fixture.buy.assert_not_called()
        assert strategy_fixture.order == strategy_fixture.sell.return_value
        strategy_fixture.log.assert_called_with(mocker.ANY, level=logging.INFO)
        assert "SIGNAL VENTE" in strategy_fixture.log.call_args[0][0]

    def test_next_no_sell_signal_not_overbought(self, strategy_fixture):
        """
        Scénario: En position, mais le RSI n'est pas en surachat.
        Attendu: Aucun ordre n'est créé (on garde la position).
        """
        # Arrange
        strategy_fixture.mock_position.__bool__.return_value = True
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 65.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.sell.assert_not_called()
        strategy_fixture.buy.assert_not_called()
        strategy_fixture.log.assert_not_called()

    def test_next_in_position_but_oversold(self, strategy_fixture):
        """
        Scénario: En position, et le RSI redevient "oversold".
        Attendu: Rien ne se passe (la logique n'achète pas à nouveau).
        """
        # Arrange
        strategy_fixture.mock_position.__bool__.return_value = True
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 25.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_not_called()
        strategy_fixture.sell.assert_not_called()
        strategy_fixture.log.assert_not_called()

    def test_next_not_in_position_but_overbought(self, strategy_fixture):
        """
        Scénario: Pas de position, et le RSI est "overbought".
        Attendu: Rien ne se passe (la logique n'initie pas de short).
        """
        # Arrange
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 75.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_not_called()
        strategy_fixture.sell.assert_not_called()
        strategy_fixture.log.assert_not_called()

    # --- Tests des cas limites (Edge Cases) ---

    def test_next_buy_at_limit(self, strategy_fixture):
        """
        Scénario: Pas de position, RSI exactement au seuil oversold (30.0).
        Attendu: PAS d'achat (la condition est < 30.0, pas <=).
        """
        # Arrange
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 30.0
        strategy_fixture.p.oversold_level = 30.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_not_called()

    def test_next_buy_just_below_limit(self, strategy_fixture):
        """
        Scénario: Pas de position, RSI juste sous le seuil (29.99).
        Attendu: Achat.
        """
        # Arrange
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 29.99
        strategy_fixture.p.oversold_level = 30.0
        strategy_fixture.data_close.__getitem__.return_value = 100.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.buy.assert_called_once()

    def test_next_sell_at_limit(self, strategy_fixture):
        """
        Scénarial: En position, RSI exactement au seuil overbought (70.0).
        Attendu: PAS de vente (la condition est > 70.0, pas >=).
        """
        # Arrange
        strategy_fixture.mock_position.__bool__.return_value = True
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 70.0
        strategy_fixture.p.overbought_level = 70.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.sell.assert_not_called()

    def test_next_sell_just_above_limit(self, strategy_fixture):
        """
        Scénario: En position, RSI juste au-dessus du seuil (70.01).
        Attendu: Vente.
        """
        # Arrange
        strategy_fixture.mock_position.__bool__.return_value = True
        # CORRECTION : Configurer la valeur de retour de l'appel `[0]`
        strategy_fixture.rsi.__getitem__.return_value = 70.01
        strategy_fixture.p.overbought_level = 70.0
        strategy_fixture.data_close.__getitem__.return_value = 100.0

        # Act
        strategy_fixture.next()

        # Assert
        strategy_fixture.sell.assert_called_once()

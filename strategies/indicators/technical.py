# strategies/indicators/technical.py

from typing import Optional
import pandas as pd
import pandas_ta as ta


class TechnicalIndicators:
    """Wrapper pour les indicateurs techniques pandas-ta."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialise le wrapper d'indicateurs techniques.

        Args:
            df: DataFrame avec colonnes OHLCV standard
        """
        self.df = df.copy()

    def add_rsi(self, length: int = 14, column: str = "close") -> pd.DataFrame:
        """Ajoute le RSI (Relative Strength Index)."""
        self.df.ta.rsi(close=column, length=length, append=True)
        return self.df

    def add_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close"
    ) -> pd.DataFrame:
        """Ajoute le MACD (Moving Average Convergence Divergence)."""
        self.df.ta.macd(close=column, fast=fast, slow=slow, signal=signal, append=True)
        return self.df

    def add_bbands(
        self, length: int = 20, std: float = 2.0, column: str = "close"
    ) -> pd.DataFrame:
        """Ajoute les Bollinger Bands."""
        self.df.ta.bbands(close=column, length=length, std=std, append=True)
        return self.df

    def add_obv(self) -> pd.DataFrame:
        """Ajoute l'OBV (On Balance Volume)."""
        self.df.ta.obv(append=True)
        return self.df

    def add_vwap(self) -> pd.DataFrame:
        """Ajoute le VWAP (Volume Weighted Average Price)."""
        self.df.ta.vwap(append=True)
        return self.df

    def add_atr(self, length: int = 14) -> pd.DataFrame:
        """Ajoute l'ATR (Average True Range) pour la volatilité."""
        self.df.ta.atr(length=length, append=True)
        return self.df

    def add_sma(self, length: int = 20, column: str = "close") -> pd.DataFrame:
        """Ajoute une Simple Moving Average."""
        self.df.ta.sma(close=column, length=length, append=True)
        return self.df

    def add_ema(self, length: int = 20, column: str = "close") -> pd.DataFrame:
        """Ajoute une Exponential Moving Average."""
        self.df.ta.ema(close=column, length=length, append=True)
        return self.df

    def add_all_basic(self) -> pd.DataFrame:
        """Ajoute un ensemble d'indicateurs de base."""
        self.add_rsi()
        self.add_macd()
        self.add_bbands()
        self.add_obv()
        self.add_atr()
        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne le DataFrame avec tous les indicateurs ajoutés."""
        return self.df

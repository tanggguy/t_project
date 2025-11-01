# strategies/indicators/custom_indicators.py

from typing import Optional, Tuple
import pandas as pd
import numpy as np


class CustomIndicators:
    """Indicateurs propriétaires et combinaisons personnalisées."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialise le wrapper d'indicateurs custom.

        Args:
            df: DataFrame avec colonnes OHLCV et indicateurs techniques
        """
        self.df = df.copy()

    def add_trend_strength(self, fast_ma: int = 10, slow_ma: int = 30) -> pd.DataFrame:
        """
        Calcule la force de la tendance basée sur la distance entre MAs.

        Retourne un ratio: (fast_ma - slow_ma) / slow_ma * 100
        """
        if f"SMA_{fast_ma}" not in self.df.columns:
            self.df[f"SMA_{fast_ma}"] = self.df["close"].rolling(window=fast_ma).mean()
        if f"SMA_{slow_ma}" not in self.df.columns:
            self.df[f"SMA_{slow_ma}"] = self.df["close"].rolling(window=slow_ma).mean()

        self.df["TREND_STRENGTH"] = (
            (self.df[f"SMA_{fast_ma}"] - self.df[f"SMA_{slow_ma}"])
            / self.df[f"SMA_{slow_ma}"]
            * 100
        )
        return self.df

    def add_volatility_regime(
        self, atr_length: int = 14, ma_length: int = 50
    ) -> pd.DataFrame:
        """
        Identifie le régime de volatilité: 1 (haute), 0 (normale), -1 (basse).

        Basé sur ATR par rapport à sa moyenne mobile.
        """
        atr_col = (
            f"ATR_{atr_length}" if f"ATR_{atr_length}" in self.df.columns else "ATRr_14"
        )

        if atr_col not in self.df.columns:
            raise ValueError(f"ATR column {atr_col} not found. Add ATR first.")

        atr_ma = self.df[atr_col].rolling(window=ma_length).mean()
        atr_std = self.df[atr_col].rolling(window=ma_length).std()

        self.df["VOLATILITY_REGIME"] = 0
        self.df.loc[self.df[atr_col] > atr_ma + atr_std, "VOLATILITY_REGIME"] = 1
        self.df.loc[self.df[atr_col] < atr_ma - atr_std, "VOLATILITY_REGIME"] = -1

        return self.df

    def add_volume_confirmation(self, volume_ma: int = 20) -> pd.DataFrame:
        """
        Signal de confirmation basé sur le volume.

        1 si volume > moyenne, 0 sinon.
        """
        self.df[f"VOLUME_MA_{volume_ma}"] = (
            self.df["volume"].rolling(window=volume_ma).mean()
        )
        self.df["VOLUME_CONFIRMATION"] = (
            self.df["volume"] > self.df[f"VOLUME_MA_{volume_ma}"]
        ).astype(int)
        return self.df

    def add_price_momentum(self, periods: int = 10) -> pd.DataFrame:
        """
        Calcule le momentum du prix sur N périodes.

        Retourne le changement en pourcentage.
        """
        self.df[f"MOMENTUM_{periods}"] = (
            (self.df["close"] - self.df["close"].shift(periods))
            / self.df["close"].shift(periods)
            * 100
        )
        return self.df

    def add_support_resistance_distance(self, lookback: int = 20) -> pd.DataFrame:
        """
        Calcule la distance au support et résistance récents.

        Support = min des N dernières bougies
        Resistance = max des N dernières bougies
        """
        self.df["SUPPORT"] = self.df["low"].rolling(window=lookback).min()
        self.df["RESISTANCE"] = self.df["high"].rolling(window=lookback).max()

        self.df["DISTANCE_TO_SUPPORT"] = (
            (self.df["close"] - self.df["SUPPORT"]) / self.df["close"] * 100
        )
        self.df["DISTANCE_TO_RESISTANCE"] = (
            (self.df["RESISTANCE"] - self.df["close"]) / self.df["close"] * 100
        )
        return self.df

    def add_multi_timeframe_trend(
        self, short: int = 10, medium: int = 30, long: int = 50
    ) -> pd.DataFrame:
        """
        Alignement de tendance multi-timeframe.

        Retourne un score de -3 à +3 basé sur l'alignement des MAs.
        """
        if f"SMA_{short}" not in self.df.columns:
            self.df[f"SMA_{short}"] = self.df["close"].rolling(window=short).mean()
        if f"SMA_{medium}" not in self.df.columns:
            self.df[f"SMA_{medium}"] = self.df["close"].rolling(window=medium).mean()
        if f"SMA_{long}" not in self.df.columns:
            self.df[f"SMA_{long}"] = self.df["close"].rolling(window=long).mean()

        score = 0
        self.df["MTF_TREND"] = 0

        # Price above/below each MA
        self.df.loc[self.df["close"] > self.df[f"SMA_{short}"], "MTF_TREND"] += 1
        self.df.loc[self.df["close"] < self.df[f"SMA_{short}"], "MTF_TREND"] -= 1

        self.df.loc[self.df["close"] > self.df[f"SMA_{medium}"], "MTF_TREND"] += 1
        self.df.loc[self.df["close"] < self.df[f"SMA_{medium}"], "MTF_TREND"] -= 1

        self.df.loc[self.df["close"] > self.df[f"SMA_{long}"], "MTF_TREND"] += 1
        self.df.loc[self.df["close"] < self.df[f"SMA_{long}"], "MTF_TREND"] -= 1

        return self.df

    def add_rsi_divergence(
        self, rsi_col: str = "RSI_14", lookback: int = 5
    ) -> pd.DataFrame:
        """
        Détecte les divergences RSI (simplifiées).

        1 = divergence haussière, -1 = divergence baissière, 0 = aucune
        """
        if rsi_col not in self.df.columns:
            raise ValueError(f"RSI column {rsi_col} not found. Add RSI first.")

        self.df["RSI_DIVERGENCE"] = 0

        for i in range(lookback, len(self.df)):
            price_trend = self.df["close"].iloc[i] - self.df["close"].iloc[i - lookback]
            rsi_trend = self.df[rsi_col].iloc[i] - self.df[rsi_col].iloc[i - lookback]

            # Divergence haussière: prix baisse mais RSI monte
            if price_trend < 0 and rsi_trend > 0:
                self.df.loc[self.df.index[i], "RSI_DIVERGENCE"] = 1
            # Divergence baissière: prix monte mais RSI baisse
            elif price_trend > 0 and rsi_trend < 0:
                self.df.loc[self.df.index[i], "RSI_DIVERGENCE"] = -1

        return self.df

    def add_composite_signal(
        self,
        rsi_col: str = "RSI_14",
        macd_col: str = "MACDh_12_26_9",
        use_volume: bool = True,
    ) -> pd.DataFrame:
        """
        Signal composite combinant RSI, MACD et volume.

        Retourne un score de -3 à +3.
        """
        if rsi_col not in self.df.columns:
            raise ValueError(f"RSI column {rsi_col} not found.")
        if macd_col not in self.df.columns:
            raise ValueError(f"MACD column {macd_col} not found.")

        self.df["COMPOSITE_SIGNAL"] = 0

        # RSI component
        self.df.loc[self.df[rsi_col] < 30, "COMPOSITE_SIGNAL"] += 1
        self.df.loc[self.df[rsi_col] > 70, "COMPOSITE_SIGNAL"] -= 1

        # MACD component
        self.df.loc[self.df[macd_col] > 0, "COMPOSITE_SIGNAL"] += 1
        self.df.loc[self.df[macd_col] < 0, "COMPOSITE_SIGNAL"] -= 1

        # Volume component
        if use_volume and "VOLUME_CONFIRMATION" in self.df.columns:
            self.df.loc[self.df["VOLUME_CONFIRMATION"] == 1, "COMPOSITE_SIGNAL"] += 1

        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne le DataFrame avec tous les indicateurs custom ajoutés."""
        return self.df

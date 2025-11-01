# Pour MaCrossoverStrategy
data_df.ta.sma(length=10, append=True)
data_df.ta.sma(length=30, append=True)
data_df.dropna(inplace=True)

# Pour une stratégie MACD
data_df.ta.macd(fast=12, slow=26, signal=9, append=True)
data_df.dropna(inplace=True)

# Ou plusieurs indicateurs
data_df.ta.rsi(length=14, append=True)
data_df.ta.macd(fast=12, slow=26, signal=9, append=True)
data_df.ta.bbands(length=20, std=2, append=True)
data_df.dropna(inplace=True)
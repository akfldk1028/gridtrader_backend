import pandas as pd


class RSIAnalyzer:
    def __init__(self, df, period=14, ema_period=14):
        self.df = df
        self.period = period
        self.ema_period = ema_period
        self.compute_rsi()

    def compute_rsi(self):
        # Calculate the difference in price from previous step
        delta = self.df["close"].diff()
        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        # Calculate the EWMA
        roll_up1 = up.ewm(span=self.period).mean()
        roll_down1 = down.ewm(span=self.period).mean()
        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        self.df["RSI"] = RSI1
        # Calculate the SMA
        roll_up2 = up.rolling(self.period).mean()
        roll_down2 = down.rolling(self.period).mean()
        # Calculate the RSI based on SMA
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))
        self.df["RSI_SMA"] = RSI2
        # Calculate EMA of RSI
        self.df["RSI_EMA"] = (
            self.df["RSI"].ewm(span=self.ema_period, adjust=False).mean()
        )
import pandas as pd
import numpy as np


class StochasticRSI:
    def __init__(self, df, lengthRSI=14, lengthStoch=14, smoothK=3, smoothD=3):
        self.df = df
        self.lengthRSI = lengthRSI
        self.lengthStoch = lengthStoch
        self.smoothK = smoothK
        self.smoothD = smoothD
        self.calculate_stochastic_rsi()


    def calculate_stochastic_rsi(self):
        rsi = self.calculate_rsi()
        stoch_rsi = (rsi - rsi.rolling(window=self.lengthStoch).min()) / (
            rsi.rolling(window=self.lengthStoch).max()
            - rsi.rolling(window=self.lengthStoch).min()
        )

        # SMA를 사용하여 %K와 %D 계산
        self.df["%K"] = stoch_rsi.rolling(window=self.smoothK).mean() * 100
        self.df["%D"] = self.df["%K"].rolling(window=self.smoothD).mean()
        # # NaN 값 처리
        self.df["%K"] = self.df["%K"].fillna(0)
        self.df["%D"] = self.df["%D"].fillna(0)

    def calculate_rsi(self):
        delta = self.df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Use exponential moving average (EMA) instead of simple moving average (SMA)
        avg_gain = pd.Series(gain).ewm(span=self.lengthRSI, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(span=self.lengthRSI, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi



import pandas as pd
import numpy as np


class UltimateRSIAnalyzer:
    def __init__(self, df, length=14, smoType1='RMA', smoType2='EMA', smooth=14, price_col='close'):
        """
        df: pandas DataFrame with a 'close' (or user-specified price_col) column
        length: int, period for the main ARSI calculation
        smoType1: str, moving average type for ARSI numerator/denominator (one of 'EMA', 'SMA', 'RMA', 'TMA')
        smoType2: str, moving average type for the signal line (one of 'EMA', 'SMA', 'RMA', 'TMA')
        smooth: int, smoothing period for the signal line
        price_col: str, name of the column in df to use as price source
        """
        self.df = df.copy()
        self.length = length
        self.smoType1 = smoType1
        self.smoType2 = smoType2
        self.smooth = smooth
        self.price_col = price_col
        self._calculate()

    def _rma(self, series, length):
        """RMA (Wilder's smoothing) 구현."""
        rma = series.copy()
        # 초기값: 첫 length개 구간 평균
        if len(rma) < length:
            return pd.Series(np.nan, index=rma.index)

        initial = rma.iloc[:length].mean()
        rma.iloc[:length] = initial
        alpha = 1 / length
        for i in range(length, len(rma)):
            rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i - 1]
        return rma

    def _tma(self, series, length):
        """TMA: SMA를 두 번 적용."""
        sma_once = series.rolling(length, min_periods=length).mean()
        return sma_once.rolling(length, min_periods=length).mean()

    def _moving_average(self, series, length, ma_type):
        if ma_type == 'EMA':
            return series.ewm(span=length, adjust=False).mean()
        elif ma_type == 'SMA':
            return series.rolling(length, min_periods=length).mean()
        elif ma_type == 'RMA':
            return self._rma(series, length)
        elif ma_type == 'TMA':
            return self._tma(series, length)
        else:
            return series

    def _calculate(self):
        src = self.df[self.price_col]

        # 최고/최저값 rolling 계산
        upper = src.rolling(self.length, min_periods=self.length).max()
        lower = src.rolling(self.length, min_periods=self.length).min()
        r = upper - lower

        d = src.diff()

        # diff = upper>upper[1]?r : lower<lower[1]?-r : d
        cond_r = (upper > upper.shift(1))
        cond_l = (lower < lower.shift(1))
        diff = pd.Series(np.where(cond_r, r, np.where(cond_l, -r, d)), index=self.df.index)

        num = self._moving_average(diff, self.length, self.smoType1)
        den = self._moving_average(diff.abs(), self.length, self.smoType1)

        arsi = num / den * 50 + 50
        signal = self._moving_average(arsi, self.smooth, self.smoType2)

        self.df['RSI'] = arsi
        self.df['RSI_signal'] = signal

    def get_dataframe(self):
        """
        계산된 ARSI와 ARSI_signal 컬럼이 추가된 DataFrame 반환
        """
        return self.df

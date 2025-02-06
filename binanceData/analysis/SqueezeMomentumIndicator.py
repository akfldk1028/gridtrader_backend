import pandas as pd
import numpy as np


class SqueezeMomentumIndicator:
    def __init__(self, df, length=14, mult=2.0, lengthKC=14, multKC=1.5, useTrueRange=True):
        """
        df:   pandas DataFrame with 'close', 'high', 'low' columns.
        length:   BB length (default 20)
        mult:     BB multiplier (default 2.0)
        lengthKC: Keltner Channel length (default 20)
        multKC:   Keltner Channel multiplier (default 1.5)
        useTrueRange: bool, whether to use True Range in KC calculation
        """
        self.df = df.copy()
        self.length = length
        self.mult = mult
        self.lengthKC = lengthKC
        self.multKC = multKC
        self.useTrueRange = useTrueRange
        self._calculate()

    def _true_range(self, high, low, close):
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        return tr

    def _linreg(self, series, length):
        """
        linreg(x, length, 0)와 유사한 기능:
        각 지점에서 직전 length개의 데이터에 대해 선형회귀(최소자승법)으로
        종단점(마지막 바)에 해당하는 값을 구함.
        """
        # rolling window로 처리
        # y = a*x + b 형태에서 x는 0,1,2,...,length-1 로 가정
        # 선형회귀 공식:
        # slope = ( n*sum(i*x_i) - sum(i)*sum(x_i) ) / (n*sum(i^2)- (sum(i))^2)
        # intercept = mean(x) - slope*mean(i)
        # 여기서 i는 0~length-1 인덱스, x_i는 series 값
        # 최종값: 마지막 지점은 i=length-1이므로 val = intercept + slope*(length-1)

        x_idx = np.arange(length)
        n = length
        sum_i = np.sum(x_idx)
        sum_i2 = np.sum(x_idx * x_idx)

        # rolling apply
        def linreg_calc(x_window):
            y = x_window.values
            sum_y = np.sum(y)
            sum_i_y = np.sum(x_idx * y)
            slope = (n * sum_i_y - sum_i * sum_y) / (n * sum_i2 - (sum_i ** 2))
            intercept = (sum_y - slope * sum_i) / n
            val = intercept + slope * (length - 1)
            return val

        return series.rolling(length).apply(linreg_calc, raw=False)

    def _calculate(self):
        # PineScript 변수 매핑
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # BB 계산
        basis = close.rolling(self.length).mean()
        dev = self.mult * close.rolling(self.length).std(ddof=0)
        upperBB = basis + dev
        lowerBB = basis - dev

        # KC 계산
        ma = close.rolling(self.lengthKC).mean()
        if self.useTrueRange:
            tr = self._true_range(high, low, close)
        else:
            tr = high - low
        rangema = tr.rolling(self.lengthKC).mean()
        upperKC = ma + rangema * self.multKC
        lowerKC = ma - rangema * self.multKC

        # 스퀴즈 상태
        sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
        noSqz = (~sqzOn) & (~sqzOff)

        # val 계산:
        # val = linreg(source - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close,lengthKC)), lengthKC,0)
        # highest, lowest
        highest_val = high.rolling(self.lengthKC).max()
        lowest_val = low.rolling(self.lengthKC).min()
        mid_val = (highest_val + lowest_val) / 2
        mid_sma = close.rolling(self.lengthKC).mean()
        # avg(x,y) = (x+y)/2
        # 여기서 avg(avg(highest,lowest), sma) = ((mid_val) + mid_sma)/2
        combined_avg = (mid_val + mid_sma) / 2
        source = close  # 원본 스크립트에서 source=close
        input_series = source - combined_avg

        val = self._linreg(input_series, self.lengthKC)

        # 히스토그램 컬러(bcolor)
        # bcolor = iff( val > 0,
        #            iff( val > nz(val[1]), lime, green),
        #            iff( val < nz(val[1]), red, maroon))

        # val.shift(1) 접근
        val_prev = val.shift(1)

        def get_bcolor(cur, prev):
            if pd.isna(prev):
                # 첫 바엔 이전 값 없으니 cur > 0 이면 prev대신 cur 사용
                prev = cur
            if cur > 0:
                if cur > prev:
                    return 'lime'
                else:
                    return 'green'
            else:
                if cur < prev:
                    return 'red'
                else:
                    return 'maroon'

        bcolor = [get_bcolor(c, p) for c, p in zip(val, val_prev)]

        # scolor = noSqz ? blue : sqzOn ? black : gray
        def get_scolor(sqzOnVal, noSqzVal):
            if noSqzVal:
                return 'blue'
            elif sqzOnVal:
                return 'black'
            else:
                return 'gray'

        scolor = [get_scolor(on, no_) for on, no_ in zip(sqzOn, noSqz)]

        self.df['SqueezeMomentum'] = val
        self.df['SqueezeColor'] = bcolor
        # self.df['SqueezeStateColor'] = scolor

    def get_dataframe(self):
        return self.df

# 사용 예시:
# df = pd.DataFrame({
#    'timestamp': [...],
#    'open': [...],
#    'high': [...],
#    'low': [...],
#    'close': [...]
# }).set_index('timestamp')
#
# indicator = SqueezeMomentumIndicator(df)
# result_df = indicator.get_dataframe()
# print(result_df[['SqueezeMomentum', 'SqueezeColor', 'SqueezeStateColor']].head())

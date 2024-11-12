class IchimokuIndicator:
    def __init__(self, df, tenkan=9, kijun=26, senkou_b=52, displacement=26):
        self.df = df.copy()
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.displacement = displacement
        self.calculate_ichimoku()

    def calculate_ichimoku(self):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        # 전환선 (Tenkan-sen)
        self.df['Tenkan_sen'] = (high.rolling(window=self.tenkan).max() + low.rolling(window=self.tenkan).min()) / 2

        # 기준선 (Kijun-sen)
        self.df['Kijun_sen'] = (high.rolling(window=self.kijun).max() + low.rolling(window=self.kijun).min()) / 2

        # 선행스팬 A (Senkou Span A)
        self.df['Senkou_Span_A'] = ((self.df['Tenkan_sen'] + self.df['Kijun_sen']) / 2).shift(self.displacement)

        # 선행스팬 B (Senkou Span B)
        self.df['Senkou_Span_B'] = ((high.rolling(window=self.senkou_b).max() + low.rolling(window=self.senkou_b).min()) / 2).shift(self.displacement)

        # 후행스팬 (Chikou Span)
        self.df['Chikou_Span'] = close.shift(-self.displacement)

    def get_ichimoku(self):
        return self.df[['timestamp', 'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span']]
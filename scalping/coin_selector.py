import os
from crewai import Agent, Task, Crew, Process
from datetime import datetime
import requests
from decimal import Decimal
from .models import CoinScalpingAnalysis


class CoinSelector:
    def __init__(self):
        self.analyst = Agent(
            role='Crypto Scalping Analyst',
            goal='Identify the best cryptocurrencies for scalping trades',
            backstory="""Expert cryptocurrency analyst specializing in scalping opportunities. 
            Skilled at analyzing market volatility, volume, and short-term price movements 
            to identify the best trading opportunities.""",
            verbose=True
        )

    def get_market_data(self, symbol):
        """바이낸스 API에서 시장 데이터 가져오기"""
        session = requests.Session()
        try:
            response = session.get(
                f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}",
                verify=False,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            return response.json()
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None
        finally:
            session.close()

    def analyze_coins(self, coin_list=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']):
        results = []

        for coin in coin_list:
            market_data = self.get_market_data(coin)
            if not market_data:
                continue

            analysis_task = Task(
                description=f"""Analyze {coin} for scalping opportunities:
                현재 가격: {market_data.get('lastPrice')}
                24시간 거래량: {market_data.get('volume')}
                24시간 가격변화: {market_data.get('priceChangePercent')}%

                Analyze the following aspects and provide a scalping score from 0-100:
                1. Volume and liquidity
                2. Price volatility and movement patterns
                3. Overall scalping suitability

                Format your response exactly as follows:

                Scalping Score: [0-100]
                Priority: [HIGH/MEDIUM/LOW]
                Analysis: [Detailed explanation of why this coin is or isn't suitable for scalping]
                """,
                agent=self.analyst,
                expected_output="Structured analysis of scalping potential"
            )

            crew = Crew(
                agents=[self.analyst],
                tasks=[analysis_task],
                verbose=True
            )

            result = crew.kickoff()
            result_str = str(result)

            try:
                # Parse results and create record
                analysis = CoinScalpingAnalysis.objects.create(
                    coin_symbol=coin,
                    current_price=Decimal(str(market_data.get('lastPrice', '0'))),
                    volume_24h=Decimal(str(market_data.get('volume', '0'))),
                    price_change_24h=Decimal(str(market_data.get('priceChangePercent', '0'))),
                    scalping_score=self.extract_score(result_str),
                    priority=self.extract_priority(result_str),
                    analysis=self.extract_analysis(result_str)
                )
                results.append(analysis)
            except Exception as e:
                print(f"Error processing analysis for {coin}: {e}")

        return results

    def extract_score(self, text):
        try:
            for line in text.split('\n'):
                if 'Scalping Score:' in line:
                    score = Decimal(line.split(':')[1].strip().split()[0])
                    return min(max(score, Decimal('0')), Decimal('100'))
            return Decimal('0')
        except:
            return Decimal('0')

    def extract_priority(self, text):
        for line in text.split('\n'):
            if 'Priority:' in line:
                priority = line.split(':')[1].strip()
                if priority in ['HIGH', 'MEDIUM', 'LOW']:
                    return priority
        return 'LOW'

    def extract_analysis(self, text):
        for line in text.split('\n'):
            if 'Analysis:' in line:
                return line.split(':')[1].strip()
        return ''


def analyze_scalping_opportunities():
    selector = CoinSelector()
    coins_to_analyze = ['BTCUSDT', 'ETHUSDT', 'LINKUSDT', 'DOGEUSDT', 'SOLUSDT']
    analyses = selector.analyze_coins(coins_to_analyze)

    # 높은 우선순위의 코인들 반환
    return CoinScalpingAnalysis.objects.filter(
        priority='HIGH'
    ).order_by('-scalping_score')[:5]
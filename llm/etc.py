from TradeStrategy.models import StrategyConfig
from django.core.exceptions import ObjectDoesNotExist


def get_strategy_config(strategy_name='240824'):
    try:
        strategy_config = StrategyConfig.objects.get(name=strategy_name)
        first_config = strategy_config.config['INIT']

        vt_symbol = first_config.get('vt_symbol')
        symbol = vt_symbol.split('.')[0]  # "BNBUSDT.BINANCE"에서 "BNBUSDT" 추출
        grid_strategy = first_config['setting'].get('grid_strategy')


        return {
            'vt_symbol': symbol,
            'grid_strategy': grid_strategy
        }
    except ObjectDoesNotExist:
        print(f"Strategy configuration not found for: {strategy_name}")
        return None


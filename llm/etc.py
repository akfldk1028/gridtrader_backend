from TradeStrategy.models import StrategyConfig
from django.core.exceptions import ObjectDoesNotExist
from asgiref.sync import sync_to_async
from django.db import transaction

@sync_to_async
def get_strategy_config(strategy_name='240824'):
    try:
        with transaction.atomic():
            strategy_config = StrategyConfig.objects.get(name=strategy_name)
            first_config = strategy_config.config.get('INIT', {})

            vt_symbol = first_config.get('vt_symbol', '')
            symbol = vt_symbol.split('.')[0] if vt_symbol else ''
            grid_strategy = first_config.get('setting', {}).get('grid_strategy')

            if not symbol or not grid_strategy:
                raise ValueError("Invalid configuration: missing symbol or grid_strategy")

            return {
                'vt_symbol': symbol,
                'grid_strategy': grid_strategy
            }
    except ObjectDoesNotExist:
        print(f"Strategy configuration not found for: {strategy_name}")
    except ValueError as e:
        print(f"Invalid configuration for {strategy_name}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in get_strategy_config: {str(e)}")

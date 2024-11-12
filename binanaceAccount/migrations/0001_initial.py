# Generated by Django 5.1 on 2024-08-26 01:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BinanceFuturePosition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(help_text='거래 쌍 심볼', max_length=20)),
                ('position_amt', models.DecimalField(decimal_places=8, help_text='포지션 수량', max_digits=30)),
                ('entry_price', models.DecimalField(decimal_places=8, help_text='진입 가격', max_digits=30)),
                ('mark_price', models.DecimalField(decimal_places=8, help_text='마크 가격', max_digits=30)),
                ('un_realized_profit', models.DecimalField(decimal_places=8, help_text='미실현 손익', max_digits=30)),
                ('leverage', models.IntegerField(help_text='레버리지')),
                ('profit_percentage', models.DecimalField(decimal_places=8, help_text='수익률 (%)', max_digits=30)),
            ],
        ),
        migrations.CreateModel(
            name='BinanceOrder',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbol', models.CharField(help_text='거래 쌍 심볼', max_length=20)),
                ('order_id', models.BigIntegerField(help_text='Binance에서 제공한 고유 주문 ID')),
                ('side', models.CharField(help_text='주문 방향 (매수 또는 매도)', max_length=50)),
                ('type', models.CharField(help_text='주문 유형 (예: 시장가, 지정가)', max_length=20)),
                ('quantity', models.DecimalField(decimal_places=8, help_text='주문 수량', max_digits=30)),
                ('reduce_only', models.BooleanField(help_text='포지션 감소 전용 주문 여부')),
                ('price', models.DecimalField(blank=True, decimal_places=8, help_text='주문 가격 (해당되는 경우)', max_digits=30, null=True)),
                ('status', models.CharField(help_text='주문의 현재 상태', max_length=20)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='BinanceSymbolSettings',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbol', models.CharField(help_text='거래 쌍 심볼', max_length=20)),
                ('leverage', models.IntegerField(help_text='이 심볼에 대한 현재 레버리지 설정')),
                ('margin_type', models.CharField(help_text='마진 유형 (격리 또는 교차)', max_length=30)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='SpotBalance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('asset', models.CharField(help_text='자산 심볼', max_length=10)),
                ('free', models.DecimalField(decimal_places=8, help_text='사용 가능한 자산 수량', max_digits=30)),
                ('locked', models.DecimalField(decimal_places=8, help_text='잠긴 자산 수량', max_digits=30)),
            ],
        ),
        migrations.CreateModel(
            name='FutureBalance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('asset', models.CharField(help_text='자산 심볼 (USDT)', max_length=10)),
                ('balance', models.DecimalField(decimal_places=8, help_text='전체 잔액', max_digits=30)),
                ('cross_wallet_balance', models.DecimalField(decimal_places=8, help_text='교차 마진 지갑 잔액', max_digits=30)),
                ('cross_un_pnl', models.DecimalField(decimal_places=8, help_text='교차 마진 미실현 손익', max_digits=30)),
                ('available_balance', models.DecimalField(decimal_places=8, help_text='사용 가능한 잔액', max_digits=30)),
                ('max_withdraw_amount', models.DecimalField(decimal_places=8, help_text='최대 출금 가능 금액', max_digits=30)),
                ('margin_available', models.BooleanField(help_text='마진 사용 가능 여부')),
                ('update_time', models.BigIntegerField(blank=True, help_text='마지막 업데이트 시간 (밀리초)', null=True)),
                ('positions', models.ManyToManyField(blank=True, help_text='연관된 선물 포지션들', related_name='future_positions', to='binanaceAccount.binancefutureposition')),
            ],
        ),
        migrations.CreateModel(
            name='BinanceFuturesAccount',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('account_type', models.CharField(default='FUTURES', help_text="계정 유형 (항상 'FUTURES')", max_length=10)),
                ('total_wallet_balance', models.DecimalField(decimal_places=8, help_text='전체 지갑 잔액', max_digits=30)),
                ('total_unrealized_profit', models.DecimalField(decimal_places=8, help_text='전체 미실현 손익', max_digits=30)),
                ('total_margin_balance', models.DecimalField(decimal_places=8, help_text='전체 마진 잔액', max_digits=30)),
                ('available_balance', models.DecimalField(decimal_places=8, help_text='사용 가능한 잔액', max_digits=30)),
                ('max_withdraw_amount', models.DecimalField(decimal_places=8, help_text='최대 출금 가능 금액', max_digits=30)),
                ('futurebalance', models.ForeignKey(blank=True, help_text='연관된 현물 계정', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='future_balance', to='binanaceAccount.futurebalance')),
            ],
        ),
        migrations.CreateModel(
            name='BinanceAccount',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('account_type', models.CharField(choices=[('SPOT', 'Spot'), ('FUTURES', 'Futures')], default='SPOT', help_text='계정 유형 (현물 또는 선물)', max_length=10)),
                ('can_trade', models.BooleanField(help_text='거래 가능 여부')),
                ('can_withdraw', models.BooleanField(help_text='출금 가능 여부')),
                ('can_deposit', models.BooleanField(help_text='입금 가능 여부')),
                ('update_time', models.BigIntegerField(help_text='마지막 업데이트 시간 (밀리초)')),
                ('maker_commission', models.IntegerField(help_text='메이커 수수료 비율')),
                ('taker_commission', models.IntegerField(help_text='테이커 수수료 비율')),
                ('spotbalance', models.ForeignKey(blank=True, help_text='연관된 현물 계정', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='spot_balances', to='binanaceAccount.spotbalance')),
            ],
        ),
    ]

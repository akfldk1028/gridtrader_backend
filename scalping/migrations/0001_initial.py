# Generated by Django 5.1 on 2024-11-12 00:09

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TradingRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('timestamp', models.DateTimeField(verbose_name='거래 시간')),
                ('coin_symbol', models.CharField(max_length=20, verbose_name='코인 심볼')),
                ('trade_type', models.CharField(choices=[('BUY', 'Buy'), ('SELL', 'Sell'), ('HOLD', 'Hold')], max_length=4, verbose_name='거래 유형')),
                ('trade_ratio', models.DecimalField(decimal_places=4, max_digits=5, verbose_name='거래 비율(%)')),
                ('trade_amount_krw', models.DecimalField(decimal_places=2, max_digits=20, verbose_name='거래 금액(KRW)')),
                ('trade_reason', models.TextField(verbose_name='거래 이유')),
                ('coin_balance', models.DecimalField(decimal_places=8, max_digits=20, verbose_name='코인 보유량')),
                ('balance', models.DecimalField(decimal_places=2, max_digits=20, verbose_name='USDT/KRW 보유량')),
                ('current_price', models.DecimalField(decimal_places=2, max_digits=20, verbose_name='현재 가격')),
                ('trade_reflection', models.TextField(verbose_name='거래 반성')),
            ],
            options={
                'ordering': ['-timestamp'],
                'indexes': [models.Index(fields=['-timestamp'], name='scalping_tr_timesta_375033_idx'), models.Index(fields=['coin_symbol'], name='scalping_tr_coin_sy_16de82_idx')],
            },
        ),
    ]

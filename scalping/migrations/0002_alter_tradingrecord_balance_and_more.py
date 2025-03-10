# Generated by Django 5.1 on 2024-11-12 05:42

from decimal import Decimal
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scalping', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tradingrecord',
            name='balance',
            field=models.DecimalField(decimal_places=2, default=Decimal('0.00'), max_digits=20, verbose_name='USDT/KRW 보유량'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='coin_balance',
            field=models.DecimalField(decimal_places=8, default=Decimal('0E-8'), max_digits=20, verbose_name='코인 보유량'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='current_price',
            field=models.DecimalField(decimal_places=2, default=Decimal('0.00'), max_digits=20, verbose_name='현재 가격'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='trade_amount_krw',
            field=models.DecimalField(decimal_places=2, default=Decimal('0.00'), max_digits=20, verbose_name='거래 금액(KRW)'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='trade_ratio',
            field=models.DecimalField(decimal_places=4, default=Decimal('0.0000'), max_digits=5, verbose_name='거래 비율(%)'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='trade_reason',
            field=models.TextField(default='', verbose_name='거래 이유'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='trade_reflection',
            field=models.TextField(default='', verbose_name='거래 반성'),
        ),
        migrations.AlterField(
            model_name='tradingrecord',
            name='trade_type',
            field=models.CharField(choices=[('BUY', 'Buy'), ('SELL', 'Sell'), ('HOLD', 'Hold')], default='HOLD', max_length=4, verbose_name='거래 유형'),
        ),
    ]

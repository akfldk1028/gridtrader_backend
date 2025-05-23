# Generated by Django 5.1 on 2024-12-18 04:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('binanceData', '0003_binancetradingsummary'),
    ]

    operations = [
        migrations.CreateModel(
            name='KoreaStockData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbols', models.JSONField(blank=True, null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='StockData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('symbols', models.JSONField(blank=True, null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]

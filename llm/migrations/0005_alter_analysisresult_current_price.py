# Generated by Django 5.1 on 2024-09-09 04:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('llm', '0004_analysisresult_korean_summary'),
    ]

    operations = [
        migrations.AlterField(
            model_name='analysisresult',
            name='current_price',
            field=models.FloatField(blank=True, default=0, null=True),
        ),
    ]

# Generated by Django 5.1 on 2024-09-14 12:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('llm', '0005_alter_analysisresult_current_price'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysisresult',
            name='analysis_results_1hour',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='analysisresult',
            name='analysis_results_30m',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='analysisresult',
            name='analysis_results_daily',
            field=models.TextField(blank=True, null=True),
        ),
    ]

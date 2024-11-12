# Generated by Django 5.1 on 2024-08-29 02:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('binanaceAccount', '0002_dailybalance_remove_futurebalance_positions_and_more'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='dailybalance',
            options={'get_latest_by': 'created_at', 'ordering': ['-created_at']},
        ),
        migrations.RemoveField(
            model_name='dailybalance',
            name='date',
        ),
    ]

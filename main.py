"""
Главный скрипт для проведения вычислительного эксперимента.
Апробация контура автоматического управления (Глава 3).
"""

import numpy as np
import torch

from config import SEED
from trainer import train_model
from experiment import run_full_experiment
from visualization import plot_all_scenarios


def main():
    """Основная функция эксперимента."""
    print("=" * 70)
    print("АПРОБАЦИЯ КОНТУРА АВТОМАТИЧЕСКОГО УПРАВЛЕНИЯ")
    print("=" * 70)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = train_model()

    results_df, false_alarm_rate = run_full_experiment(model)

    print("\n" + "=" * 70)
    print("Таблица 1 - Результаты апробации (среднее по 30 реализациям)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nS4: доля тактов с ложным срабатыванием регулятора: {false_alarm_rate:.2f}%")

    plot_all_scenarios(model)

    print("\nЭксперимент завершён успешно!")


if __name__ == '__main__':
    main()
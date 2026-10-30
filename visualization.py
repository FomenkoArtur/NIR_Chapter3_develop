"""
Визуализация результатов эксперимента.
"""

import matplotlib.pyplot as plt
import numpy as np
from config import G, DELTA, T_START, SEED, SCENARIOS
from experiment import run_experiment_episode


def plot_transition_process(model, scenario='S1'):
    """
    Построение графика переходного процесса для одного сценария.
    """
    K_no_ctrl, _, _ = run_experiment_episode(scenario, 'none', seed=SEED)
    K_ctrl, _, _ = run_experiment_episode(scenario, 'cnn', seed=SEED, model=model)

    plt.figure(figsize=(10, 6))

    plt.plot(K_no_ctrl, color='red', alpha=0.8, label='Без регулятора')
    plt.plot(K_ctrl, color='blue', lw=2, label='С регулятором (CNN)')

    plt.axhline(G, color='k', alpha=0.3, label=f'Программное задание g = {G}')

    plt.axhline(G + DELTA, color='green', ls='--', alpha=0.6, label=f'Допустимая полоса +/- {DELTA}')
    plt.axhline(G - DELTA, color='green', ls='--', alpha=0.6)

    if scenario != 'S4':
        plt.axvline(T_START, color='gray', ls=':', label='Момент возникновения особой причины')

    values = np.concatenate([K_no_ctrl, K_ctrl, [G - DELTA, G + DELTA]])
    margin = 0.05 * np.ptp(values)
    plt.ylim(values.min() - margin, values.max() + margin)

    desc = SCENARIOS[scenario]['description']
    plt.title(f'Переходный процесс суперкритерия K(t) — {desc}')
    plt.xlabel('Такт моделирования (t)')
    plt.ylabel('Суперкритерий K(t)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f'Рисунок_{scenario}_переходный_процесс.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Сохранен график: {filename}")

    plt.show()


def plot_all_scenarios(model):
    """
    Построение графиков для всех сценариев (S4, S1, S2, S3).
    """
    print("\nПостроение графиков для всех сценариев...")
    for scenario in ['S4', 'S1', 'S2', 'S3']:
        plot_transition_process(model, scenario=scenario)
    print("Все графики построены и сохранены.")
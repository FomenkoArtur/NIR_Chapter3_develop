"""
Проведение вычислительного эксперимента (п. 3.3.1).
"""

import numpy as np
import pandas as pd

from config import T_START, N_REAL, SEED_EXPERIMENT_START, SEED_FALSE_ALARM_START
from episode import CNNPolicy, NullPolicy, run_episode
from scenarios import get_scenario_description
from metrics import compute_metrics, out_of_control_rate


def run_experiment_episode(scenario, mode, seed=None, model=None):
    """
    Прогон одного эпизода эксперимента.
    """
    policy = CNNPolicy(model) if mode == 'cnn' and model is not None else NullPolicy()
    episode = run_episode(scenario, policy, seed=seed)
    return episode.K, episode.eps, episode.u


def run_full_experiment(model):
    """
    Проведение полного вычислительного эксперимента.
    """
    print("Вычислительный эксперимент: 30 реализаций на сценарий...")

    results = []

    for scenario in ['S1', 'S2', 'S3']:
        ts_list = []
        iae_no_ctrl_list = []
        iae_ctrl_list = []
        oc_no_ctrl_list = []
        oc_ctrl_list = []

        for rep in range(N_REAL):
            seed = SEED_EXPERIMENT_START + 1000 * {'S1': 1, 'S2': 2, 'S3': 3}[scenario] + rep

            base = run_episode(scenario, NullPolicy(), seed=seed)
            _, iae_no_ctrl, _ = compute_metrics(base.eps, base.u, T_START)

            ctrl = run_episode(scenario, CNNPolicy(model), seed=seed)
            ts_ctrl, iae_ctrl, _ = compute_metrics(ctrl.eps, ctrl.u, T_START)

            ts_list.append(ts_ctrl)
            iae_no_ctrl_list.append(iae_no_ctrl)
            iae_ctrl_list.append(iae_ctrl)
            oc_no_ctrl_list.append(100 * out_of_control_rate(base.N, T_START))
            oc_ctrl_list.append(100 * out_of_control_rate(ctrl.N, T_START))

        mean_ts = round(np.mean(ts_list))
        mean_iae_no = round(np.mean(iae_no_ctrl_list), 1)
        mean_iae_ctrl = round(np.mean(iae_ctrl_list), 1)
        reduction = round(100 * (1 - mean_iae_ctrl / mean_iae_no))
        mean_oc_no = round(np.mean(oc_no_ctrl_list))
        mean_oc_ctrl = round(np.mean(oc_ctrl_list))

        description = get_scenario_description(scenario)
        results.append([scenario, description, mean_ts, mean_iae_no, mean_iae_ctrl, reduction,
                        mean_oc_no, mean_oc_ctrl])

    # Оценка ложных срабатываний в S4
    false_alarm_rates = []
    for rep in range(N_REAL):
        seed = SEED_FALSE_ALARM_START + rep
        episode = run_episode('S4', CNNPolicy(model), seed=seed)
        _, _, idle_rate = compute_metrics(episode.eps, episode.u, T_START)
        false_alarm_rates.append(idle_rate)

    mean_false_alarm = 100 * np.mean(false_alarm_rates)

    df = pd.DataFrame(
        results,
        columns=['Сценарий', 'Особая причина изменчивости', 'Ts, тактов (с регулятором)',
                 'IAE без регулятора', 'IAE с регулятором', 'Снижение IAE, %',
                 'Nᵢ вне [0,05; 0,95] без регулятора, %', 'Nᵢ вне [0,05; 0,95] с регулятором, %']
    )

    means = df.iloc[:, 2:].mean().round(1)
    row = ['Среднее', '-'] + means.tolist()
    row[5] = int(means.iloc[3])
    row[6] = int(round(means.iloc[4]))
    row[7] = int(round(means.iloc[5]))
    df.loc[len(df)] = row

    return df, mean_false_alarm

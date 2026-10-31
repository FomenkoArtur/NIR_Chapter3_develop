"""
Проведение вычислительного эксперимента (п. 3.3.1).
"""

import numpy as np
import pandas as pd

from config import T_START, N_REAL, SEED_EXPERIMENT_START, SEED_FALSE_ALARM_START
from episode import CNNPolicy, NullPolicy, run_episode
from fastsim import BatchCNNPolicy, BatchNullPolicy, episode_noise, run_episodes_batch
from scenarios import get_disturbance_profile, get_scenario_description
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
    Проведение полного вычислительного эксперимента: 30 реализаций сценария считаются одним пакетом.
    """
    print("Вычислительный эксперимент: 30 реализаций на сценарий...")

    results = []

    for scenario in ['S1', 'S2', 'S3']:
        ts_list = []
        iae_no_ctrl_list = []
        iae_ctrl_list = []
        oc_no_ctrl_list = []
        oc_ctrl_list = []

        seeds = [SEED_EXPERIMENT_START + 1000 * {'S1': 1, 'S2': 2, 'S3': 3}[scenario] + rep
                 for rep in range(N_REAL)]
        profiles = [get_disturbance_profile(scenario, T_START, 1.0)] * N_REAL
        noise = np.stack([episode_noise(seed) for seed in seeds])
        t0s = [T_START] * N_REAL

        base = run_episodes_batch(profiles, BatchNullPolicy(), noise, t0s)
        ctrl = run_episodes_batch(profiles, BatchCNNPolicy(model), noise, t0s)   # сеть: 1 вызов на такт

        for rep in range(N_REAL):
            _, iae_no_ctrl, _ = compute_metrics(base.eps[rep], base.u[rep], T_START)
            ts_ctrl, iae_ctrl, _ = compute_metrics(ctrl.eps[rep], ctrl.u[rep], T_START)

            ts_list.append(ts_ctrl)
            iae_no_ctrl_list.append(iae_no_ctrl)
            iae_ctrl_list.append(iae_ctrl)
            oc_no_ctrl_list.append(100 * out_of_control_rate(base.N[rep], T_START))
            oc_ctrl_list.append(100 * out_of_control_rate(ctrl.N[rep], T_START))

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
    seeds = [SEED_FALSE_ALARM_START + rep for rep in range(N_REAL)]
    noise = np.stack([episode_noise(seed) for seed in seeds])
    episodes = run_episodes_batch([get_disturbance_profile('S4', T_START, 1.0)] * N_REAL,
                                  BatchCNNPolicy(model), noise, [T_START] * N_REAL)
    false_alarm_rates = [compute_metrics(episodes.eps[rep], episodes.u[rep], T_START)[2]
                         for rep in range(N_REAL)]

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
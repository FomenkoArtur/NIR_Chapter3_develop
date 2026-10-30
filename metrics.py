"""
Расчёт метрик качества переходного процесса (формулы (33), (36), п. 3.1.3).
"""

import numpy as np
from config import T_OBS, DELTA, N_CONS, IDLE_THRESHOLD, N_LOW, N_HIGH


def compute_metrics(eps_history, u_history, t0):
    """
    Вычисление метрик качества переходного процесса.
    """
    eps_obs = eps_history[t0:t0+T_OBS]
    u_obs = u_history[t0:t0+T_OBS]

    iae = float(np.abs(eps_obs).sum())

    ts = T_OBS
    for t in range(len(eps_obs) - N_CONS + 1):
        if np.all(np.abs(eps_obs[t:t+N_CONS]) <= DELTA):
            ts = t
            break

    u_abs = np.abs(u_obs)
    if u_abs.ndim > 1:
        u_abs = u_abs.max(axis=1)
    idle_rate = float(np.mean(u_abs > IDLE_THRESHOLD))

    return ts, iae, idle_rate


def out_of_control_rate(n_history, t0):
    """
    Доля тактов горизонта наблюдения, в которых хотя бы один нормализованный
    показатель N_i вышел за границы [N_LOW; N_HIGH] ~ [0,05; 0,95], то есть
    отклонился от медианы окна более чем на 3 s_i (п. 2.1.1).
    Робастная сигмоида принимает значения строго в (0; 1), поэтому
    граница [0; 1] для этого признака неприменима.
    """
    n_obs = n_history[t0:t0+T_OBS]
    outside = (n_obs < N_LOW) | (n_obs > N_HIGH)
    return float(np.mean(outside.any(axis=1)))

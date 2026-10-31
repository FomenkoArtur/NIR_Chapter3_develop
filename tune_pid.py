"""
Автонастройка ПИД-учителя: CMA-ES по цифровому двойнику.
Запуск: python tune_pid.py   ->  tuned/pid.json
Кандидаты каждого поколения CMA-ES считаются одним векторным пакетом (fastsim.py).
"""

import json
import os

import numpy as np

import config as C
from fastsim import BatchPIDPolicy, episode_noise, run_episodes_batch
from metrics import compute_metrics
from scenarios import custom_profile, get_disturbance_profile

SEED_PID_TUNE = 120000          # зёрна, не пересекающиеся с обучением/тестом/экспериментом
MANUAL = (1.6, 0.05, 0.20, 10)  # ручные значения (Kp, Ki, Kd, LI) - для сравнения
W_TS, W_TV, W_FA = 3.0, 0.5, 2.0  # веса: время перехода, "дёрганье" u, ложные срабатывания

# пространство поиска: log10 Kp, log10 Ki, log10 Kd, LI
LO = np.array([np.log10(0.3), -3.0, -3.0, 2.0])
HI = np.array([np.log10(6.0), np.log10(0.5), np.log10(2.0), 32.0])


def build_cases(seed0=SEED_PID_TUNE):
    """Тренд и скачок на каждый из 8 показателей, знак чередуется."""
    trend = C.SCENARIOS['S1']['rate_value']
    jump = abs(C.SCENARIOS['S2']['jump_value'])
    cases, k = [], 0
    for p in range(C.N_PARAMS):
        for kind, val in (('trend', trend), ('jump', jump)):
            sign = 1 if (p + (kind == 'jump')) % 2 == 0 else -1
            cases.append((custom_profile(p, kind, sign * val, C.T_START), seed0 + k))
            k += 1
    quiet = [seed0 + 5000 + i for i in range(3)]  # S4 - без возмущений
    return cases, quiet


def evaluate(xs, cases, quiet):
    """
    Функция качества (меньше - лучше) сразу для всех кандидатов xs: один пакет
    "кандидаты x (16 случаев + 3 эпизода S4)". Шум каждого эпизода зависит только от зерна.
    """
    n_c, n_e = len(xs), len(cases) + len(quiet)
    profiles = [p for p, _ in cases] + [get_disturbance_profile('S4', C.T_START, 1.0)] * len(quiet)
    noise = np.stack([episode_noise(s) for s in [s for _, s in cases] + list(quiet)])
    params = [decode(x) for x in xs]
    kp, ki, kd, li = (np.repeat([p[i] for p in params], n_e) for i in range(4))

    batch = run_episodes_batch(profiles * n_c, BatchPIDPolicy(kp, ki, kd, li),
                               np.tile(noise, (n_c, 1, 1)), [C.T_START] * (n_c * n_e))
    costs = []
    for c in range(n_c):
        iae, ts, tv, fa = [], [], [], []
        for e in range(n_e):
            b = c * n_e + e
            t, a, idle = compute_metrics(batch.eps[b], batch.u[b], C.T_START)
            if e < len(cases):
                u = batch.u[b, C.T_START:C.T_START + C.T_OBS]
                iae.append(a)
                ts.append(t)
                tv.append(np.abs(np.diff(u, axis=0)).sum(axis=1).mean())
            else:
                fa.append(idle)
        costs.append(float(np.mean(iae) + W_TS * np.mean(ts) / C.T_OBS + W_TV * np.mean(tv)
                           + W_FA * np.mean(fa)))
    return costs


def decode(u):
    v = LO + np.clip(u, 0, 1) * (HI - LO)
    return 10 ** v[0], 10 ** v[1], 10 ** v[2], int(round(v[3]))


def encode(kp, ki, kd, li):
    return (np.array([np.log10(kp), np.log10(ki), np.log10(kd), li]) - LO) / (HI - LO)


def main():
    import cma

    cases, quiet = build_cases()
    cost = lambda x: evaluate([x], cases, quiet)[0]

    x0 = encode(*MANUAL)
    base = cost(x0)
    print(f"Ручные значения {MANUAL}: cost = {base:.4f}")

    es = cma.CMAEvolutionStrategy(x0, 0.25, {'bounds': [0, 1], 'popsize': 8, 'maxiter': 40,
                                             'seed': C.SEED, 'verbose': -9})
    while not es.stop():
        xs = es.ask()
        fs = evaluate(xs, cases, quiet)
        es.tell(xs, fs)
        print(f"  поколение {es.countiter}: лучший cost = {min(fs):.4f}")

    best = es.result.xbest
    kp, ki, kd, li = decode(best)
    best_cost = cost(best)
    print(f"Найдено: Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f} LI={li}: cost = {best_cost:.4f}")

    if best_cost < base:
        os.makedirs('tuned', exist_ok=True)
        with open(os.path.join('tuned', 'pid.json'), 'w', encoding='utf-8') as f:
            json.dump({'KP': round(kp, 4), 'KI': round(ki, 4), 'KD': round(kd, 4),
                       'PID_WINDOW': li, 'cost': best_cost, 'cost_manual': base}, f, indent=2)
        print("Сохранено в tuned/pid.json")
    else:
        print("Улучшения относительно ручных значений нет, файл не записан.")


if __name__ == '__main__':
    main()
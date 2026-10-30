"""
Автонастройка ПИД-учителя: CMA-ES по цифровому двойнику.
Запуск: python tune_pid.py   ->  tuned/pid.json
"""

import json
import os

import numpy as np

import config as C
from episode import PIDPolicy, run_episode
from metrics import compute_metrics
from scenarios import custom_profile

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


def evaluate(make_policy, cases, quiet):
    """Общая функция качества для ПИД и CNN (меньше - лучше)."""
    iae, ts, tv, fa = [], [], [], []
    for profile, seed in cases:
        ep = run_episode('tune', make_policy(), seed=seed, profile=profile)
        t, a, _ = compute_metrics(ep.eps, ep.u, C.T_START)
        u = ep.u[C.T_START:C.T_START + C.T_OBS]
        iae.append(a)
        ts.append(t)
        tv.append(np.abs(np.diff(u, axis=0)).sum(axis=1).mean())
    for seed in quiet:
        ep = run_episode('S4', make_policy(), seed=seed)
        fa.append(compute_metrics(ep.eps, ep.u, C.T_START)[2])
    return float(np.mean(iae) + W_TS * np.mean(ts) / C.T_OBS + W_TV * np.mean(tv) + W_FA * np.mean(fa))


def decode(u):
    v = LO + np.clip(u, 0, 1) * (HI - LO)
    return 10 ** v[0], 10 ** v[1], 10 ** v[2], int(round(v[3]))


def encode(kp, ki, kd, li):
    return (np.array([np.log10(kp), np.log10(ki), np.log10(kd), li]) - LO) / (HI - LO)


def main():
    import cma  # импорт здесь, чтобы tune_cnn.py не требовал cma

    cases, quiet = build_cases()
    cost = lambda x: evaluate(lambda: PIDPolicy(*decode(x)), cases, quiet)

    x0 = encode(*MANUAL)
    base = cost(x0)
    print(f"Ручные значения {MANUAL}: cost = {base:.4f}")

    es = cma.CMAEvolutionStrategy(x0, 0.25, {'bounds': [0, 1], 'popsize': 8, 'maxiter': 40,
                                             'seed': C.SEED, 'verbose': -9})
    while not es.stop():
        xs = es.ask()
        fs = [cost(x) for x in xs]
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
"""
Автонастройка ПИД-учителя: CMA-ES по цифровому двойнику.
Запуск: python tune_pid.py   ->  tuned/pid_int.json (интегратор) или tuned/pid.json (NIR_PID_INT=window)
Кандидаты каждого поколения CMA-ES считаются одним векторным пакетом (fastsim.py).
"""

import json
import os

import numpy as np

import config as C
from fastsim import BatchPIDPolicy, episode_noise, run_episodes_batch
from metrics import compute_metrics
from plant import plant                                 # нужна матрица V, чтобы считать ошибки зон
from scenarios import custom_profile, get_disturbance_profile

SEED_PID_TUNE = 120000          # зёрна, не пересекающиеся с обучением/тестом/экспериментом
SEED_PID_VAL = 130000           # отдельные зёрна для проверки результата: тюнер их не видит
INTEGRATOR = getattr(C, 'PID_INTEGRATOR', 'window')     # 'true' - настоящий интегратор, 'window' - сумма по окну (config.py)
MANUAL = (1.6, 0.10 if INTEGRATOR == 'true' else 0.05, 0.20, 10)   # ручные значения (Kp, Ki, Kd, LI) - для сравнения
PID_FILE = 'pid_int.json' if INTEGRATOR == 'true' else 'pid.json'  # результат: у режимов разный смысл Ki
W_TS, W_TV = 3.0, 0.5           # веса: время перехода, "дёрганье" u
W_SS, SS_TAIL = 10.0, 30        # вес статической ошибки при тренде и число последних тактов, по которым она считается
W_OS, OS_FROM = 5.0, 10         # вес перерегулирования на скачке и с какого такта после скачка его искать
N_QUIET, N_QUIET_VAL = 24, 30   # число эпизодов S4 (без возмущений) при подборе и при проверке
FA_LIMIT = 0.005                # допустимая доля тактов с ложным срабатыванием (0,5 %)
W_FA, W_FA_HARD = 20.0, 200.0   # штраф за ложные срабатывания: линейный и добавочный сверх FA_LIMIT

# пространство поиска: log10 Kp, log10 Ki, log10 Kd, LI
LO = np.array([np.log10(0.3), -3.0, -3.0, 2.0])
HI = np.array([np.log10(6.0), np.log10(0.5), np.log10(2.0), 32.0])


def build_cases(seed0=SEED_PID_TUNE, n_quiet=N_QUIET):    # n_quiet - сколько тихих эпизодов S4 сделать
    """Тренд и скачок на каждый из 8 показателей, знак чередуется."""
    trend = C.SCENARIOS['S1']['rate_value']                # скорость тренда из сценария S1
    jump = abs(C.SCENARIOS['S2']['jump_value'])            # величина скачка из сценария S2
    cases, k = [], 0
    for p in range(C.N_PARAMS):
        for kind, val in (('trend', trend), ('jump', jump)):   # порядок важен: чётный случай - тренд, нечётный - скачок
            sign = 1 if (p + (kind == 'jump')) % 2 == 0 else -1
            cases.append((custom_profile(p, kind, sign * val, C.T_START), seed0 + k))
            k += 1
    quiet = [seed0 + 5000 + i for i in range(n_quiet)]  # S4 - без возмущений, по одному зерну на эпизод
    return cases, quiet


def evaluate(xs, cases, quiet, return_fa=False):
    """
    Функция качества (меньше - лучше) сразу для всех кандидатов xs: один пакет
    "кандидаты x (16 случаев + эпизоды S4)". Шум каждого эпизода зависит только от зерна.
    return_fa=True - вернуть также долю ложных тактов каждого кандидата.
    """
    n_c, n_e = len(xs), len(cases) + len(quiet)            # число кандидатов и эпизодов на кандидата
    profiles = [p for p, _ in cases] + [get_disturbance_profile('S4', C.T_START, 1.0)] * len(quiet)
    noise = np.stack([episode_noise(s) for s in [s for _, s in cases] + list(quiet)])
    params = [decode(x) for x in xs]
    kp, ki, kd, li = (np.repeat([p[i] for p in params], n_e) for i in range(4))

    batch = run_episodes_batch(profiles * n_c, BatchPIDPolicy(kp, ki, kd, li),
                               np.tile(noise, (n_c, 1, 1)), [C.T_START] * (n_c * n_e))
    costs, fas = [], []                                    # fas - доля ложных тактов по кандидатам
    for c in range(n_c):
        iae, ts, tv, fa, ss, os_ = [], [], [], [], [], []
        for e in range(n_e):
            b = c * n_e + e
            t, a, idle = compute_metrics(batch.eps[b], batch.u[b], C.T_START)
            if e < len(cases):                             # эпизоды с возмущением: качество переходного процесса
                u = batch.u[b, C.T_START:C.T_START + C.T_OBS]
                iae.append(a)
                ts.append(t)
                tv.append(np.abs(np.diff(u, axis=0)).sum(axis=1).mean())
                if e % 2 == 0:                             # случаи идут парами (тренд, скачок): чётные - тренд, как S1
                    n_tail = batch.N[b, C.T_START + C.T_OBS - SS_TAIL:C.T_START + C.T_OBS]   # последние SS_TAIL тактов
                    e_zone = n_tail @ plant.V.T - C.G      # ошибка каждой зоны на этих тактах
                    ss.append(np.abs(e_zone.mean(axis=0)).max())   # статическая ошибка = среднее за хвост (шум усредняется)
                else:                                      # нечётные - скачок на показатель e // 2
                    e_jump = (batch.N[b, C.T_START:C.T_START + C.T_OBS] @ plant.V.T - C.G)[:, e // 2]   # ошибка зоны скачка
                    sign = np.sign(e_jump[1:6].mean())     # знак начального отклонения (первые такты после скачка)
                    os_.append(max(0.0, float((-sign * e_jump[OS_FROM:]).max())))   # выброс в противоположную сторону
            else:                                          # тихие эпизоды S4: только ложные срабатывания
                fa.append(idle)
        fa_mean = float(np.mean(fa))                       # средняя доля ложных тактов кандидата
        costs.append(float(np.mean(iae)                                      # качество: интегральная абсолютная ошибка
                           + W_TS * np.mean(ts) / C.T_OBS                   # время переходного процесса
                           + W_TV * np.mean(tv)                             # "дёрганье" управления
                           + W_SS * np.mean(ss)                             # статическая ошибка при тренде (как S1)
                           + W_OS * np.mean(os_)                            # перерегулирование на скачке (как S2, S3)
                           + W_FA * fa_mean                                 # ложные срабатывания: линейный штраф
                           + W_FA_HARD * max(0.0, fa_mean - FA_LIMIT)))     # и жёсткий сверх допустимого предела
        fas.append(fa_mean)
    return (costs, fas) if return_fa else costs


def decode(u):
    v = LO + np.clip(u, 0, 1) * (HI - LO)
    return 10 ** v[0], 10 ** v[1], 10 ** v[2], int(round(v[3]))


def encode(kp, ki, kd, li):
    return (np.array([np.log10(kp), np.log10(ki), np.log10(kd), li]) - LO) / (HI - LO)


def main():
    import cma

    cases, quiet = build_cases()
    val_cases, val_quiet = build_cases(SEED_PID_VAL, N_QUIET_VAL)      # отложенная выборка для проверки
    cost = lambda x: evaluate([x], cases, quiet)[0]

    x0 = encode(*MANUAL)
    base = cost(x0)
    print(f"Ручные значения {MANUAL}: cost = {base:.4f}")

    es = cma.CMAEvolutionStrategy(x0, 0.25, {'bounds': [0, 1], 'popsize': 12, 'maxiter': 60,
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

    # проверка на зёрнах, которых тюнер не видел: и качество, и ложные срабатывания
    (val_base, val_best), (fa_base, fa_best) = evaluate([x0, best], val_cases, val_quiet, return_fa=True)
    print(f"Проверка на отложенной выборке: cost {val_best:.4f} (ручные {val_base:.4f}), "
          f"ложные срабатывания {100 * fa_best:.2f} % (ручные {100 * fa_base:.2f} %, допустимо {100 * FA_LIMIT:.1f} %)")

    if val_best < val_base and fa_best <= FA_LIMIT:        # сохраняем, только если лучше ручных И в пределах FA
        os.makedirs('tuned', exist_ok=True)
        with open(os.path.join('tuned', PID_FILE), 'w', encoding='utf-8') as f:
            json.dump({'KP': round(kp, 4), 'KI': round(ki, 4), 'KD': round(kd, 4),
                       'PID_WINDOW': li, 'cost': best_cost, 'cost_manual': base,
                       'cost_val': val_best, 'cost_val_manual': val_base, 'fa_val': fa_best}, f, indent=2)
        print(f"Сохранено в tuned/{PID_FILE}")
    elif fa_best > FA_LIMIT:
        print("Ложных срабатываний больше допустимого - файл не записан.")
    else:
        print("На отложенной выборке улучшения относительно ручных значений нет, файл не записан.")


if __name__ == '__main__':
    main()
"""
Сценарии особых причин изменчивости (п. 3.2.1).
"""

import numpy as np
from config import A_CUR, CRITICAL_SCENARIOS, N_PARAMS, T_TOTAL, T_START, SCENARIOS
from plant import plant


def get_disturbance_profile(scenario, t0=T_START, scale=1.0):
    """
    Формирование профиля возмущений для заданного сценария.
    """
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    # S4 - контрольный сценарий без особой причины
    if scenario == 'S4':
        return rate * plant.S, jump * plant.S

    config = SCENARIOS[scenario]

    if config['rate_param'] is not None:
        rate[t0:, config['rate_param']] = config['rate_value'] * scale

    if config['jump_param'] is not None:
        jump[config['jump_param']] = config['jump_value'] * scale

    return rate * plant.S, jump * plant.S


def custom_profile(param, kind, value, t0=T_START):
    """
    Профиль возмущения на произвольный показатель (для обучающей выборки).
    kind: 'trend' - линейный тренд, 'jump' - скачок.
    """
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    if kind == 'trend':
        rate[t0:, param] = value
    else:
        jump[param] = value

    return rate * plant.S, jump * plant.S


def critical_profile(name):
    """
    Профиль возмущений критической ситуации тестовой выборки (CRITICAL_SCENARIOS в config.py).
    Возвращает (rate, jump) в том же формате, что get_disturbance_profile и custom_profile:
    rate - добавка к состоянию на каждом такте, jump - скачок на такте t0.
    Амплитуды в конфиге заданы в долях размаха S_i, здесь пересчитываются умножением на plant.S.
    """
    spec = CRITICAL_SCENARIOS[name]
    t0 = spec['t0']
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    for comp in spec['components']:
        kind, p, value = comp['kind'], comp['param'], comp['value']
        start = t0 + comp.get('start', 0)
        assert start < T_TOTAL, f'{name}: компонента начинается за пределами эпизода'

        if kind == 'jump':
            if start == t0:
                jump[p] += value                    # как скачок обучающей выборки
            else:
                rate[start, p] += value             # скачок позже t0 (каскад)
        elif kind == 'trend':
            rate[start:, p] += value
        elif kind == 'pulse':
            end = start + comp['duration']
            assert end < T_TOTAL, f'{name}: импульс выходит за пределы эпизода'
            rate[start, p] += value
            rate[end, p] -= value * A_CUR ** comp['duration']   # возврат с учётом инерции объекта
        elif kind == 'sine':
            period = comp['period']
            k = np.arange(T_TOTAL - start)
            # колебания состояния с амплитудой value: производная от value * sin(2*pi*k/period)
            rate[start:, p] += value * (2 * np.pi / period) * np.cos(2 * np.pi * k / period)
        else:
            raise ValueError(f'{name}: неизвестный вид компоненты {kind!r}')

    return rate * plant.S, jump * plant.S


def get_scenario_description(scenario):
    """Получение текстового описания сценария."""
    if scenario == 'S4':
        return 'Без особой причины (контрольный сценарий)'
    return SCENARIOS[scenario]['description']
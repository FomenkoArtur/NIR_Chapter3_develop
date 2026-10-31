"""
Конфигурация эксперимента.
"""

import json
import os

import numpy as np

_TUNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuned')


def _load(name):
    """Результат автонастройки (tuned/<name>) или {}, если настройка не запускалась."""
    path = os.path.join(_TUNED_DIR, name)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


# ========== ВЕСА ==========
N_PARAMS = 8
W_OPS = np.array([0.5, 0.3, 0.2])
N_GRP = np.array([3, 3, 2])

# ========== НОРМАЛИЗАЦИЯ ==========
BETA = 100.0
N_WINDOW = 200
GAMMA = 1.0
S_MIN_FRAC = 1 / 6
IQR_TO_SIGMA = 1.349
Z_LIMIT = 3.0
N_LOW = 1 / (1 + np.exp(Z_LIMIT))
N_HIGH = 1 / (1 + np.exp(-Z_LIMIT))

# ========== ЗОНЫ УПРАВЛЕНИЯ ==========
ZONE_LAYOUTS = {
    'single': [list(range(N_PARAMS))],
    'operation': [[0, 1, 2], [3, 4, 5], [6, 7]],
    'parameter': [[i] for i in range(N_PARAMS)],
}
CONTROL_LAYOUT = 'parameter'
ZONES = ZONE_LAYOUTS[CONTROL_LAYOUT]
N_ZONES = len(ZONES)
N_FEATURES = 1 + 2 * N_PARAMS

# ========== ВРЕМЕННЫЕ ПАРАМЕТРЫ ==========
WINDOW = 32
PID_WINDOW = 10
DELTA = 0.03
N_CONS = 9
T_OBS = 100
T_START = 40
N_REAL = 30
T_TOTAL = T_START + T_OBS
N_BURN_IN = N_WINDOW
assert T_START >= WINDOW
G = 0.5

# ========== ПАРАМЕТРЫ ЦИФРОВОГО ДВОЙНИКА ==========
A_CUR = 0.99
SIG_CUR = 0.0067
A_BASE = 0.90
SIG_BASE = 0.30
N_CALIB = 200
A2 = 2.66
D4 = 3.267
D3 = 0.0
C_RAW_FACTOR = 0.10

# ========== ПАРАМЕТРЫ УЧИТЕЛЯ (ПИД) ==========
# Ручные значения (KP повышен с 1.2 до 1.6: интегральная часть учителя - сумма по окну PID_WINDOW
# тактов, а не настоящий интегратор, поэтому при тренде (S1) остаётся статическая ошибка).
# Если запущен tune_pid.py, значения берутся из tuned/pid.json.
KP = 1.6
KD = 0.20
KI = 0.05
I_MAX = 0.4
_pid = _load('pid.json')
KP = _pid.get('KP', KP)
KI = _pid.get('KI', KI)
KD = _pid.get('KD', KD)
PID_WINDOW = int(_pid.get('PID_WINDOW', PID_WINDOW))

# ========== ПАРАМЕТРЫ ОБУЧЕНИЯ ==========
BATCH_SIZE = 128
MAX_EPOCHS = 800
PATIENCE = 60
VAL_SPLIT = 0.2
LR = 5e-4
WEIGHT_DECAY = 0.0
ACTIVE_W = 5.0
QUIET_BAND = 0.06     # цель учителя |u| <= QUIET_BAND - канал в норме
FA_MARGIN = 0.08      # на таких каналах штраф за |u| > FA_MARGIN
FA_W = 20.0

# Какая сеть обучается (кнопка «Обучить» и main.py):
#   'fast'   - сеть из NIR_optimized: 17 входных каналов, три свёртки 32/64/64, один проход на отсчёт;
#   'shared' - прежняя сеть Modified: одна свёртка прогоняется по каждому из 8 показателей отдельно
#              (8 строк на отсчёт, вычислений в разы больше), размеры слоёв и LR/BATCH_SIZE/WEIGHT_DECAY
#              берутся из tuned/cnn.json.
# Переключить без правки файла: переменная окружения NIR_ARCH=shared.
CNN_ARCH = os.environ.get('NIR_ARCH', 'fast').strip().lower()   # читаем переменную окружения, по умолчанию 'fast'
assert CNN_ARCH in ('fast', 'shared'), "NIR_ARCH должна быть 'fast' или 'shared'"   # защита от опечатки

# Архитектура сети: ручные значения (для 'fast' они совпадают с NIR_optimized), для 'shared' при наличии
# tuned/cnn.json - результат tune_cnn.py
MODEL_HP = {'k1': 5, 'k2': 3, 'k3': 3, 'c1': 32, 'c2': 64, 'c3': 64,
            'pool': 8, 'fc': 128, 'dropout': 0.0}
# tuned/cnn.json подбирался на GPU под сеть 'shared'; для 'fast' он не применяется, остаются LR = 5e-4,
# BATCH_SIZE = 128, WEIGHT_DECAY = 0 (как в NIR_optimized)
_cnn = _load('cnn.json') if CNN_ARCH == 'shared' else {}   # для 'fast' пустой словарь - tuned-значения не подмешиваются
MODEL_HP.update(_cnn.get('model', {}))
LR = _cnn.get('train', {}).get('LR', LR)
BATCH_SIZE = int(_cnn.get('train', {}).get('BATCH_SIZE', BATCH_SIZE))
WEIGHT_DECAY = _cnn.get('train', {}).get('WEIGHT_DECAY', WEIGHT_DECAY)
TRAIN_HP = {'LR': LR, 'BATCH_SIZE': BATCH_SIZE, 'WEIGHT_DECAY': WEIGHT_DECAY}

# ========== ПОРОГИ ==========
IDLE_THRESHOLD = 0.10

# ========== СЦЕНАРИИ ==========
SCENARIOS = {
    'S4': {'description': 'Без особой причины (контрольный сценарий)', 'rate_param': None, 'rate_value': 0.0, 'jump_param': None, 'jump_value': 0.0},
    'S1': {'description': 'Тренд x1 (износ инструмента, токарная обработка)', 'rate_param': 0, 'rate_value': 0.027, 'jump_param': None, 'jump_value': 0.0},
    'S2': {'description': 'Скачок x6 (вибрация, фрезерная обработка)', 'rate_param': None, 'rate_value': 0.0, 'jump_param': 5, 'jump_value': 1.6},
    'S3': {'description': 'Скачок x8 (недостаточное охлаждение, закалка)', 'rate_param': None, 'rate_value': 0.0, 'jump_param': 7, 'jump_value': -1.6}
}

CRITICAL_SCENARIOS = {
    'C1': {'description': 'Мощный скачок x1', 't0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 0, 'value': 5.0}]},
    'C2': {'description': 'Крутой тренд x6', 't0': 65, 'noise_scale': 1.0, 'components': [{'kind': 'trend', 'param': 5, 'value': 0.12}]},
    'C3': {'description': 'Скачок на трёх операциях', 't0': 70, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 1, 'value': 4.0}, {'kind': 'jump', 'param': 4, 'value': 4.0}, {'kind': 'jump', 'param': 6, 'value': 4.0}]},
    'C4': {'description': 'Неуправляемый дрейф', 't0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'trend', 'param': 0, 'value': 0.15}, {'kind': 'trend', 'param': 2, 'value': 0.15}, {'kind': 'trend', 'param': 3, 'value': 0.15}, {'kind': 'trend', 'param': 6, 'value': 0.15}]},
    'C5': {'description': 'Авария: скачок всех показателей', 't0': 75, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': i, 'value': 4.0} for i in range(N_PARAMS)]},
    'C6': {'description': 'Скачок и тренд (x3)', 't0': 65, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 2, 'value': 3.0}, {'kind': 'trend', 'param': 2, 'value': 0.08}]},
    'C7': {'description': 'Импульсный выброс x5 и x8', 't0': 80, 'noise_scale': 1.0, 'components': [{'kind': 'pulse', 'param': 4, 'value': 6.0, 'duration': 8}, {'kind': 'pulse', 'param': 7, 'value': 6.0, 'duration': 8}]},
    'C8': {'description': 'Вибрация: колебания x4 и x6', 't0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'sine', 'param': 3, 'value': 2.0, 'period': 12}, {'kind': 'sine', 'param': 5, 'value': 2.0, 'period': 12}]},
    'C9': {'description': 'Каскад скачков', 't0': 65, 'noise_scale': 10.0, 'components': [{'kind': 'jump', 'param': 0, 'value': 3.0, 'start': 0}, {'kind': 'jump', 'param': 4, 'value': -3.0, 'start': 12}, {'kind': 'jump', 'param': 7, 'value': 3.0, 'start': 24}]},
}

SEED = 42
SEED_TEACHER_START = 1000
SEED_EXPERIMENT_START = 7000
SEED_CRITICAL_START = 50000
SEED_FALSE_ALARM_START = 90000
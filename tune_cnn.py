"""
tune_cnn.py - ОДИН самодостаточный файл для Google Colab (GPU T4): подбор гиперпараметров CNN
(Optuna TPE + Hyperband), критерий - замкнутый контур. Внутри: параметры проекта, цифровой двойник,
ПИД-учитель, генерация выборок, сеть, обучение на GPU, оценка по замкнутому контуру (все эпизоды пакетом).

Colab:   !pip -q install optuna
         !python tune_cnn.py 100 2        # число испытаний, число воркеров (по умолчанию 100 и число ядер)
Результат: tuned/cnn.json - скачать и положить в папку tuned/ проекта.
Необязательно: загрузить tuned/cnn_study.db (продолжить прежний поиск) и pid.json (если ПИД перенастраивали).
"""
import ctypes
import platform
import re
import struct
import subprocess
from dataclasses import dataclass
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import math
import multiprocessing as mp
import sys
import time
from copy import deepcopy

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim




# ========================= HW-OPT BEGIN: оптимизатор под текущее железо =========================
GB = 1 << 30
MAX_WORKERS = 16
MAX_PER_GPU = 4
GPU_MB_PER_WORKER = 700


@dataclass
class Gpu:
    index: int
    name: str
    total_mb: int
    free_mb: int


@dataclass
class Hardware:
    cpu: str
    logical: int
    physical: int
    fast: int
    ram_total: int
    ram_free: int
    gpus: list


def _read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return None


def _run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def _ranges(text):
    out = set()
    for part in (text or '').split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def _allowed_cpus():
    if hasattr(os, 'sched_getaffinity'):
        return sorted(os.sched_getaffinity(0))
    count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    return list(range(count))


def _cgroup_cpus():
    text = _read('/sys/fs/cgroup/cpu.max')
    if text:
        quota, period = text.split()[:2]
        return None if quota == 'max' else math.ceil(int(quota) / int(period))
    quota = _read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = _read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota and period and int(quota) > 0:
        return math.ceil(int(quota) / int(period))
    return None


def _cores_windows():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GetLogicalProcessorInformationEx
    query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    size = ctypes.c_ulong(0)
    query(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not query(0, buf, ctypes.byref(size)):
        return None
    raw, offset, cores = buf.raw[:size.value], 0, []
    while offset < len(raw):
        length = struct.unpack_from('<I', raw, offset + 4)[0]
        groups = struct.unpack_from('<H', raw, offset + 30)[0]
        threads = sum(bin(struct.unpack_from('<Q', raw, offset + 32 + 16 * g)[0]).count('1')
                      for g in range(groups))
        cores.append((raw[offset + 9], threads))
        offset += length
    return cores


def _cores_linux(cpus):
    fast_set = _ranges(_read('/sys/devices/cpu_core/cpus'))
    allowed, seen, cores = set(cpus), set(), []
    for c in cpus:
        base = f'/sys/devices/system/cpu/cpu{c}/'
        siblings = frozenset(_ranges(_read(base + 'topology/thread_siblings_list')) or {c})
        if siblings in seen:
            continue
        seen.add(siblings)
        capacity = _read(base + 'cpu_capacity')
        perf = int(c in fast_set) if fast_set else int(capacity or 0)
        cores.append((perf, len(siblings & allowed)))
    return cores


def _cores_macos():
    physical = int(_run('sysctl', '-n', 'hw.physicalcpu') or 0)
    if not physical:
        return None
    logical = int(_run('sysctl', '-n', 'hw.logicalcpu') or physical)
    fast = int(_run('sysctl', '-n', 'hw.perflevel0.physicalcpu') or 0) or physical
    smt = max(1, logical // physical)
    return [(1, smt)] * fast + [(0, smt)] * (physical - fast)


def _cpu_name():
    system = platform.system()
    if system == 'Windows':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    elif system == 'Linux':
        found = re.search(r'model name\s*:\s*(.+)', _read('/proc/cpuinfo') or '')
        if found:
            return found.group(1)
    elif system == 'Darwin':
        name = _run('sysctl', '-n', 'machdep.cpu.brand_string')
        if name:
            return name
    return platform.processor() or platform.machine()


def _memory():
    system = platform.system()
    total = free = None
    if system == 'Windows':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                        ('page_total', ctypes.c_ulonglong), ('page_avail', ctypes.c_ulonglong),
                        ('virt_total', ctypes.c_ulonglong), ('virt_avail', ctypes.c_ulonglong),
                        ('ext_avail', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total, free = status.total, status.avail
    elif system == 'Linux':
        info = {}
        for line in (_read('/proc/meminfo') or '').splitlines():
            key, _, value = line.partition(':')
            if value.split():
                info[key] = int(value.split()[0]) * 1024
        total, free = info.get('MemTotal'), info.get('MemAvailable')
        limit = _read('/sys/fs/cgroup/memory.max') or _read('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if limit and limit.isdigit() and total and int(limit) < total:
            used = _read('/sys/fs/cgroup/memory.current') or _read('/sys/fs/cgroup/memory/memory.usage_in_bytes')
            total = int(limit)
            free = max(total - int(used or 0), 0)
    elif system == 'Darwin':
        total = int(_run('sysctl', '-n', 'hw.memsize') or 0) or None
        free = int(total * 0.6) if total else None
    total = total or 8 * GB
    return total, free or total // 2


def _nvidia_gpus():
    out = _run('nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
               '--format=csv,noheader,nounits')
    gpus = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4 and parts[0].isdigit() and parts[2].isdigit() and parts[3].isdigit():
            gpus[int(parts[0])] = Gpu(int(parts[0]), parts[1], int(parts[2]), int(parts[3]))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        ids = [v.strip() for v in visible.split(',') if v.strip()]
        if all(v.isdigit() for v in ids):
            return [Gpu(i, gpus[int(v)].name, gpus[int(v)].total_mb, gpus[int(v)].free_mb)
                    for i, v in enumerate(ids) if int(v) in gpus]
    return list(gpus.values())


_HARDWARE = None


def detect():
    global _HARDWARE
    if _HARDWARE is not None:
        return _HARDWARE
    cpus = _allowed_cpus()
    logical = len(cpus)
    limit = _cgroup_cpus()
    if limit:
        logical = max(1, min(logical, limit))
    cores = None
    try:
        system = platform.system()
        if system == 'Windows':
            cores = _cores_windows()
        elif system == 'Linux':
            cores = _cores_linux(cpus)
        elif system == 'Darwin':
            cores = _cores_macos()
    except Exception:
        cores = None
    if not cores:
        try:
            import psutil
            physical = psutil.cpu_count(logical=False) or logical
        except ImportError:
            physical = logical
        cores = [(0, 1)] * max(1, min(physical, logical))
    best = max(perf for perf, _ in cores)
    physical = max(1, min(len(cores), logical))
    fast = max(1, min(sum(perf == best for perf, _ in cores), physical))
    total, free = _memory()
    _HARDWARE = Hardware(_cpu_name(), logical, physical, fast, total, free, _nvidia_gpus())
    return _HARDWARE


def usable_gpus(hardware=None):
    hardware = hardware or detect()
    if os.environ.get('NIR_DEVICE', '').lower() == 'cpu' or not torch.cuda.is_available():
        return []
    if hardware.gpus:
        return hardware.gpus
    gpus = []
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory >> 20
        gpus.append(Gpu(i, torch.cuda.get_device_name(i), total, total))
    return gpus


def describe(hardware=None):
    hw = hardware or detect()
    hybrid = f', производительных {hw.fast}' if hw.fast < hw.physical else ''
    gpus = '; '.join(f'{g.name} {g.free_mb}/{g.total_mb} МБ' for g in usable_gpus(hw)) or 'нет'
    return (f'CPU: {hw.cpu}, ядер {hw.physical}{hybrid}, потоков {hw.logical}; '
            f'ОЗУ свободно {hw.ram_free / GB:.1f} из {hw.ram_total / GB:.1f} ГБ; GPU: {gpus}')


def plan_workers(n_trials, requested=None):
    hw = detect()
    slots = [g.index for rnd in range(MAX_PER_GPU) for g in usable_gpus(hw)
             if g.free_mb >= (rnd + 1) * GPU_MB_PER_WORKER]
    if slots:
        by_ram = int((hw.ram_free - GB) // (1.8 * GB))
        auto = min(len(slots), hw.logical, by_ram, MAX_WORKERS, n_trials)
    else:
        by_ram = int((hw.ram_free - GB) // GB)
        auto = min(hw.physical, by_ram, MAX_WORKERS, n_trials)
    workers = max(1, requested or auto)
    devices = [slots[i % len(slots)] for i in range(workers)] if slots else [None] * workers
    threads = 1 if slots else max(1, hw.physical // workers)
    return {'workers': workers, 'devices': devices, 'threads': threads, 'hardware': hw}
# ========================= HW-OPT END =========================
# ============================ ПАРАМЕТРЫ ПРОЕКТА (config.py) ============================
N_PARAMS, N_ZONES, N_FEATURES = 8, 8, 17
SEED, SEED_TEACHER_START, SEED_CRITICAL_START, SEED_CNN_TUNE = 42, 1000, 50000, 150000
WINDOW, T_OBS, T_START = 32, 100, 40
T_TOTAL = T_START + T_OBS
BETA, N_WINDOW, IQR_TO_SIGMA, S_MIN_FRAC = 100.0, 200, 1.349, 1 / 6
A_CUR, SIG_CUR, A_BASE, SIG_BASE, N_CALIB, A2, C_RAW_FACTOR, G = 0.99, 0.0067, 0.90, 0.30, 200, 2.66, 0.10, 0.5
DELTA, N_CONS, IDLE_THRESHOLD = 0.03, 9, 0.10
ACTIVE_W = 5.0
W_TS, W_TV, W_FA = 3.0, 0.5, 2.0                 # веса функции качества (как в tune_pid.py)
TREND, JUMP = 0.027, 1.6                         # S1: rate_value, S2: |jump_value|
MAX_EP, PATIENCE_EP = 200, 25
STORAGE_URL = 'sqlite:///' + os.path.join('tuned', 'cnn_study.db')
MANUAL = {'k1': 5, 'k2': 3, 'k3': 3, 'c1': 32, 'c2': 64, 'c3': 64, 'fc': 128,
          'pool': 8, 'dropout': 0.0, 'LR': 5e-4, 'WEIGHT_DECAY': 1e-8, 'BATCH_SIZE': 128}

# ПИД-учитель: результат tune_pid.py (tuned/pid.json); если рядом лежит pid.json - берётся он
PID = {'KP': 2.9687, 'KI': 0.2993, 'KD': 0.0405, 'PID_WINDOW': 24}
I_MAX = 0.4
for _p in ('pid.json', os.path.join('tuned', 'pid.json')):
    if os.path.exists(_p):
        with open(_p, encoding='utf-8') as f:
            PID.update({k: v for k, v in json.load(f).items() if k in PID})
        break

CRITICAL_SCENARIOS = {
    'C1': {'t0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 0, 'value': 5.0}]},
    'C2': {'t0': 65, 'noise_scale': 1.0, 'components': [{'kind': 'trend', 'param': 5, 'value': 0.12}]},
    'C3': {'t0': 70, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 1, 'value': 4.0}, {'kind': 'jump', 'param': 4, 'value': 4.0}, {'kind': 'jump', 'param': 6, 'value': 4.0}]},
    'C4': {'t0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'trend', 'param': 0, 'value': 0.15}, {'kind': 'trend', 'param': 2, 'value': 0.15}, {'kind': 'trend', 'param': 3, 'value': 0.15}, {'kind': 'trend', 'param': 6, 'value': 0.15}]},
    'C5': {'t0': 75, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': i, 'value': 4.0} for i in range(N_PARAMS)]},
    'C6': {'t0': 65, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 2, 'value': 3.0}, {'kind': 'trend', 'param': 2, 'value': 0.08}]},
    'C7': {'t0': 80, 'noise_scale': 1.0, 'components': [{'kind': 'pulse', 'param': 4, 'value': 6.0, 'duration': 8}, {'kind': 'pulse', 'param': 7, 'value': 6.0, 'duration': 8}]},
    'C8': {'t0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'sine', 'param': 3, 'value': 2.0, 'period': 12}, {'kind': 'sine', 'param': 5, 'value': 2.0, 'period': 12}]},
    'C9': {'t0': 65, 'noise_scale': 10.0, 'components': [{'kind': 'jump', 'param': 0, 'value': 3.0, 'start': 0}, {'kind': 'jump', 'param': 4, 'value': -3.0, 'start': 12}, {'kind': 'jump', 'param': 7, 'value': 3.0, 'start': 24}]},
}


# ============================ ЦИФРОВОЙ ДВОЙНИК (plant.py) ============================
def make_plant():
    """Калибровка карт Шухарта (14), веса (27); зона = показатель, поэтому B = V = единичная матрица."""
    np.random.seed(SEED)
    base = np.zeros((N_CALIB, N_PARAMS))
    for t in range(1, N_CALIB):
        base[t] = A_BASE * base[t - 1] + np.random.normal(0, SIG_BASE, N_PARAMS)
    mr_bar = np.abs(np.diff(base, axis=0)).mean(axis=0)
    x_bar = base.mean(axis=0) + BETA
    ucl, lcl = x_bar + A2 * mr_bar, x_bar - A2 * mr_bar
    S = ucl - lcl
    w = np.zeros(N_PARAMS)
    for s, n, lam in ((0, 3, 0.5), (3, 3, 0.3), (6, 2, 0.2)):
        w[s:s + n] = lam / n
    return dict(x_bar=x_bar, UCL=ucl, LCL=lcl, S=S, S_MIN=S_MIN_FRAC * S, C_RAW=C_RAW_FACTOR * S, w=w)


P = make_plant()
X_BAR, S_SCALE, W = P['x_bar'], P['S'], P['w']


def custom_profile(param, kind, value, t0=T_START):
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)
    if kind == 'trend':
        rate[t0:, param] = value
    else:
        jump[param] = value
    return rate * S_SCALE, jump * S_SCALE


def quiet_profile():
    return np.zeros((T_TOTAL, N_PARAMS)), np.zeros(N_PARAMS)


def critical_profile(name):
    spec = CRITICAL_SCENARIOS[name]
    t0 = spec['t0']
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)
    for comp in spec['components']:
        kind, p, value = comp['kind'], comp['param'], comp['value']
        start = t0 + comp.get('start', 0)
        if kind == 'jump':
            if start == t0:
                jump[p] += value
            else:
                rate[start, p] += value
        elif kind == 'trend':
            rate[start:, p] += value
        elif kind == 'pulse':
            end = start + comp['duration']
            rate[start, p] += value
            rate[end, p] -= value * A_CUR ** comp['duration']
        elif kind == 'sine':
            k = np.arange(T_TOTAL - start)
            rate[start:, p] += value * (2 * np.pi / comp['period']) * np.cos(2 * np.pi * k / comp['period'])
    return rate * S_SCALE, jump * S_SCALE


# ============================ ПАКЕТНЫЙ РАСЧЁТ ЭПИЗОДОВ (fastsim.py) ============================
class BatchNormalizer:
    """Робастная сигмоидная нормализация (10)-(12) для B эпизодов сразу (равные веса, gamma = 1)."""

    def __init__(self, batch):
        self.buf = np.full((batch, N_WINDOW, N_PARAMS), np.nan)
        self.count = 0
        self.valid = np.zeros((batch, N_PARAMS), dtype=np.int64)
        self.m = self.s = None

    def update(self, x):
        if self.count:
            admitted = ((x >= P['LCL']) & (x <= P['UCL'])) | (self.valid == 0)
        else:
            admitted = np.ones(x.shape, dtype=bool)
        row = np.where(admitted, x, np.nan)
        if self.count < N_WINDOW:
            self.buf[:, self.count] = row
            self.count += 1
        else:
            self.valid -= ~np.isnan(self.buf[:, 0])
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = row
        self.valid += ~np.isnan(row)

        n = self.valid
        srt = np.sort(self.buf[:, :self.count], axis=1)
        q25, q50, q75 = (np.take_along_axis(srt, np.maximum(r - 1, 0)[:, None, :], axis=1)[:, 0, :]
                         for r in ((n + 3) // 4, (n + 1) // 2, (3 * n + 3) // 4))
        m, s = q50, np.maximum((q75 - q25) / IQR_TO_SIGMA, P['S_MIN'])
        empty = n == 0
        if empty.any():
            m, s = np.where(empty, self.m, m), np.where(empty, self.s, s)
        self.m, self.s = m, s
        return 0.5 * (1.0 + np.tanh(0.5 * (x - m) / s))


def episode_noise(seed, noise_scale=1.0):
    np.random.seed(seed)
    return np.random.normal(0, SIG_CUR * noise_scale, (T_TOTAL - 1, N_PARAMS))


def build_features_batch(eps, n):
    """(B, W) и (B, W, 8) -> (B, 17, W)."""
    eps = np.asarray(eps, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    w = W.astype(np.float32)
    return np.concatenate([eps[:, None, :] / 0.5, ((n - 0.5) * 2.0).transpose(0, 2, 1),
                           (n * w).transpose(0, 2, 1) * 4.0], axis=1)


def window_features(eps, n):
    """Окна длины WINDOW для каждого такта (слева повторяется такт 0): -> (B, T, 17, WINDOW)."""
    eps = np.asarray(eps, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    pad = WINDOW - 1
    eps = np.concatenate([np.repeat(eps[:, :1], pad, axis=1), eps], axis=1)
    n = np.concatenate([np.repeat(n[:, :1], pad, axis=1), n], axis=1)
    w = W.astype(np.float32)[:, None]
    e_win = np.lib.stride_tricks.sliding_window_view(eps, WINDOW, axis=1)
    n_win = np.lib.stride_tricks.sliding_window_view(n, WINDOW, axis=1)
    return np.concatenate([e_win[:, :, None, :] / 0.5, (n_win - 0.5) * 2.0, (n_win * w) * 4.0], axis=2)


class PIDPolicy:
    """ПИД-учитель (зона = показатель); сумма по окну - последние PID_WINDOW тактов."""

    def reset(self, batch):
        self.n_i = int(PID['PID_WINDOW'])
        self.hist = np.zeros((batch, self.n_i, N_ZONES))
        self.e_prev = np.zeros((batch, N_ZONES))

    def act(self, eps, x_norm):
        e = x_norm - G
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = e
        i_term = np.clip(PID['KI'] * self.hist.sum(axis=1), -I_MAX, I_MAX)
        u_raw = PID['KP'] * e + PID['KD'] * (e - self.e_prev) + i_term
        u = np.clip(u_raw, -1.0, 1.0)
        saturated = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))
        self.hist[:, -1] = np.where(saturated, 0.0, e)
        self.e_prev = e
        return u


class CNNPolicy:
    """Сеть вызывается один раз на такт для всех эпизодов (на GPU); окно слева дополняется тактом 0."""

    def __init__(self, model, dev):
        self.model, self.dev = model.eval(), dev

    def reset(self, batch):
        self.t = 0
        self.eps = np.empty((batch, T_TOTAL))
        self.n = np.empty((batch, T_TOTAL, N_PARAMS))

    def act(self, eps, x_norm):
        t = self.t
        self.eps[:, t], self.n[:, t] = eps, x_norm
        self.t += 1
        idx = np.maximum(np.arange(t - WINDOW + 1, t + 1), 0)
        f = torch.from_numpy(build_features_batch(self.eps[:, idx], self.n[:, idx])).to(self.dev)
        with torch.inference_mode():
            return self.model(f).cpu().numpy().astype(float)


def run_episodes_batch(profiles, policy, noise, t0s):
    """Возвращает (eps, u, N): (B, T), (B, T, 8), (B, T, 8). Формула (34)."""
    batch = len(profiles)
    rates = np.stack([p[0] for p in profiles])
    jumps = np.stack([p[1] for p in profiles])
    t0s = np.asarray(t0s)
    state = np.tile(X_BAR, (batch, 1))
    normalizer = BatchNormalizer(batch)
    policy.reset(batch)
    u = np.zeros((batch, N_ZONES))
    N = np.empty((batch, T_TOTAL, N_PARAMS))
    E = np.empty((batch, T_TOTAL))
    U = np.empty((batch, T_TOTAL, N_ZONES))
    for t in range(T_TOTAL):
        hit = t0s == t
        if hit.any():
            state = state + np.where(hit[:, None], jumps, 0.0)
        if t > 0:
            state = X_BAR + A_CUR * (state - X_BAR) + rates[:, t] - P['C_RAW'] * u + noise[:, t - 1]
        x_norm = normalizer.update(state)
        eps = G - x_norm @ W
        u = policy.act(eps, x_norm)
        N[:, t], E[:, t], U[:, t] = x_norm, eps, u
    return E, U, N


def training_profiles():
    profiles = [('S4', None)] * 4
    for param in range(N_PARAMS):
        for kind, value in (('trend', TREND), ('trend', -TREND), ('jump', JUMP), ('jump', -JUMP)):
            profiles.append((kind, (param, kind, value)))
    for p1, p2 in [(0, 4), (1, 5), (2, 7), (3, 6)]:
        for sign in (1, -1):
            profiles.append(('jump2', ((p1, p2), 'jump2', sign * JUMP * 0.8)))
    return profiles


def _flatten(E, N, U, t_first):
    X = window_features(E, N)[:, t_first:]
    return X.reshape(-1, *X.shape[2:]), U[:, t_first:].reshape(-1, U.shape[-1])


def build_datasets():
    """Обучающая выборка (ПИД-учитель, 44 профиля) и тестовая (9 критических ситуаций)."""
    np.random.seed(SEED)
    profiles, t0s, noises = [], [], []
    for ep, (name, spec) in enumerate(training_profiles(), start=1):
        t0 = np.random.randint(WINDOW + 5, WINDOW + 20)
        scale = np.random.uniform(0.7, 1.3)
        if spec is None:
            profile = quiet_profile()
        else:
            param, kind, value = spec
            if kind == 'jump2':
                r1, j1 = custom_profile(param[0], 'jump', value * scale, t0)
                r2, j2 = custom_profile(param[1], 'jump', value * scale, t0)
                profile = (r1 + r2, j1 + j2)
            else:
                profile = custom_profile(param, kind, value * scale, t0)
        profiles.append(profile)
        t0s.append(t0)
        noises.append(episode_noise(SEED_TEACHER_START + ep))
    E, U, N = run_episodes_batch(profiles, PIDPolicy(), np.stack(noises), t0s)
    X_train, Y_train = _flatten(E, N, U, 0)

    profiles, t0s, noises = [], [], []
    for ep, (name, spec) in enumerate(CRITICAL_SCENARIOS.items(), start=1):
        profiles.append(critical_profile(name))
        t0s.append(spec['t0'])
        noises.append(episode_noise(SEED_CRITICAL_START + ep, spec.get('noise_scale', 1.0)))
    E, U, N = run_episodes_batch(profiles, PIDPolicy(), np.stack(noises), t0s)
    X_test, Y_test = _flatten(E, N, U, 1)
    groups = np.repeat(list(CRITICAL_SCENARIOS), T_TOTAL - 1)

    keys = {row.tobytes() for row in X_train.astype(np.float32)}
    assert not any(row.tobytes() in keys for row in X_test.astype(np.float32)), "Пересечение окон train/test"
    return [torch.tensor(a, dtype=torch.float32) for a in (X_train, Y_train, X_test, Y_test)] + [groups]


# ============================ СЕТЬ И ОБУЧЕНИЕ (model.py, trainer.py) ============================
class CNNController(nn.Module):
    def __init__(self, hp):
        super().__init__()
        ks = (hp['k1'], hp['k2'], hp['k3'])
        ch = (N_FEATURES, hp['c1'], hp['c2'], hp['c3'])
        self.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], kernel_size=ks[i], padding=ks[i] // 2) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(ch[i + 1]) for i in range(3)])
        self.pool = nn.AdaptiveAvgPool1d(hp['pool'])
        self.fc1 = nn.Linear(hp['c3'] * hp['pool'], hp['fc'])
        self.drop = nn.Dropout(hp['dropout'])
        self.fc2 = nn.Linear(hp['fc'], N_ZONES)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(bn(conv(x)))
        x = self.drop(self.relu(self.fc1(self.pool(x).flatten(1))))
        return self.tanh(self.fc2(x))


def weighted_mse(pred, target):
    err = (pred - target) ** 2
    active = (target.abs().max(dim=1, keepdim=True).values > 0.1).float()
    return (err * (1.0 + ACTIVE_W * active)).mean()


def train_model(model_hp, train_hp, data, dev, callback):
    """Обучение на GPU (данные уже в VRAM). Возвращает модель на dev с лучшими весами."""
    Xt, Yt, Xv, Yv = data
    n, bs = len(Xt), int(train_hp['BATCH_SIZE'])
    model = CNNController(model_hp).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=train_hp['LR'], weight_decay=train_hp['WEIGHT_DECAY'],
                           fused=(dev.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    best, bad, best_state = float('inf'), 0, None

    for epoch in range(MAX_EP):
        model.train()
        perm = torch.randperm(n, device=dev)
        Xp, Yp = Xt[perm], Yt[perm]                    # одна перестановка за эпоху вместо 2 выборок на батч
        for i in range(0, n, bs):
            xb, yb = Xp[i:i + bs], Yp[i:i + bs]
            if len(xb) < 2:                            # BatchNorm не работает на батче из 1 элемента
                continue
            optimizer.zero_grad(set_to_none=True)
            weighted_mse(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = weighted_mse(model(Xv), Yv).item()
        scheduler.step(val_loss)
        callback(epoch, val_loss)
        if val_loss < best - 1e-6:
            best, bad, best_state = val_loss, 0, deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE_EP:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


# ============================ КРИТЕРИЙ: ЗАМКНУТЫЙ КОНТУР (tune_pid.evaluate) ============================
def build_cases(seed0=SEED_CNN_TUNE):
    """Тренд и скачок на каждый из 8 показателей (знак чередуется) + 3 эпизода без возмущений (S4)."""
    cases, k = [], 0
    for p in range(N_PARAMS):
        for kind, val in (('trend', TREND), ('jump', JUMP)):
            sign = 1 if (p + (kind == 'jump')) % 2 == 0 else -1
            cases.append((custom_profile(p, kind, sign * val, T_START), seed0 + k))
            k += 1
    return cases, [seed0 + 5000 + i for i in range(3)]


def evaluate(model, dev, cases, quiet):
    profiles = [p for p, _ in cases] + [quiet_profile()] * len(quiet)
    noise = np.stack([episode_noise(s) for s in [s for _, s in cases] + list(quiet)])
    E, U, _ = run_episodes_batch(profiles, CNNPolicy(model, dev), noise, [T_START] * len(profiles))
    iae, ts, tv, fa = [], [], [], []
    for b in range(len(profiles)):
        eps, u = E[b, T_START:T_START + T_OBS], U[b, T_START:T_START + T_OBS]
        if b < len(cases):
            iae.append(np.abs(eps).sum())
            t_s = T_OBS
            for t in range(len(eps) - N_CONS + 1):
                if np.all(np.abs(eps[t:t + N_CONS]) <= DELTA):
                    t_s = t
                    break
            ts.append(t_s)
            tv.append(np.abs(np.diff(u, axis=0)).sum(axis=1).mean())
        else:
            fa.append(np.mean(np.abs(u).max(axis=1) > IDLE_THRESHOLD))
    return float(np.mean(iae) + W_TS * np.mean(ts) / T_OBS + W_TV * np.mean(tv) + W_FA * np.mean(fa))


# ============================ OPTUNA ============================
def make_storage():
    return optuna.storages.RDBStorage(STORAGE_URL, engine_kwargs={'connect_args': {'timeout': 120}})


def make_pruner():
    return optuna.pruners.HyperbandPruner(min_resource=10, max_resource=MAX_EP, reduction_factor=3)


def split(p):
    return ({k: p[k] for k in ('k1', 'k2', 'k3', 'c1', 'c2', 'c3', 'fc', 'pool', 'dropout')},
            {k: p[k] for k in ('LR', 'WEIGHT_DECAY', 'BATCH_SIZE')})


def worker(wid, n_trials, gpu_index, threads):                     # HW-OPT
    torch.set_num_threads(threads)                                 # HW-OPT
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if gpu_index is None:                                          # HW-OPT
        dev = torch.device('cpu')                                  # HW-OPT
    else:                                                          # HW-OPT
        torch.cuda.set_device(gpu_index)                           # HW-OPT
        dev = torch.device('cuda', gpu_index)                      # HW-OPT
        torch.backends.cudnn.benchmark = True                      # HW-OPT
    tag = f"[w{wid}]"
    X_train, Y_train, X_test, Y_test, _ = build_datasets()
    data = tuple(a.to(dev) for a in (X_train, Y_train, X_test, Y_test))      # в VRAM один раз на воркер
    cases, quiet = build_cases()

    def objective(trial):
        p = {
            'k1': trial.suggest_categorical('k1', [3, 5, 7]),
            'k2': trial.suggest_categorical('k2', [3, 5, 7]),
            'k3': trial.suggest_categorical('k3', [3, 5, 7]),
            'c1': trial.suggest_categorical('c1', [16, 32, 64]),
            'c2': trial.suggest_categorical('c2', [32, 64, 128]),
            'c3': trial.suggest_categorical('c3', [32, 64, 128]),
            'fc': trial.suggest_categorical('fc', [64, 128, 256]),
            'pool': trial.suggest_categorical('pool', [4, 8, 16]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'LR': trial.suggest_float('LR', 1e-4, 3e-3, log=True),
            'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', 1e-8, 1e-2, log=True),
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [64, 128, 256]),
        }
        model_hp, train_hp = split(p)
        trial.set_user_attr('receptive_field', p['k1'] + p['k2'] + p['k3'] - 2)
        t_trial = time.time()
        print(f"{tag} trial {trial.number} старт: {p}", flush=True)

        def cb(epoch, val_loss):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"{tag} trial {trial.number} ОТСЕЧЁН на эпохе {epoch} (t={time.time() - t_trial:.1f} c)", flush=True)
                raise optuna.TrialPruned()

        torch.manual_seed(SEED)
        model = train_model(model_hp, train_hp, data, dev, cb)
        t_train = time.time() - t_trial
        cost = evaluate(model, dev, cases, quiet)
        cost = cost if np.isfinite(cost) else 1e3
        print(f"{tag} trial {trial.number} финиш: cost={cost:.4f}, обучение {t_train:.1f} c, "
              f"всего {time.time() - t_trial:.1f} c", flush=True)
        return cost

    study = optuna.load_study(study_name='cnn', storage=make_storage(),
                              sampler=optuna.samplers.TPESampler(seed=SEED + wid, constant_liar=True),
                              pruner=make_pruner())
    study.optimize(objective, n_trials=n_trials)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    requested = int(sys.argv[2]) if len(sys.argv) > 2 else None     # HW-OPT
    plan = plan_workers(n_trials, requested)                        # HW-OPT
    n_workers = plan['workers']                                     # HW-OPT
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs('tuned', exist_ok=True)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED, constant_liar=True),
                                pruner=make_pruner(), storage=make_storage(), study_name='cnn', load_if_exists=True)
    print(f"[tune_cnn] Железо: {describe(plan['hardware'])}", flush=True)   # HW-OPT
    print(f"[tune_cnn] Study 'cnn': уже есть испытаний {len(study.trials)}; новых {n_trials}, воркеров {n_workers} "
          f"(устройства {plan['devices']}, потоков на воркер {plan['threads']}); ПИД: {PID}", flush=True)
    if len(study.trials) == 0:
        study.enqueue_trial(MANUAL)                  # ручная конфигурация - первое испытание

    ctx = mp.get_context('spawn')                    # spawn обязателен при работе с CUDA
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(i, math.ceil(n_trials / n_workers),
                                              plan['devices'][i], plan['threads']))   # HW-OPT
             for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"[tune_cnn] Оптимизация завершена за {time.time() - t0:.1f} c", flush=True)

    study = optuna.load_study(study_name='cnn', storage=make_storage())
    best = study.best_trial
    print(f"Ручная конфигурация: cost = {study.trials[0].value}")
    print(f"Лучшая: cost = {best.value:.4f}, рецептивное поле = {best.user_attrs['receptive_field']}")
    print(best.params)
    model_hp, train_hp = split(best.params)
    with open(os.path.join('tuned', 'cnn.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': model_hp, 'train': train_hp, 'cost': best.value,
                   'receptive_field': best.user_attrs['receptive_field']}, f, indent=2)
    print("Сохранено в tuned/cnn.json")


if __name__ == '__main__':
    main()
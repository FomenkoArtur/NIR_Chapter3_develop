"""
tune_cnn.py - ОДИН самодостаточный файл для Google Colab (GPU T4): подбор гиперпараметров CNN
(Optuna TPE + Hyperband), критерий - замкнутый контур. Внутри: параметры проекта, цифровой двойник,
ПИД-учитель, генерация выборок, сеть, обучение на GPU, оценка по замкнутому контуру. Расчёт эпизодов
(объект, нормализация, учитель, сеть) и генерация выборок целиком на GPU, без пересылок по тактам.

Colab:   !pip -q install optuna
         !python tune_cnn.py 100 2        # число испытаний, число воркеров (по умолчанию 100 и число ядер)
Результат: tuned/cnn.json - скачать и положить в папку tuned/ проекта.
Необязательно: загрузить tuned/cnn_study_shared.db (продолжить прежний поиск этой же архитектуры) и pid.json
(если ПИД перенастраивали). Старый cnn_study.db от прежней сети не загружать.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

# ============================ ПАРАМЕТРЫ ПРОЕКТА (config.py) ============================
N_PARAMS, N_ZONES, N_FEATURES = 8, 8, 17
SEED, SEED_TEACHER_START, SEED_CRITICAL_START, SEED_CNN_TUNE = 42, 1000, 50000, 150000
WINDOW, T_OBS, T_START = 32, 100, 40
T_TOTAL = T_START + T_OBS
BETA, N_WINDOW, IQR_TO_SIGMA, S_MIN_FRAC = 100.0, 200, 1.349, 1 / 6
A_CUR, SIG_CUR, A_BASE, SIG_BASE, N_CALIB, A2, C_RAW_FACTOR, G = 0.99, 0.0067, 0.90, 0.30, 200, 2.66, 0.10, 0.5
DELTA, N_CONS, IDLE_THRESHOLD = 0.03, 9, 0.10
ACTIVE_W, QUIET_BAND, FA_MARGIN, FA_W = 5.0, 0.06, 0.08, 20.0
W_TS, W_TV, W_FA = 3.0, 0.5, 2.0                 # веса функции качества (как в tune_pid.py)
TREND, JUMP = 0.027, 1.6                         # S1: rate_value, S2: |jump_value|
MAX_EP, PATIENCE_EP = 200, 25
STORAGE_URL = 'sqlite:///' + os.path.join('tuned', 'cnn_study_shared.db')
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


# ============================ ПАКЕТНЫЙ РАСЧЁТ ЭПИЗОДОВ НА GPU (fastsim.py) ============================
DT = torch.float64                               # объект и нормализация считаются в double, как раньше в numpy


def episode_noise(seed, noise_scale=1.0):
    np.random.seed(seed)
    return np.random.normal(0, SIG_CUR * noise_scale, (T_TOTAL - 1, N_PARAMS))


class PIDPolicy:
    """ПИД-учитель (зона = показатель) на устройстве; сумма по окну - последние PID_WINDOW тактов (кольцевой буфер)."""

    def reset(self, batch, sim):
        self.n_i = int(PID['PID_WINDOW'])
        self.hist = torch.zeros((batch, self.n_i, N_ZONES), dtype=DT, device=sim.dev)
        self.e_prev = torch.zeros((batch, N_ZONES), dtype=DT, device=sim.dev)

    def act(self, t, eps, x_norm):
        e = x_norm - G
        p = t % self.n_i
        self.hist[:, p] = e
        i_term = torch.clamp(PID['KI'] * self.hist.sum(dim=1), -I_MAX, I_MAX)
        u_raw = PID['KP'] * e + PID['KD'] * (e - self.e_prev) + i_term
        u = torch.clamp(u_raw, -1.0, 1.0)
        saturated = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))
        self.hist[:, p] = torch.where(saturated, torch.zeros_like(e), e)       # антиприращение интеграла
        self.e_prev = e
        return u


class CNNPolicy:
    """Сеть вызывается один раз на такт для всех эпизодов; окна и признаки собираются на устройстве."""

    def __init__(self, model):
        self.model = model.eval()

    def reset(self, batch, sim):
        self.eps = torch.empty((batch, T_TOTAL), dtype=DT, device=sim.dev)
        self.n = torch.empty((batch, T_TOTAL, N_PARAMS), dtype=DT, device=sim.dev)
        self.w32 = sim.w.float()
        self.offsets = torch.arange(WINDOW, device=sim.dev) - (WINDOW - 1)

    def act(self, t, eps, x_norm):
        self.eps[:, t], self.n[:, t] = eps, x_norm
        idx = torch.clamp(self.offsets + t, min=0)                      # окно слева дополняется тактом 0
        e, n = self.eps[:, idx].float(), self.n[:, idx].float()
        f = torch.cat([e[:, None, :] / 0.5, ((n - 0.5) * 2.0).transpose(1, 2),
                       (n * self.w32).transpose(1, 2) * 4.0], dim=1)    # (B, 17, W)
        return self.model(f).double()


class Sim:
    """Расчёт B независимых эпизодов целиком на устройстве: объект (34), нормализация (10)-(12), регулятор."""

    def __init__(self, dev):
        self.dev = dev
        as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=dev)
        self.x_bar, self.lcl, self.ucl = as_t(X_BAR), as_t(P['LCL']), as_t(P['UCL'])
        self.s_min, self.c_raw, self.w = as_t(P['S_MIN']), as_t(P['C_RAW']), as_t(W)

    @torch.no_grad()
    def run(self, profiles, policy, noise, t0s):
        """Возвращает (E, U, N) на устройстве: (B, T), (B, T, 8), (B, T, 8). noise - (B, T - 1, 8) из episode_noise."""
        dev, batch = self.dev, len(profiles)
        rates = torch.as_tensor(np.stack([p[0] for p in profiles]), dtype=DT, device=dev)    # (B, T, 8)
        jumps = torch.as_tensor(np.stack([p[1] for p in profiles]), dtype=DT, device=dev)    # (B, 8)
        noise = torch.as_tensor(noise, dtype=DT, device=dev)
        t0s = torch.as_tensor(np.asarray(t0s), device=dev)
        no_jump, nan8 = torch.zeros_like(jumps), torch.full_like(jumps, float('nan'))

        policy.reset(batch, self)
        buf = torch.full((batch, N_PARAMS, N_WINDOW), float('nan'), dtype=DT, device=dev)    # кольцевое окно L_N
        valid = torch.zeros((batch, N_PARAMS), dtype=torch.long, device=dev)
        m = s = None
        state = self.x_bar.repeat(batch, 1)
        u = torch.zeros((batch, N_ZONES), dtype=DT, device=dev)
        N = torch.empty((batch, T_TOTAL, N_PARAMS), dtype=DT, device=dev)
        E = torch.empty((batch, T_TOTAL), dtype=DT, device=dev)
        U = torch.empty((batch, T_TOTAL, N_ZONES), dtype=DT, device=dev)

        for t in range(T_TOTAL):
            state = state + torch.where((t0s == t)[:, None], jumps, no_jump)
            if t > 0:
                state = self.x_bar + A_CUR * (state - self.x_bar) + rates[:, t] - self.c_raw * u + noise[:, t - 1]

            # нормализация: в окно попадают только значения внутри границ карты (либо окно ещё пусто)
            admitted = ((state >= self.lcl) & (state <= self.ucl)) | (valid == 0)
            row = torch.where(admitted, state, nan8)
            pos = t % N_WINDOW
            valid = valid - (~torch.isnan(buf[:, :, pos])).long()
            buf[:, :, pos] = row
            valid = valid + (~torch.isnan(row)).long()
            srt = torch.sort(buf, dim=2).values                          # NaN уходят в конец
            q = lambda r: torch.gather(srt, 2, torch.clamp(r - 1, min=0).unsqueeze(2)).squeeze(2)
            q25, q50, q75 = q((valid + 3) // 4), q((valid + 1) // 2), q((3 * valid + 3) // 4)
            m_new, s_new = q50, torch.maximum((q75 - q25) / IQR_TO_SIGMA, self.s_min)
            if m is None:
                m, s = m_new, s_new
            else:
                empty = valid == 0
                m, s = torch.where(empty, m, m_new), torch.where(empty, s, s_new)
            x_norm = 0.5 * (1.0 + torch.tanh(0.5 * (state - m) / s))

            eps = G - x_norm @ self.w
            u = policy.act(t, eps, x_norm)
            N[:, t], E[:, t], U[:, t] = x_norm, eps, u
        return E, U, N

    def windows(self, E, N):
        """Окна длины WINDOW для каждого такта (слева повторяется такт 0): -> (B, T, 17, WINDOW), float32."""
        e, n = E.float(), N.float()
        pad = WINDOW - 1
        e = torch.cat([e[:, :1].expand(-1, pad), e], dim=1)
        n = torch.cat([n[:, :1].expand(-1, pad, -1), n], dim=1)
        e_win, n_win = e.unfold(1, WINDOW, 1), n.unfold(1, WINDOW, 1)   # (B, T, W), (B, T, 8, W)
        w = self.w.float().view(N_PARAMS, 1)
        return torch.cat([e_win[:, :, None, :] / 0.5, (n_win - 0.5) * 2.0, (n_win * w) * 4.0], dim=2)


def training_profiles():
    profiles = [('S4', None)] * 4
    for param in range(N_PARAMS):
        for kind, value in (('trend', TREND), ('trend', -TREND), ('jump', JUMP), ('jump', -JUMP)):
            profiles.append((kind, (param, kind, value)))
    for p1, p2 in [(0, 4), (1, 5), (2, 7), (3, 6)]:
        for sign in (1, -1):
            profiles.append(('jump2', ((p1, p2), 'jump2', sign * JUMP * 0.8)))
    return profiles


def _flatten(sim, E, N, U, t_first):
    X = sim.windows(E, N)[:, t_first:]
    return X.reshape(-1, *X.shape[2:]), U[:, t_first:].reshape(-1, U.shape[-1]).float()


def build_datasets(sim):
    """Обучающая выборка (ПИД-учитель, 44 профиля) и тестовая (9 критических ситуаций); тензоры на устройстве sim."""
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
    E, U, N = sim.run(profiles, PIDPolicy(), np.stack(noises), t0s)
    X_train, Y_train = _flatten(sim, E, N, U, 0)

    profiles, t0s, noises = [], [], []
    for ep, (name, spec) in enumerate(CRITICAL_SCENARIOS.items(), start=1):
        profiles.append(critical_profile(name))
        t0s.append(spec['t0'])
        noises.append(episode_noise(SEED_CRITICAL_START + ep, spec.get('noise_scale', 1.0)))
    E, U, N = sim.run(profiles, PIDPolicy(), np.stack(noises), t0s)
    X_test, Y_test = _flatten(sim, E, N, U, 1)
    groups = np.repeat(list(CRITICAL_SCENARIOS), T_TOTAL - 1)

    keys = {row.tobytes() for row in X_train.cpu().numpy()}
    assert not any(row.tobytes() in keys for row in X_test.cpu().numpy()), "Пересечение окон train/test"
    return [X_train, Y_train, X_test, Y_test, groups]


# ============================ СЕТЬ И ОБУЧЕНИЕ (model.py, trainer.py) ============================
class CNNController(nn.Module):
    """Один CNN с общими весами применяется к каждому показателю отдельно: u_i зависит только от окна N_i."""

    def __init__(self, hp):
        super().__init__()
        ks = (hp['k1'], hp['k2'], hp['k3'])
        ch = (1, hp['c1'], hp['c2'], hp['c3'])
        self.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], kernel_size=ks[i], padding=ks[i] // 2) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(ch[i + 1]) for i in range(3)])
        self.pool = nn.AdaptiveAvgPool1d(hp['pool'])
        self.fc1 = nn.Linear(hp['c3'] * hp['pool'], hp['fc'])
        self.drop = nn.Dropout(hp['dropout'])
        self.fc2 = nn.Linear(hp['fc'], 1)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()

    def forward(self, x):                                  # (B, 17, W) -> (B, 8)
        b, _, w = x.shape
        x = x[:, 1:1 + N_PARAMS, :].reshape(b * N_PARAMS, 1, w)   # признаки 1..8 = (N_i - 0,5) * 2
        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(bn(conv(x)))
        x = self.drop(self.relu(self.fc1(self.pool(x).flatten(1))))
        return self.tanh(self.fc2(x)).view(b, N_PARAMS)


def weighted_mse(pred, target):
    quiet = (target.abs() <= QUIET_BAND).float()               # канал в норме (поэлементно)
    err = (pred - target) ** 2 * (1.0 + ACTIVE_W * (1.0 - quiet))
    false_alarm = (torch.relu(pred.abs() - FA_MARGIN) ** 2 * quiet).mean()
    return err.mean() + FA_W * false_alarm


def train_model(model_hp, train_hp, data, dev, callback):
    """
    Обучение на GPU: один шаг (выборка батча, прямой и обратный проход, обрезка градиента, Adam) записан в
    CUDA Graph и запускается одним вызовом. Размер батча фиксирован: за эпоху берутся n // bs полных батчей
    из случайной перестановки, неполный хвост (< bs окон) отбрасывается - каждую эпоху разный.
    На CPU тот же шаг выполняется обычным образом.
    """
    Xt, Yt, Xv, Yv = data
    n, bs = len(Xt), int(train_hp['BATCH_SIZE'])
    nb = n // bs
    cuda = dev.type == 'cuda'
    model = CNNController(model_hp).to(dev)
    params = list(model.parameters())
    lr0 = float(train_hp['LR'])
    optimizer = optim.Adam(params, lr=torch.tensor(lr0, device=dev) if cuda else lr0,
                           weight_decay=train_hp['WEIGHT_DECAY'],
                           capturable=cuda, foreach=cuda or None)      # lr - тензор на GPU: меняется без перезаписи графа
    idx2d = torch.randperm(n, device=dev)[:nb * bs].view(nb, bs).clone()   # индексы батчей этой эпохи
    k = torch.zeros(1, dtype=torch.long, device=dev)                       # номер текущего батча (растёт внутри шага)

    def step():
        idx = idx2d.index_select(0, k).view(-1)
        xb, yb = Xt.index_select(0, idx), Yt.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        weighted_mse(model(xb), yb).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        k.add_(1)

    model.train()
    graph = None
    if cuda:
        snap = {name: v.clone() for name, v in model.state_dict().items()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                                      # прогрев: 3 шага до записи графа
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        with torch.no_grad():                                              # откат прогрева: веса, BN-статистика, Adam
            for name, v in model.state_dict().items():
                v.copy_(snap[name])
            for state in optimizer.state.values():
                for t in state.values():
                    t.zero_()
        run = graph.replay
    else:
        run = step

    lr_cur, sched_best, sched_bad = lr0, float('inf'), 0                   # ReduceLROnPlateau(min, 0.5, patience 15) вручную
    best, bad, best_state = float('inf'), 0, None
    for epoch in range(MAX_EP):
        idx2d.copy_(torch.randperm(n, device=dev)[:nb * bs].view(nb, bs))
        k.zero_()
        model.train()
        for _ in range(nb):
            run()
        model.eval()
        with torch.inference_mode():
            val_loss = weighted_mse(model(Xv), Yv).item()                  # единственная синхронизация за эпоху

        if val_loss < sched_best * (1 - 1e-4):
            sched_best, sched_bad = val_loss, 0
        else:
            sched_bad += 1
            if sched_bad > 15:
                lr_cur, sched_bad = lr_cur * 0.5, 0
                lr_slot = optimizer.param_groups[0]['lr']
                if torch.is_tensor(lr_slot):
                    lr_slot.fill_(lr_cur)
                else:
                    optimizer.param_groups[0]['lr'] = lr_cur

        callback(epoch, val_loss)
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_state = {name: v.clone() for name, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE_EP:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    for p in params:
        p.grad = None
    del graph
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


def evaluate(model, sim, cases, quiet):
    profiles = [p for p, _ in cases] + [quiet_profile()] * len(quiet)
    noise = np.stack([episode_noise(s) for s in [s for _, s in cases] + list(quiet)])
    E, U, _ = sim.run(profiles, CNNPolicy(model), noise, [T_START] * len(profiles))     # весь контур на GPU
    E, U = E.cpu().numpy(), U.cpu().numpy()                                              # одна пересылка на испытание
    iae, ts, tv, fa = [], [], [], []
    for b in range(len(profiles)):
        eps, u = E[b, T_START:T_START + T_OBS], U[b, T_START:T_START + T_OBS]
        if b < len(cases):
            iae.append(np.abs(eps).sum())
            settled = np.lib.stride_tricks.sliding_window_view(np.abs(eps) <= DELTA, N_CONS).all(axis=1)
            ts.append(int(settled.argmax()) if settled.any() else T_OBS)
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


def worker(wid, n_trials):
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    tag = f"[w{wid}]"
    sim = Sim(dev)
    data = tuple(build_datasets(sim)[:4])                                    # выборки сразу в VRAM, один раз на воркер
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
        cost = evaluate(model, sim, cases, quiet)
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
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(os.cpu_count() or 2, 3)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs('tuned', exist_ok=True)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED, constant_liar=True),
                                pruner=make_pruner(), storage=make_storage(), study_name='cnn', load_if_exists=True)
    print(f"[tune_cnn] Study 'cnn': уже есть испытаний {len(study.trials)}; новых {n_trials}, воркеров {n_workers}; "
          f"ПИД: {PID}", flush=True)
    if len(study.trials) == 0:
        study.enqueue_trial(MANUAL)                  # ручная конфигурация - первое испытание

    ctx = mp.get_context('spawn')                    # spawn обязателен при работе с CUDA
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(i, math.ceil(n_trials / n_workers))) for i in range(n_workers)]
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
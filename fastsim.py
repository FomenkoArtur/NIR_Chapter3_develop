"""
Расчёт сразу нескольких эпизодов одними векторными операциями (numpy).
Результат совпадает с поэпизодным run_episode из episode.py: те же зёрна ГСЧ, формулы и порядок
обновления объекта, нормализации и регулятора. Первая ось всех массивов - номер эпизода.
Используется в trainer.py (выборки), experiment.py (30 реализаций) и tune_pid.py (все кандидаты CMA-ES сразу).
"""

from dataclasses import dataclass

import numpy as np

from config import (A_CUR, CRITICAL_SCENARIOS, DELTA, G, GAMMA, I_MAX, IQR_TO_SIGMA, KD, KI, KP, N_PARAMS,
                    N_WINDOW, N_ZONES, PID_WINDOW, SCENARIOS, SEED_CRITICAL_START,
                    SEED_TEACHER_START, SIG_CUR, T_TOTAL, WINDOW)
from plant import plant
from scenarios import critical_profile, custom_profile, get_disturbance_profile
from config import (A_CUR, CRITICAL_SCENARIOS, DELTA, G, GAMMA, I_MAX, I_SEP, IQR_TO_SIGMA, KD, KI, KP, N_PARAMS,
                    N_WINDOW, N_ZONES, PID_INTEGRATOR, PID_WINDOW, SCENARIOS, SEED_CRITICAL_START,
                    SEED_TEACHER_START, SIG_CUR, T_TOTAL, WINDOW)

class BatchNormalizer:
    """Нормализация (10)-(12) сразу для B независимых эпизодов, окно (B, окно, 8)."""

    def __init__(self, batch, window=N_WINDOW, gamma=GAMMA):
        self.s_min, self.lcl, self.ucl = plant.S_MIN, plant.LCL, plant.UCL
        self.window, self.gamma = int(window), float(gamma)
        self.buf = np.full((batch, self.window, N_PARAMS), np.nan)
        self.count = 0
        self.valid = np.zeros((batch, N_PARAMS), dtype=np.int64)
        self.m = self.s = None

    def _quantiles(self):
        buf, n = self.buf[:, :self.count], self.valid
        if self.gamma == 1.0:    # равные веса: Q_p - порядковая статистика номера ceil(n * p)
            srt = np.sort(buf, axis=1)
            ranks = ((n + 3) // 4, (n + 1) // 2, (3 * n + 3) // 4)
            return tuple(np.take_along_axis(srt, np.maximum(r - 1, 0)[:, None, :], axis=1)[:, 0, :]
                         for r in ranks)
        data = buf[:, ::-1, :]
        ok = ~np.isnan(data)
        w = np.where(ok, self.gamma ** np.arange(self.count)[None, :, None], 0.0)
        total = w.sum(axis=1)
        w = w / np.where(total <= 0, 1.0, total)[:, None, :]
        values = np.where(ok, data, 0.0)
        order = np.argsort(values, axis=1, kind='stable')
        sorted_vals = np.take_along_axis(values, order, axis=1)
        cum_w = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1)
        out = []
        for p in (0.25, 0.50, 0.75):
            idx = (cum_w >= p - 1e-12).argmax(axis=1)
            out.append(np.take_along_axis(sorted_vals, idx[:, None, :], axis=1)[:, 0, :])
        return tuple(out)

    def update(self, x):
        """x - (B, 8) текущие критерии; возвращает N(t) формы (B, 8)."""
        if self.count:
            admitted = ((x >= self.lcl) & (x <= self.ucl)) | (self.valid == 0)
        else:
            admitted = np.ones(x.shape, dtype=bool)
        row = np.where(admitted, x, np.nan)
        if self.count < self.window:
            self.buf[:, self.count] = row
            self.count += 1
        else:
            self.valid -= ~np.isnan(self.buf[:, 0])
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = row
        self.valid += ~np.isnan(row)

        q25, q50, q75 = self._quantiles()
        m, s = q50, np.maximum((q75 - q25) / IQR_TO_SIGMA, self.s_min)
        empty = self.valid == 0
        if empty.any():
            m, s = np.where(empty, self.m, m), np.where(empty, self.s, s)
        self.m, self.s = m, s
        return 0.5 * (1.0 + np.tanh(0.5 * (x - m) / s))


def episode_noise(seed, noise_scale=1.0):
    np.random.seed(seed)
    return np.random.normal(0, SIG_CUR * noise_scale, (T_TOTAL - 1, N_PARAMS))


def build_features_batch(eps, n):
    """(B, W) и (B, W, 8) -> (B, 17, W), как episode.build_features."""
    eps = np.asarray(eps, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    w = plant.w.astype(np.float32)
    return np.concatenate([eps[:, None, :] / 0.5,
                           ((n - 0.5) * 2.0).transpose(0, 2, 1),
                           (n * w).transpose(0, 2, 1) * 4.0], axis=1)


def window_features(eps, n):
    """Окна длины WINDOW для каждого такта t (слева повторяется такт 0): -> (B, T, 17, WINDOW)."""
    eps = np.asarray(eps, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    pad = WINDOW - 1
    eps = np.concatenate([np.repeat(eps[:, :1], pad, axis=1), eps], axis=1)
    n = np.concatenate([np.repeat(n[:, :1], pad, axis=1), n], axis=1)
    w = plant.w.astype(np.float32)[:, None]
    e_win = np.lib.stride_tricks.sliding_window_view(eps, WINDOW, axis=1)
    n_win = np.lib.stride_tricks.sliding_window_view(n, WINDOW, axis=1)
    return np.concatenate([e_win[:, :, None, :] / 0.5, (n_win - 0.5) * 2.0, (n_win * w) * 4.0], axis=2)


@dataclass
class BatchEpisode:
    X: np.ndarray
    N: np.ndarray
    K: np.ndarray
    eps: np.ndarray
    u: np.ndarray


class BatchNullPolicy:
    def reset(self, batch):
        self.batch = batch

    def act(self, eps, x_norm):
        return np.zeros((self.batch, N_ZONES))


class BatchPIDPolicy:
    """
    ПИД-учитель для B эпизодов. kp, ki, kd, window, imax - числа (для всех) или массивы (B,)
    (свои значения у каждого эпизода: так считаются сразу все кандидаты CMA-ES).
    Без аргументов - значения из config. Сумма по окну - последние window тактов, как в PIDPolicy.
    Мёртвая зона: на канале с |e| <= DELTA учитель выдаёт u = 0 (сеть учится не реагировать в допуске).
    """

    def __init__(self, kp=None, ki=None, kd=None, window=None, imax=None):
        def col(v, d):
            v = np.asarray(d if v is None else v, dtype=float)
            return v[:, None] if v.ndim else v
        self.kp, self.ki, self.kd, self.imax = col(kp, KP), col(ki, KI), col(kd, KD), col(imax, I_MAX)
        self.n_i = np.asarray(PID_WINDOW if window is None else window, dtype=int)

    def reset(self, batch):
        wmax = int(self.n_i.max())
        n_i = np.broadcast_to(self.n_i, (batch,))
        self.keep = (np.arange(wmax)[None, :] >= wmax - n_i[:, None])[:, :, None]
        self.hist = np.zeros((batch, wmax, N_ZONES))
        self.e_prev = np.zeros((batch, N_ZONES))
        self.I = np.zeros((batch, N_ZONES))             # состояние настоящего интегратора (режим 'true')

    def act(self, eps, x_norm):
        e = x_norm @ plant.V.T - G
        if PID_INTEGRATOR == 'true':
            # настоящий интегратор; мёртвая зона только на P и D, иначе при тренде u дребезжит на её границе
            p = np.where(np.abs(e) <= DELTA, 0.0, self.kp * e + self.kd * (e - self.e_prev))
            u_raw = p + self.I                          # интегральная часть добавляется всегда
            u = np.clip(u_raw, -1.0, 1.0)               # выход ограничен диапазоном привода
            saturated = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))   # привод упёрся и ошибка того же знака
            freeze = saturated | (np.abs(e) > I_SEP)    # антивиндап и разделение интеграла: на большой ошибке не копим
            self.I = np.clip(self.I + np.where(freeze, 0.0, self.ki * e), -self.imax, self.imax)
            self.e_prev = e                             # запоминаем ошибку для D-части на следующем такте
            return u
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = e
        i_term = np.clip(self.ki * np.where(self.keep, self.hist, 0.0).sum(axis=1), -self.imax, self.imax)
        u_raw = self.kp * e + self.kd * (e - self.e_prev) + i_term
        u = np.clip(u_raw, -1.0, 1.0)
        saturated = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))
        self.hist[:, -1] = np.where(saturated, 0.0, e)
        self.e_prev = e
        return np.where(np.abs(e) <= DELTA, 0.0, u)      # мёртвая зона


class BatchCNNPolicy:
    """Сеть вызывается один раз на такт для всех эпизодов; окно слева дополняется первым значением."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def reset(self, batch):
        self.t = 0
        self.eps = np.empty((batch, T_TOTAL))
        self.n = np.empty((batch, T_TOTAL, N_PARAMS))

    def act(self, eps, x_norm):
        import torch

        t = self.t
        self.eps[:, t], self.n[:, t] = eps, x_norm
        self.t += 1
        idx = np.maximum(np.arange(t - WINDOW + 1, t + 1), 0)
        features = build_features_batch(self.eps[:, idx], self.n[:, idx])
        with torch.inference_mode():
            return self.model(torch.from_numpy(features)).numpy().astype(float)


def run_episodes_batch(profiles, policy, noise, t0s):
    """profiles - список (rate, jump), noise - (B, T_TOTAL - 1, 8), t0s - такты скачка. Формула (34)."""
    batch = len(profiles)
    rates = np.stack([p[0] for p in profiles])
    jumps = np.stack([p[1] for p in profiles])
    t0s = np.asarray(t0s)

    state = np.tile(plant.x_bar, (batch, 1))
    normalizer = BatchNormalizer(batch)
    policy.reset(batch)
    u = np.zeros((batch, N_ZONES))

    X = np.empty((batch, T_TOTAL, N_PARAMS))
    N = np.empty((batch, T_TOTAL, N_PARAMS))
    K = np.empty((batch, T_TOTAL))
    E = np.empty((batch, T_TOTAL))
    U = np.empty((batch, T_TOTAL, N_ZONES))

    for t in range(T_TOTAL):
        hit = t0s == t
        if hit.any():
            state = state + np.where(hit[:, None], jumps, 0.0)
        if t > 0:
            state = (plant.x_bar + A_CUR * (state - plant.x_bar) + rates[:, t]
                     - plant.C_RAW * (u @ plant.B.T) + noise[:, t - 1])

        x_norm = normalizer.update(state)
        k = x_norm @ plant.w
        eps = plant.G - k
        u = policy.act(eps, x_norm)

        X[:, t], N[:, t], K[:, t], E[:, t], U[:, t] = state, x_norm, k, eps, u

    return BatchEpisode(X, N, K, E, U)


def training_profiles():
    trend = SCENARIOS['S1']['rate_value']
    jump = abs(SCENARIOS['S2']['jump_value'])

    profiles = [('S4', None)] * 4
    for param in range(N_PARAMS):
        for kind, value in (('trend', trend), ('trend', -trend), ('jump', jump), ('jump', -jump)):
            profiles.append((kind, (param, kind, value)))

    for p1, p2 in [(0, 4), (1, 5), (2, 7), (3, 6)]:
        for sign in (1, -1):
            profiles.append(('jump2', ((p1, p2), 'jump2', sign * jump * 0.8)))
    return profiles


def _windows_to_arrays(batch, t_first):
    X = window_features(batch.eps, batch.N)[:, t_first:]
    Y = batch.u[:, t_first:]
    return X.reshape(-1, *X.shape[2:]), Y.reshape(-1, Y.shape[-1])


def train_arrays():
    """Обучающая выборка: ПИД-учитель на 44 профилях; t0 и scale берутся из ГСЧ в прежнем порядке."""
    profiles, t0s, noises = [], [], []

    for ep, (name, spec) in enumerate(training_profiles(), start=1):
        t0 = np.random.randint(WINDOW + 5, WINDOW + 20)
        scale = np.random.uniform(0.7, 1.3)

        if spec is None:
            profile = get_disturbance_profile('S4', t0, scale)
        else:
            param, kind, value = spec
            if kind == 'jump2':
                p1, p2 = param
                rate1, jump1 = custom_profile(p1, 'jump', value * scale, t0)
                rate2, jump2 = custom_profile(p2, 'jump', value * scale, t0)
                profile = (rate1 + rate2, jump1 + jump2)
            else:
                profile = custom_profile(param, kind, value * scale, t0)

        profiles.append(profile)
        t0s.append(t0)
        noises.append(episode_noise(SEED_TEACHER_START + ep))   # оставляет ГСЧ в том же состоянии, что run_episode

    return _windows_to_arrays(run_episodes_batch(profiles, BatchPIDPolicy(), np.stack(noises), t0s), 0)


def critical_arrays():
    """Тестовая выборка: девять критических ситуаций (без окна такта 0)."""
    profiles, t0s, noises, names = [], [], [], []

    for ep, (name, spec) in enumerate(CRITICAL_SCENARIOS.items(), start=1):
        profiles.append(critical_profile(name))
        t0s.append(spec['t0'])
        noises.append(episode_noise(SEED_CRITICAL_START + ep, spec.get('noise_scale', 1.0)))
        names.append(name)

    X, Y = _windows_to_arrays(run_episodes_batch(profiles, BatchPIDPolicy(), np.stack(noises), t0s), 1)
    return X, Y, np.repeat(names, T_TOTAL - 1)
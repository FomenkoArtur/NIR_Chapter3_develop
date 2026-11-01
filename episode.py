"""
Единый прогон эпизода: объект, политика управления, история состояний.
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import (DELTA, G, I_MAX, I_SEP, KD, KI, KP, N_PARAMS, N_ZONES, PID_INTEGRATOR, PID_WINDOW, SIG_CUR,
                    T_START, T_TOTAL, WINDOW)
from plant import plant
from scenarios import get_disturbance_profile


def build_features(eps_window, n_window):
    eps = np.asarray(eps_window, dtype=np.float32)
    n = np.asarray(n_window, dtype=np.float32)
    return np.vstack([
        eps[None, :] / 0.5,
        ((n - 0.5) * 2.0).T,
        (n * plant.w.astype(np.float32)).T * 4.0,
    ])


@dataclass
class Episode:
    X: np.ndarray
    N: np.ndarray
    K: np.ndarray
    eps: np.ndarray
    u: np.ndarray


class NullPolicy:
    def reset(self):
        pass

    def act(self, eps, x_norm):
        return np.zeros(N_ZONES)


class PIDPolicy:
    """ПИД-учитель. Без аргументов берёт значения из config (ручные или из tuned/pid.json)."""

    def __init__(self, kp=None, ki=None, kd=None, window=None, imax=None):
        self.kp = KP if kp is None else kp
        self.ki = KI if ki is None else ki
        self.kd = KD if kd is None else kd
        self.n_i = PID_WINDOW if window is None else int(window)
        self.imax = I_MAX if imax is None else imax

    def reset(self):
        self.e_prev = np.zeros(N_ZONES)
        self.buf = []
        self.I = np.zeros(N_ZONES)

    def act(self, eps, x_norm):
        e = plant.zone_criteria(x_norm) - G
        if PID_INTEGRATOR == 'true':
            # то же, что BatchPIDPolicy: мёртвая зона только на P и D, интегральная часть добавляется всегда
            p = np.where(np.abs(e) <= DELTA, 0.0, self.kp * e + self.kd * (e - self.e_prev))
            u_raw = p + self.I
            u = np.clip(u_raw, -1.0, 1.0)
            saturated = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))
            freeze = saturated | (np.abs(e) > I_SEP)  # антивиндап и разделение интеграла
            self.I = np.clip(self.I + np.where(freeze, 0.0, self.ki * e), -self.imax, self.imax)
            self.e_prev = e
            return u
        self.buf.append(e)
        if len(self.buf) > self.n_i:
            self.buf.pop(0)

        i_term = np.clip(self.ki * np.sum(self.buf, axis=0), -self.imax, self.imax)
        u_raw = self.kp * e + self.kd * (e - self.e_prev) + i_term
        u = np.clip(u_raw, -1.0, 1.0)

        saturated_high = (u_raw > 1.0) & (e > 0)
        saturated_low = (u_raw < -1.0) & (e < 0)
        if np.any(saturated_high) or np.any(saturated_low):
            self.buf[-1] = np.where(saturated_high | saturated_low, 0.0, self.buf[-1])

        self.e_prev = e
        return np.where(np.abs(e) <= DELTA, 0.0, u)      # мёртвая зона


class CNNPolicy:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def reset(self):
        self.eps = []
        self.n = []

    def act(self, eps, x_norm):
        self.eps.append(eps)
        self.n.append(x_norm)
        if len(self.eps) > WINDOW:
            self.eps.pop(0)
            self.n.pop(0)

        pad = WINDOW - len(self.eps)                      # слева дополняем первым значением
        eps_w = [self.eps[0]] * pad + self.eps
        n_w = [self.n[0]] * pad + self.n
        features = torch.from_numpy(build_features(eps_w, n_w)).unsqueeze(0)
        with torch.no_grad():
            u = self.model(features).squeeze(0).numpy().astype(float)

        return u


def run_episode(scenario, policy, seed=None, scale=1.0, t0=T_START, profile=None, noise_scale=1.0):
    if seed is not None:
        np.random.seed(seed)

    rate, jump = profile if profile is not None else get_disturbance_profile(scenario, t0, scale)
    state = plant.x_bar.copy()
    normalizer = plant.make_normalizer()
    policy.reset()
    u = np.zeros(N_ZONES)

    X, N, K, E, U = [], [], [], [], []
    for t in range(T_TOTAL):
        if t == t0:
            state = state + jump
        if t > 0:
            noise = np.random.normal(0, SIG_CUR * noise_scale, N_PARAMS)
            state = plant.evolve(state, rate[t], u, noise)

        x_norm = normalizer.update(state)
        k = plant.compute_super_criterion(x_norm)
        eps = plant.G - k
        u = policy.act(eps, x_norm)

        X.append(state.copy())
        N.append(x_norm)
        K.append(k)
        E.append(eps)
        U.append(u)

    return Episode(np.array(X), np.array(N), np.array(K), np.array(E), np.array(U))
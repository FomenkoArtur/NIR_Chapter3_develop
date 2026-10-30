from dataclasses import dataclass

import numpy as np

from config import T_START
from episode import CNNPolicy, NullPolicy, run_episode
from metrics import compute_metrics, out_of_control_rate


@dataclass
class EpisodeResult:
    X: np.ndarray
    N: np.ndarray
    K: np.ndarray
    eps: np.ndarray
    u: np.ndarray
    ts: int
    iae: float
    idle_rate: float
    oc_rate: float


def simulate(scenario, seed, scale=1.0, model=None):
    policy = CNNPolicy(model) if model is not None else NullPolicy()
    episode = run_episode(scenario, policy, seed=seed, scale=scale)
    ts, iae, idle_rate = compute_metrics(episode.eps, episode.u, T_START)
    oc_rate = out_of_control_rate(episode.N, T_START)
    return EpisodeResult(episode.X, episode.N, episode.K, episode.eps, episode.u,
                         ts, iae, idle_rate, oc_rate)

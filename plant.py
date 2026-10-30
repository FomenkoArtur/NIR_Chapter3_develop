"""
Цифровой двойник технологического процесса, калибровка контрольных карт
и робастная сигмоидная нормализация частных критериев.

Соответствие тексту РПЗ:
    (3)  x_i = z_i + beta_i             - приведение критериев к положительному виду
    (9)  omega_k = gamma^k / sum(gamma^j) - веса наблюдений скользящего окна
    (10) F_i(y, t), Q_p;i(t)            - взвешенная эмпирическая функция распределения и квантили
    (11) s_i(t) = max(IQR / 1,349; s_min)
    (12) N_i(t) = 1 / (1 + exp(-(x_i(t) - m_i(t)) / s_i(t)))
    (13) K(t) = sum(w_i * N_i)
    (14) границы карты X-MR, (27) иерархические веса, (34) модель объекта.
"""

import numpy as np
from config import (N_PARAMS, A_BASE, SIG_BASE, N_CALIB, A2, D4, D3,
                    A_CUR, SIG_CUR, C_RAW_FACTOR, G, ZONES, N_ZONES,
                    BETA, N_WINDOW, GAMMA, S_MIN_FRAC, IQR_TO_SIGMA)


class RobustSigmoidNormalizer:
    """
    Робастная сигмоидная нормализация по взвешенному скользящему окну.

    На каждом такте медиана m_i(t) и масштаб s_i(t) оцениваются по наблюдениям
    последних L_N тактов, причём наблюдению возраста k присвоен вес
    gamma^k / sum(gamma^j) (формула (9)). Квантили определяются по взвешенной
    эмпирической функции распределения (формула (10)), масштаб - по формуле (11),
    нормализованное значение - по формуле (12).

    Граница карты Шухарта фиксирована по результатам калибровки: значение x_i(t),
    вышедшее за [LCL_i; UCL_i], в окно i-го критерия не попадает. Пока критерий
    находится вне границ, оценки m_i, s_i не изменяются, поэтому устойчивое
    смещение процесса не принимается за новую норму.

    Объект хранит только окно из L_N последних тактов, поэтому повторная
    обработка накопленной выборки не требуется.
    """

    def __init__(self, s_min, lcl, ucl, window=N_WINDOW, gamma=GAMMA):
        self.s_min = np.asarray(s_min, dtype=float)
        self.lcl = np.asarray(lcl, dtype=float)
        self.ucl = np.asarray(ucl, dtype=float)
        self.window = int(window)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        """Очистка окна (начало нового эпизода)."""
        self._buf = []
        self.m = None
        self.s = None

    @staticmethod
    def _weighted_quantile(sorted_vals, cum_w, p):
        """Формула (10): Q_p = min{y : F(y) >= p} для каждого критерия (столбца)."""
        idx = (cum_w >= p - 1e-12).argmax(axis=0)
        return sorted_vals[idx, np.arange(sorted_vals.shape[1])]

    def estimates(self):
        """Текущие оценки медианы m_i(t) и масштаба s_i(t) по окну."""
        data = np.array(self._buf[::-1])                 # строка k - значение возраста k тактов
        valid = ~np.isnan(data)
        ages = np.arange(len(data))[:, None]
        w = np.where(valid, self.gamma ** ages, 0.0)     # формула (9) для принятых наблюдений
        total = w.sum(axis=0)
        empty = total <= 0
        w = w / np.where(empty, 1.0, total)

        values = np.where(valid, data, 0.0)
        order = np.argsort(values, axis=0, kind='stable')
        sorted_vals = np.take_along_axis(values, order, axis=0)
        cum_w = np.cumsum(np.take_along_axis(w, order, axis=0), axis=0)

        q25 = self._weighted_quantile(sorted_vals, cum_w, 0.25)
        q50 = self._weighted_quantile(sorted_vals, cum_w, 0.50)
        q75 = self._weighted_quantile(sorted_vals, cum_w, 0.75)

        m = q50
        s = np.maximum((q75 - q25) / IQR_TO_SIGMA, self.s_min)
        if empty.any():
            m = np.where(empty, self.m, m)
            s = np.where(empty, self.s, s)
        return m, s

    def update(self, x):
        """
        Принимает вектор текущих значений критериев x(t), обновляет окно
        и возвращает вектор N(t) по формуле (12).
        """
        x = np.asarray(x, dtype=float).copy()
        admitted = (x >= self.lcl) & (x <= self.ucl)
        if self._buf:
            admitted |= np.isnan(np.array(self._buf)).all(axis=0)
        else:
            admitted[:] = True

        self._buf.append(np.where(admitted, x, np.nan))
        if len(self._buf) > self.window:
            self._buf.pop(0)

        m, s = self.estimates()
        self.m, self.s = m, s                            # оценки текущего такта (для bounds)
        z = (x - m) / s
        # 1 / (1 + exp(-z)) в численно устойчивой форме
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def bounds(self, lcl, ucl):
        """
        Контрольные границы карты Шухарта LCL_i, UCL_i в шкале N_i на текущем такте:
        те же оконные оценки m_i(t), s_i(t) и та же робастная сигмоида (12), что и для x_i(t).
        Сигмоида монотонна, поэтому x_i внутри [LCL_i; UCL_i] тогда и только тогда,
        когда N_i внутри [N_LCL_i(t); N_UCL_i(t)]. Вызывать после update().
        """
        def sigmoid(v):
            return 0.5 * (1.0 + np.tanh(0.5 * (np.asarray(v, dtype=float) - self.m) / self.s))
        return sigmoid(lcl), sigmoid(ucl)


class Plant:
    """Цифровой двойник технологического процесса участка металлообработки."""

    def __init__(self, seed=None):
        """
        Инициализация и калибровка контрольных карт.
        """
        if seed is not None:
            np.random.seed(seed)

        self._calibrate_control_charts()
        self._compute_weights()
        self._build_zones()

        self.C_RAW = C_RAW_FACTOR * self.S
        self.G = G

        # Нижняя граница масштаба нормализации s_min (формула (11)):
        # доля S_MIN_FRAC от ширины контрольного коридора H_i = UCL_i - LCL_i
        self.S_MIN = S_MIN_FRAC * self.S

    def _calibrate_control_charts(self):
        """
        Калибровка контрольных карт Шухарта X-MR по формуле (14).
        Показатели приведены к положительному виду сдвигом beta (формула (3)):
        центральная линия и границы карты смещаются на beta, размах не меняется.
        """
        base = np.zeros((N_CALIB, N_PARAMS))
        for t in range(1, N_CALIB):
            base[t] = A_BASE * base[t-1] + np.random.normal(0, SIG_BASE, N_PARAMS)

        mr_bar = np.abs(np.diff(base, axis=0)).mean(axis=0)
        self.x_bar = base.mean(axis=0) + BETA
        self.UCL = self.x_bar + A2 * mr_bar
        self.LCL = self.x_bar - A2 * mr_bar
        self.S = self.UCL - self.LCL

    def _compute_weights(self):
        """
        Расчёт весовых коэффициентов по иерархической схеме (27): w_i = lambda_j * v_i.
        """
        from config import W_OPS, N_GRP

        self.w = np.zeros(N_PARAMS)
        for k in range(len(W_OPS)):
            s = int(N_GRP[:k].sum())
            self.w[s:s+N_GRP[k]] = W_OPS[k] / N_GRP[k]

        assert np.isclose(self.w.sum(), 1.0), f"Сумма весов {self.w.sum()} != 1.0"

    def _build_zones(self):
        """
        Матрицы адресации воздействия: B - действие зоны на показатели,
        V - внутризонные веса частных критериев.
        """
        self.B = np.zeros((N_PARAMS, N_ZONES))
        self.V = np.zeros((N_ZONES, N_PARAMS))
        for j, idx in enumerate(ZONES):
            self.B[idx, j] = 1.0
            self.V[j, idx] = self.w[idx] / self.w[idx].sum()

    def make_normalizer(self):
        """Новый нормализатор (чистое окно) для одного эпизода."""
        return RobustSigmoidNormalizer(self.S_MIN, self.LCL, self.UCL)

    def zone_criteria(self, x_norm):
        """Зональные суперкритерии K_q."""
        return self.V @ x_norm

    def compute_super_criterion(self, x_norm):
        """Критериальная свёртка по формуле (13)."""
        return float(self.w @ x_norm)

    def evolve(self, L_prev, rate, u, noise):
        """
        Эволюция состояния объекта на один такт, формула (34).
        Состояние возвращается к центру калибровки x_bar.
        Воздействие u (вектор по зонам) адресуется только показателям своей зоны.
        """
        return (self.x_bar + A_CUR * (L_prev - self.x_bar) + rate
                - self.C_RAW * (self.B @ u) + noise)


# Глобальный экземпляр
plant = Plant(seed=42)
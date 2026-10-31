import torch.nn as nn
from config import CNN_ARCH, MODEL_HP, N_FEATURES, N_PARAMS, N_ZONES


class CNNControllerShared(nn.Module):       # прежняя сеть Modified, код ниже не менялся
    def __init__(self, hp=None, in_channels=N_FEATURES, n_outputs=N_ZONES):
        super().__init__()
        assert n_outputs == N_PARAMS, "Нужен CONTROL_LAYOUT = 'parameter'"
        h = dict(MODEL_HP)
        h.update(hp or {})
        self.hp = h
        ks = (h['k1'], h['k2'], h['k3'])
        ch = (1, h['c1'], h['c2'], h['c3'])                    # на входе один канал: N_i
        self.convs = nn.ModuleList(
            [nn.Conv1d(ch[i], ch[i + 1], kernel_size=ks[i], padding=ks[i] // 2) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(ch[i + 1]) for i in range(3)])
        self.pool = nn.AdaptiveAvgPool1d(h['pool'])
        self.fc1 = nn.Linear(h['c3'] * h['pool'], h['fc'])
        self.drop = nn.Dropout(h['dropout'])
        self.fc2 = nn.Linear(h['fc'], 1)
        # ReLU на месте: после BatchNorm и после Linear входное значение для обратного прохода не нужно,
        # поэтому результат тот же, а на каждый такой слой приходится на одну большую временную матрицу меньше
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    @property
    def receptive_field(self):
        return 1 + sum(k - 1 for k in (self.hp['k1'], self.hp['k2'], self.hp['k3']))

    def forward(self, x):
        b, _, w = x.shape
        x = x[:, 1:1 + N_PARAMS, :].reshape(b * N_PARAMS, 1, w)   # признаки 1..8 = (N_i - 0,5) * 2
        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(bn(conv(x)))
        x = self.drop(self.relu(self.fc1(self.pool(x).flatten(1))))
        return self.tanh(self.fc2(x)).view(b, N_PARAMS)


class CNNControllerFast(nn.Module):
    """
    Сеть из NIR_optimized: на вход сразу все 17 каналов (eps, N_i, N_i * w_i), один проход на отсчёт.
    Размеры слоёв заданы здесь и от tuned/cnn.json не зависят.
    """

    def __init__(self, hp=None, in_channels=N_FEATURES, n_outputs=N_ZONES):
        super().__init__()
        if hp:                                                  # размеры слоёв здесь фиксированы - молча не игнорируем
            raise ValueError("CNNControllerFast не принимает hp; для настраиваемой сети: NIR_ARCH=shared")
        self.c1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)   # 17 -> 32 канала, окно 5 тактов
        self.bn1 = nn.BatchNorm1d(32)                           # нормализация по батчу после 1-й свёртки
        self.c2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)   # 32 -> 64 канала, окно 3 такта
        self.bn2 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)   # 64 -> 64 канала, окно 3 такта
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(8)                     # длина WINDOW = 32 сжимается до 8 позиций
        self.fc1 = nn.Linear(64 * 8, 128)                       # 64 канала x 8 позиций = 512 -> 128
        self.fc2 = nn.Linear(128, n_outputs)                    # 128 -> 8 управляющих воздействий
        # ReLU на месте: после BatchNorm и после Linear входное значение для обратного прохода не нужно,
        # результат тот же, а временных матриц на слой меньше (так же сделано в CNNControllerShared)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()                                   # выход u в диапазоне (-1; 1)

    def forward(self, x):                                       # x: (B, 17, WINDOW)
        x = self.relu(self.bn1(self.c1(x)))                     # (B, 32, WINDOW)
        x = self.relu(self.bn2(self.c2(x)))                     # (B, 64, WINDOW)
        x = self.relu(self.bn3(self.c3(x)))                     # (B, 64, WINDOW)
        x = self.pool(x).flatten(1)                             # (B, 512)
        x = self.relu(self.fc1(x))                              # (B, 128)
        return self.tanh(self.fc2(x))                           # (B, 8)


# CNNController - то имя, которое импортируют trainer.py и GUI; какая именно сеть за ним стоит, решает config.CNN_ARCH
CNNController = CNNControllerFast if CNN_ARCH == 'fast' else CNNControllerShared
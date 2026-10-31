import torch.nn as nn
from config import MODEL_HP, N_FEATURES, N_PARAMS, N_ZONES


class CNNController(nn.Module):
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
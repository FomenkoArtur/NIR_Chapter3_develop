"""
Архитектура свёрточной нейронной сети регулятора (п. 2.2.3).
Гиперпараметры берутся из config.MODEL_HP (ручные или из tuned/cnn.json).
"""

import torch.nn as nn
from config import MODEL_HP, N_FEATURES, N_ZONES


class CNNController(nn.Module):
    def __init__(self, hp=None, in_channels=N_FEATURES, n_outputs=N_ZONES):
        super().__init__()
        h = dict(MODEL_HP)
        h.update(hp or {})
        self.hp = h

        ks = (h['k1'], h['k2'], h['k3'])
        ch = (in_channels, h['c1'], h['c2'], h['c3'])
        self.convs = nn.ModuleList(
            [nn.Conv1d(ch[i], ch[i + 1], kernel_size=ks[i], padding=ks[i] // 2) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(ch[i + 1]) for i in range(3)])
        self.pool = nn.AdaptiveAvgPool1d(h['pool'])
        self.fc1 = nn.Linear(h['c3'] * h['pool'], h['fc'])
        self.drop = nn.Dropout(h['dropout'])
        self.fc2 = nn.Linear(h['fc'], n_outputs)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    @property
    def receptive_field(self):
        return 1 + sum(k - 1 for k in (self.hp['k1'], self.hp['k2'], self.hp['k3']))

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(bn(conv(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.tanh(self.fc2(x))
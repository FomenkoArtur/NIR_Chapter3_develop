from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from PySide6.QtCore import QThread, Signal
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, SEED
from experiment import run_full_experiment
from model import CNNController
from trainer import generate_training_data, weighted_mse


class TrainWorker(QThread):
    epoch_done = Signal(int, float, float)
    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            self.message.emit("Формирование обучающей выборки...")
            X_train, Y_train, X_val, Y_val = generate_training_data()
            self.message.emit(
                f"Объём выборки: обучение {len(X_train)}, проверка {len(X_val)}"
            )

            loader = DataLoader(
                TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
            )
            model = CNNController()
            optimizer = optim.Adam(model.parameters(), lr=LR) if LR else optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15
            )

            best_loss = float('inf')
            best_state = None
            bad_epochs = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                if self._stop_requested:
                    self.message.emit("Обучение остановлено пользователем.")
                    break

                model.train()
                total = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = weighted_mse(model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += loss.item() * len(xb)
                train_loss = total / len(X_train)

                model.eval()
                with torch.no_grad():
                    val_loss = weighted_mse(model(X_val), Y_val).item()
                scheduler.step(val_loss)

                self.epoch_done.emit(epoch, train_loss, val_loss)

                if val_loss < best_loss - 1e-6:
                    best_loss = val_loss
                    bad_epochs = 0
                    best_state = deepcopy(model.state_dict())
                else:
                    bad_epochs += 1
                    if bad_epochs >= PATIENCE:
                        self.message.emit(
                            f"Ранний останов на эпохе {epoch}, val MSE = {best_loss:.6f}"
                        )
                        break

            if best_state is None:
                self.failed.emit("Обучение прервано до завершения первой эпохи.")
                return

            model.load_state_dict(best_state)
            model.eval()
            self.succeeded.emit(model)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExperimentWorker(QThread):
    succeeded = Signal(object, float)
    failed = Signal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

    def run(self):
        try:
            frame, false_alarm = run_full_experiment(self._model)
            self.succeeded.emit(frame, float(false_alarm))
        except Exception as exc:
            self.failed.emit(str(exc))
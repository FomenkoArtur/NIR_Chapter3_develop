import time

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

import hw
from config import SEED
from experiment import run_full_experiment
from trainer import build_datasets, get_device, train_model


class TrainWorker(QThread):
    epoch_done = Signal(int, float, float)
    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    SPEED_EVERY = 25          # как часто сообщать скорость обучения, эпох

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            hw.configure_torch_threads()
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            if hw.on_battery():
                self.message.emit("Ноутбук работает от батареи: процессор ограничивает мощность, "
                                  "обучение будет заметно медленнее. Подключите зарядку.")
            self.message.emit("Формирование обучающей выборки...")
            data = build_datasets()
            self.message.emit(f"Объём выборки: обучение {len(data[0])}, проверка {len(data[2])}")
            self.message.emit(f"Режим вычислений: {hw.describe()}, устройство: {get_device()}")

            clock = {'t': time.perf_counter()}

            def on_epoch(epoch, train_loss, val_loss):
                self.epoch_done.emit(epoch, train_loss, val_loss)
                if epoch % self.SPEED_EVERY == 0:
                    now = time.perf_counter()
                    self.message.emit(f"Скорость: {(now - clock['t']) / self.SPEED_EVERY:.1f} с на эпоху")
                    clock['t'] = now

            # тот же train_model, что и в main.py: параметры LR, BATCH_SIZE, WEIGHT_DECAY и архитектура
            # берутся из config (в том числе из tuned/cnn.json)
            with hw.keep_awake():
                model = train_model(data=data, verbose=False, on_epoch=on_epoch, log=self.message.emit,
                                    should_stop=lambda: self._stop_requested)

            if model is None:
                self.failed.emit("Обучение прервано до завершения первой эпохи.")
                return
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
            hw.configure_torch_threads()
            frame, false_alarm = run_full_experiment(self._model)
            self.succeeded.emit(frame, float(false_alarm))
        except Exception as exc:
            self.failed.emit(str(exc))
import tempfile
import wave
from pathlib import Path
import numpy as np
from PySide6.QtCore import QRectF, QTimer, QUrl
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

# ==============================================================================
# НАСТРОЙКИ МОДУЛЯ
# Установите False, чтобы полностью отключить индикатор и убрать его из GUI
# ==============================================================================
ENABLED = True

try:
    from config import DELTA, T_TOTAL, Z_LIMIT
except ImportError:
    DELTA = 0.02
    T_TOTAL = 100
    Z_LIMIT = 3.0

# Пороги срабатывания
YELLOW_EPS = DELTA
RED_EPS = 2 * DELTA
RED_EXCESS = 0.5
HYSTERESIS = 0.8  # Для предотвращения "моргания" при пограничных значениях
STOP_AFTER = 10  # Тактов подряд в красной зоне для критического предупреждения

# Настройки звука
POLL_MS = 50
SAMPLE_RATE = 44100
ALARM_FREQUENCY = 1200
ALARM_VOLUME = 0.7

# Состояния
IDLE, OK, WARN, CRIT = -1, 0, 1, 2

COLORS = {
    IDLE: QColor("#9aa0a6"),  # Серый
    OK: QColor("#2e9e4f"),  # Зеленый
    WARN: QColor("#f2b705"),  # Желтый
    CRIT: QColor("#d93025"),  # Красный
}

TITLES = {
    IDLE: "Нет данных",
    OK: "Норма",
    WARN: "Внимание",
    CRIT: "Критическое отклонение",
}

HINTS = {
    IDLE: "Запустите моделирование",
    OK: "Процесс стабилен, вмешательство не требуется",
    WARN: "Есть отклонение, следите за процессом",
    CRIT: "Выход за допустимые пределы",
}

STOP_TITLE = "ОСТАНОВИТЕ СТАНОК"
STOP_HINT = "Критическое состояние затянулось"


def _next_level(level, eps, excess):
    """Логика переключения состояний с гистерезисом."""
    if eps > RED_EPS or excess > RED_EXCESS:
        return CRIT
    if level == CRIT and (eps > RED_EPS * HYSTERESIS or excess > RED_EXCESS * HYSTERESIS):
        return CRIT
    if eps > YELLOW_EPS or excess > 0.0:
        return WARN
    if level >= WARN and eps > YELLOW_EPS * HYSTERESIS:
        return WARN
    return OK


def classify(result):
    """Классифицирует массив результатов по уровням тревоги."""
    if result is None:
        return None

    eps_arr = np.abs(getattr(result, 'eps', np.array([])))
    if len(eps_arr) == 0:
        return None

    # Нормализованное значение N лежит в (0; 1), поэтому выход за границы
    # оценивается по аргументу сигмоиды z = (x - m) / s = ln(N / (1 - N)):
    # excess = max(|z| / Z_LIMIT - 1; 0), то есть превышение границы 3s в долях 3s.
    n_arr = np.clip(getattr(result, 'N', np.full_like(eps_arr, 0.5)), 1e-12, 1.0 - 1e-12)
    z_abs = np.abs(np.log(n_arr / (1.0 - n_arr)))
    excess = np.clip(z_abs / Z_LIMIT - 1.0, 0.0, None)
    if excess.ndim > 1:
        excess = excess.max(axis=1)

    levels = np.zeros(len(eps_arr), dtype=int)
    level = OK
    for t in range(len(eps_arr)):
        e_val = float(eps_arr[t])
        x_val = float(excess[t]) if hasattr(excess, '__getitem__') else float(excess)
        level = _next_level(level, e_val, x_val)
        levels[t] = level
    return levels


def _alarm_wav_path():
    """Генерирует короткий WAV-файл с сигналом тревоги во временной папке ОС."""

    def tone(duration):
        t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
        fade = np.minimum(1.0, np.minimum(t, duration - t) / 0.01)
        return 0.6 * fade * np.sin(2 * np.pi * ALARM_FREQUENCY * t)

    def silence(duration):
        return np.zeros(int(SAMPLE_RATE * duration))

    signal = np.concatenate([tone(0.15), silence(0.10), tone(0.15), silence(0.50)])
    path = Path(tempfile.gettempdir()) / "rpms_alarm.wav"

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((signal * 32767).astype("<i2").tobytes())
    return path


class Alarm:
    """Управляет воспроизведением звукового сигнала с защитой от сбоев."""

    def __init__(self):
        self._active = False
        self._effect = None
        self._fallback = QTimer()
        self._fallback.setInterval(700)
        self._fallback.timeout.connect(QApplication.beep)

        try:
            from PySide6.QtMultimedia import QSoundEffect
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(str(_alarm_wav_path())))
            effect.setLoopCount(-2)  # -2 означает бесконечный цикл (QSoundEffect.Infinite)
            effect.setVolume(ALARM_VOLUME)
            self._effect = effect
        except Exception:
            self._effect = None

    def _effect_usable(self):
        if self._effect is None:
            return False
        try:
            return self._effect.status() != 2  # 2 = Status.Error в Qt
        except Exception:
            return False

    def start(self):
        if self._active:
            return
        self._active = True
        if self._effect_usable():
            try:
                self._effect.play()
            except Exception:
                self._effect = None
                QApplication.beep()
                self._fallback.start()
        else:
            QApplication.beep()
            self._fallback.start()

    def stop(self):
        if not self._active:
            return
        self._active = False
        self._fallback.stop()
        if self._effect is not None:
            try:
                self._effect.stop()
            except Exception:
                pass


class Led(QWidget):
    """Визуальный круглый индикатор состояния (светодиод) с градиентом."""

    def __init__(self, size=44, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = COLORS[IDLE]

    def set_color(self, color):
        if color != self._color:
            self._color = color
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        gradient = QRadialGradient(
            rect.center().x() - rect.width() * 0.15,
            rect.center().y() - rect.height() * 0.20,
            rect.width() * 0.75,
        )
        gradient.setColorAt(0.0, self._color.lighter(150))
        gradient.setColorAt(1.0, self._color.darker(115))
        painter.setPen(QPen(self._color.darker(160), 1.5))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(rect)


class RpmsIndicator(QGroupBox):
    """Групповой виджет индикатора состояния процесса."""

    def __init__(self, window):
        super().__init__("Состояние процесса")
        self._window = window
        self._source = None
        self._levels = None
        self._reference = None
        self._alarm = Alarm()

        self.led = Led()

        self.title_label = QLabel()
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self.title_label.setFont(font)

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)

        self.readout_label = QLabel()

        self.sound_check = QCheckBox("Звук")
        self.sound_check.setChecked(True)
        self.sound_check.toggled.connect(self._tick)

        head = QHBoxLayout()
        head.addWidget(self.title_label, 1)
        head.addWidget(self.sound_check)

        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addLayout(head)
        texts.addWidget(self.hint_label)
        texts.addWidget(self.readout_label)

        layout = QHBoxLayout(self)
        layout.addWidget(self.led)
        layout.addLayout(texts, 1)

        self.setFixedHeight(96)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._tick()

    def _rebind(self, results):
        self._source = results
        if not results:
            self._levels = None
            self._reference = None
            return

        self._reference = results.get('cnn', next(iter(results.values())))
        self._levels = classify(self._reference)

    def _tick(self):
        window = self._window
        if window.results is not self._source:
            self._rebind(window.results)

        count = window.frame_slider.value()
        if self._levels is None or count <= 0:
            self._show(IDLE, HINTS[IDLE], "")
            self._alarm.stop()
            return

        index = min(count, T_TOTAL) - 1
        level = int(self._levels[index])

        streak = 0
        while streak <= index and self._levels[index - streak] == CRIT:
            streak += 1

        eps = abs(float(self._reference.eps[index]))
        readout = f"|ε| = {eps:.3f} (допуск ±{DELTA})"
        title, hint = TITLES[level], HINTS[level]

        if level == CRIT:
            readout += f", тактов подряд: {streak}"
            if streak >= STOP_AFTER:
                title, hint = STOP_TITLE, STOP_HINT

        self._show(level, hint, readout, title)

        playing = window.timer.isActive() and self.isVisible() and self.sound_check.isChecked()
        if level == CRIT and playing:
            self._alarm.start()
        else:
            self._alarm.stop()

    def _show(self, level, hint, readout, title=None):
        self.led.set_color(COLORS[level])
        self.title_label.setText(title or TITLES[level])
        self.hint_label.setText(hint)
        self.readout_label.setText(readout)


def attach_rpms_indicator(window, layout):
    """
    Встраивает индикатор в переданный layout.
    Если ENABLED = False, функция ничего не делает и возвращает None.
    """
    if not ENABLED:
        return None
    indicator = RpmsIndicator(window)
    layout.addWidget(indicator)
    return indicator
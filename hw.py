"""
Настройка вычислительных ресурсов под ноутбук Lenovo 83A2: Intel Core i3-1315U
(2 P-ядра с Hyper-Threading + 4 E-ядра; в Windows логические процессоры 0-3 - P-ядра, 4-7 - E-ядра),
16 ГБ DDR4-3200, дискретной видеокарты нет.

Режим вычислений берётся в таком порядке (первое найденное):
  1) переменные окружения NIR_THREADS, NIR_PIN, NIR_DEVICE;
  2) файл tuned/hw_train.json - его записывает bench_train.py после замера на этом компьютере;
  3) значения по умолчанию ниже (2 потока, без закрепления, устройство выбирается само).

NIR_PIN - номера логических процессоров через запятую ("0,2"), на которых работают потоки PyTorch;
"none" или пустая строка - не закреплять. Закрепление действует, только если номеров ровно столько, сколько потоков.
NIR_DEVICE - auto (CUDA, если есть, иначе процессор), cpu или xpu (видеокарта Intel, если установлен PyTorch с XPU).
"""

import contextlib
import ctypes
import json
import os
import sys

_TUNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned", "hw_train.json")


def _load_tuned():
    try:
        with open(_TUNED_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_tuned = _load_tuned()


def _setting(env_name, key, default):
    value = os.environ.get(env_name)
    return value if value is not None and value != "" else _tuned.get(key, default)


try:
    THREADS = max(1, int(_setting("NIR_THREADS", "threads", 2)))
except (TypeError, ValueError):
    THREADS = 2

PIN = str(os.environ.get("NIR_PIN", _tuned.get("pin", "")) or "")
if PIN.strip().lower() == "none":
    PIN = ""
DEVICE = str(_setting("NIR_DEVICE", "device", "auto")).lower()

_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_PROCESS_POWER_THROTTLING = 4
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong)]


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong)]


def _configure_windows_process():
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                   ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(handle, _ABOVE_NORMAL_PRIORITY_CLASS)
        state = _PowerThrottlingState(1, 1, 0)       # отключить EcoQoS (режим энергосбережения процесса)
        kernel32.SetProcessInformation(handle, _PROCESS_POWER_THROTTLING,
                                       ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass


def pin_list():
    """Номера логических процессоров для закрепления потоков или [], если закрепление не нужно."""
    ids = [p.strip() for p in PIN.split(",") if p.strip()]
    if not ids or not all(p.isdigit() for p in ids) or len(ids) != THREADS:
        return []
    return ids


def prepare():
    """Вызывается до import torch: число потоков, закрепление потоков и режим процесса в Windows."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    ids = pin_list()
    if ids and sys.platform == "win32" and "KMP_AFFINITY" not in os.environ:
        os.environ["KMP_AFFINITY"] = "granularity=fine,proclist=[%s],explicit" % ",".join(ids)
    _configure_windows_process()


def configure_torch_threads():
    """Вызывается в каждом рабочем потоке, где выполняются расчёты PyTorch."""
    import torch
    torch.set_num_threads(THREADS)


def pick_device():
    """Имя устройства для обучения: 'cpu', 'cuda' или 'xpu'."""
    import torch
    if DEVICE == "cpu":
        return "cpu"
    if DEVICE == "xpu":
        xpu = getattr(torch, "xpu", None)
        return "xpu" if xpu is not None and xpu.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe():
    ids = pin_list()
    pin = f", потоки закреплены за логическими процессорами {','.join(ids)}" if ids else ""
    return f"потоков PyTorch: {THREADS}{pin}"


def on_battery():
    """True, если ноутбук работает от батареи (при этом процессор ограничивает мощность)."""
    if sys.platform != "win32":
        return False
    try:
        status = _SystemPowerStatus()
        if ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(status)):
            return status.ACLineStatus == 0
    except Exception:
        pass
    return False


@contextlib.contextmanager
def keep_awake():
    """Пока идёт обучение, Windows не уходит в сон. Действует для потока, который вызвал функцию."""
    set_state = None
    if sys.platform == "win32":
        try:
            set_state = ctypes.WinDLL("kernel32").SetThreadExecutionState
            set_state.argtypes = [ctypes.c_uint32]
            set_state.restype = ctypes.c_uint32
            set_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        except Exception:
            set_state = None
    try:
        yield
    finally:
        if set_state is not None:
            try:
                set_state(_ES_CONTINUOUS)
            except Exception:
                pass
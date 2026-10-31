"""
Настройка вычислительных ресурсов под Intel Core i3-1315U (2 P-ядра + 4 E-ядра, 8 потоков).

Число потоков подбирается скриптом bench_hw.py; значение можно переопределить
переменной окружения NIR_THREADS без правки кода.
"""

import ctypes
import os
import sys

THREADS = int(os.environ.get("NIR_THREADS", "2"))

_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_PROCESS_POWER_THROTTLING = 4


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong)]


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
        state = _PowerThrottlingState(1, 1, 0)
        kernel32.SetProcessInformation(handle, _PROCESS_POWER_THROTTLING,
                                       ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass


def prepare():
    """Вызывается до import torch: потоки по умолчанию и режим процесса в Windows."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    _configure_windows_process()


def configure_torch_threads():
    """Вызывается в каждом рабочем потоке, где выполняются расчёты PyTorch."""
    import torch
    torch.set_num_threads(THREADS)
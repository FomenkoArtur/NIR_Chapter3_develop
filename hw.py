import ctypes
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass

GB = 1 << 30
MAX_WORKERS = 16
MAX_PER_GPU = 4
GPU_MB_PER_WORKER = 700
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuned')

_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_PROCESS_POWER_THROTTLING = 4


@dataclass
class Gpu:
    index: int
    name: str
    total_mb: int
    free_mb: int


@dataclass
class Hardware:
    cpu: str
    logical: int
    physical: int
    fast: int
    ram_total: int
    ram_free: int
    gpus: list


def _read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return None


def _run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def _ranges(text):
    out = set()
    for part in (text or '').split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def _allowed_cpus():
    if hasattr(os, 'sched_getaffinity'):
        return sorted(os.sched_getaffinity(0))
    count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    return list(range(count))


def _cgroup_cpus():
    text = _read('/sys/fs/cgroup/cpu.max')
    if text:
        quota, period = text.split()[:2]
        return None if quota == 'max' else math.ceil(int(quota) / int(period))
    quota = _read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = _read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota and period and int(quota) > 0:
        return math.ceil(int(quota) / int(period))
    return None


def _cores_windows():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GetLogicalProcessorInformationEx
    query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    size = ctypes.c_ulong(0)
    query(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not query(0, buf, ctypes.byref(size)):
        return None
    raw, offset, cores = buf.raw[:size.value], 0, []
    while offset < len(raw):
        length = struct.unpack_from('<I', raw, offset + 4)[0]
        groups = struct.unpack_from('<H', raw, offset + 30)[0]
        threads = sum(bin(struct.unpack_from('<Q', raw, offset + 32 + 16 * g)[0]).count('1')
                      for g in range(groups))
        cores.append((raw[offset + 9], threads))
        offset += length
    return cores


def _cores_linux(cpus):
    fast_set = _ranges(_read('/sys/devices/cpu_core/cpus'))
    allowed, seen, cores = set(cpus), set(), []
    for c in cpus:
        base = f'/sys/devices/system/cpu/cpu{c}/'
        siblings = frozenset(_ranges(_read(base + 'topology/thread_siblings_list')) or {c})
        if siblings in seen:
            continue
        seen.add(siblings)
        capacity = _read(base + 'cpu_capacity')
        perf = int(c in fast_set) if fast_set else int(capacity or 0)
        cores.append((perf, len(siblings & allowed)))
    return cores


def _cores_macos():
    physical = int(_run('sysctl', '-n', 'hw.physicalcpu') or 0)
    if not physical:
        return None
    logical = int(_run('sysctl', '-n', 'hw.logicalcpu') or physical)
    fast = int(_run('sysctl', '-n', 'hw.perflevel0.physicalcpu') or 0) or physical
    smt = max(1, logical // physical)
    return [(1, smt)] * fast + [(0, smt)] * (physical - fast)


def _cpu_name():
    system = platform.system()
    if system == 'Windows':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    elif system == 'Linux':
        found = re.search(r'model name\s*:\s*(.+)', _read('/proc/cpuinfo') or '')
        if found:
            return found.group(1)
    elif system == 'Darwin':
        name = _run('sysctl', '-n', 'machdep.cpu.brand_string')
        if name:
            return name
    return platform.processor() or platform.machine()


def _memory():
    system = platform.system()
    total = free = None
    if system == 'Windows':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                        ('page_total', ctypes.c_ulonglong), ('page_avail', ctypes.c_ulonglong),
                        ('virt_total', ctypes.c_ulonglong), ('virt_avail', ctypes.c_ulonglong),
                        ('ext_avail', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total, free = status.total, status.avail
    elif system == 'Linux':
        info = {}
        for line in (_read('/proc/meminfo') or '').splitlines():
            key, _, value = line.partition(':')
            if value.split():
                info[key] = int(value.split()[0]) * 1024
        total, free = info.get('MemTotal'), info.get('MemAvailable')
        limit = _read('/sys/fs/cgroup/memory.max') or _read('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if limit and limit.isdigit() and total and int(limit) < total:
            used = _read('/sys/fs/cgroup/memory.current') or _read('/sys/fs/cgroup/memory/memory.usage_in_bytes')
            total = int(limit)
            free = max(total - int(used or 0), 0)
    elif system == 'Darwin':
        total = int(_run('sysctl', '-n', 'hw.memsize') or 0) or None
        free = int(total * 0.6) if total else None
    total = total or 8 * GB
    return total, free or total // 2


def _nvidia_gpus():
    out = _run('nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
               '--format=csv,noheader,nounits')
    gpus = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4 and parts[0].isdigit() and parts[2].isdigit() and parts[3].isdigit():
            gpus[int(parts[0])] = Gpu(int(parts[0]), parts[1], int(parts[2]), int(parts[3]))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        ids = [v.strip() for v in visible.split(',') if v.strip()]
        if all(v.isdigit() for v in ids):
            return [Gpu(i, gpus[int(v)].name, gpus[int(v)].total_mb, gpus[int(v)].free_mb)
                    for i, v in enumerate(ids) if int(v) in gpus]
    return list(gpus.values())


_HARDWARE = None


def detect():
    global _HARDWARE
    if _HARDWARE is not None:
        return _HARDWARE
    cpus = _allowed_cpus()
    logical = len(cpus)
    limit = _cgroup_cpus()
    if limit:
        logical = max(1, min(logical, limit))
    cores = None
    try:
        system = platform.system()
        if system == 'Windows':
            cores = _cores_windows()
        elif system == 'Linux':
            cores = _cores_linux(cpus)
        elif system == 'Darwin':
            cores = _cores_macos()
    except Exception:
        cores = None
    if not cores:
        try:
            import psutil
            physical = psutil.cpu_count(logical=False) or logical
        except ImportError:
            physical = logical
        cores = [(0, 1)] * max(1, min(physical, logical))
    best = max(perf for perf, _ in cores)
    physical = max(1, min(len(cores), logical))
    fast = max(1, min(sum(perf == best for perf, _ in cores), physical))
    total, free = _memory()
    _HARDWARE = Hardware(_cpu_name(), logical, physical, fast, total, free, _nvidia_gpus())
    return _HARDWARE


def usable_gpus(hardware=None):
    hardware = hardware or detect()
    if os.environ.get('NIR_DEVICE', '').lower() == 'cpu':
        return []
    import torch
    if not torch.cuda.is_available():
        return []
    if hardware.gpus:
        return hardware.gpus
    gpus = []
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory >> 20
        gpus.append(Gpu(i, torch.cuda.get_device_name(i), total, total))
    return gpus


def describe(hardware=None):
    hw = hardware or detect()
    hybrid = f', производительных {hw.fast}' if hw.fast < hw.physical else ''
    gpus = '; '.join(f'{g.name} {g.free_mb}/{g.total_mb} МБ' for g in usable_gpus(hw)) or 'нет'
    return (f'CPU: {hw.cpu}, ядер {hw.physical}{hybrid}, потоков {hw.logical}; '
            f'ОЗУ свободно {hw.ram_free / GB:.1f} из {hw.ram_total / GB:.1f} ГБ; GPU: {gpus}')


def plan_workers(n_trials, requested=None):
    hw = detect()
    slots = [g.index for rnd in range(MAX_PER_GPU) for g in usable_gpus(hw)
             if g.free_mb >= (rnd + 1) * GPU_MB_PER_WORKER]
    if slots:
        by_ram = int((hw.ram_free - GB) // (1.8 * GB))
        auto = min(len(slots), hw.logical, by_ram, MAX_WORKERS, n_trials)
    else:
        by_ram = int((hw.ram_free - GB) // GB)
        auto = min(hw.physical, by_ram, MAX_WORKERS, n_trials)
    workers = max(1, requested or auto)
    devices = [slots[i % len(slots)] for i in range(workers)] if slots else [None] * workers
    threads = 1 if slots else max(1, hw.physical // workers)
    return {'workers': workers, 'devices': devices, 'threads': threads, 'hardware': hw}


def cpu_workers(limit, ram_per_worker=GB // 2):
    hw = detect()
    by_ram = int((hw.ram_free - GB) // ram_per_worker)
    return max(1, min(limit, hw.physical, by_ram, MAX_WORKERS))


def default_threads(hardware=None):
    override = os.environ.get('NIR_THREADS')
    if override:
        return max(1, int(override))
    return max(1, min((hardware or detect()).fast, 8))


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [('Version', ctypes.c_ulong),
                ('ControlMask', ctypes.c_ulong),
                ('StateMask', ctypes.c_ulong)]


def _configure_windows_process():
    if sys.platform != 'win32':
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
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
    hardware = detect()
    threads = str(default_threads(hardware))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = threads
    _configure_windows_process()
    return hardware


def autotune_threads(hardware=None):
    import time
    import torch
    import torch.nn as nn

    hw = hardware or detect()
    path = os.path.join(CACHE_DIR, 'hw_profile.json')
    key = f'{hw.cpu}|{hw.logical}|{hw.physical}|{hw.fast}|{torch.__version__}'
    try:
        with open(path, encoding='utf-8') as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return int(cache[key]['threads'])

    candidates = sorted(c for c in {1, 2, 4, 8, hw.fast, hw.physical, hw.logical}
                        if c <= min(hw.logical, 16))
    rng_state = torch.get_rng_state()
    net = nn.Sequential(
        nn.Conv1d(17, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 8))
    optimizer = torch.optim.Adam(net.parameters())
    x, y = torch.randn(128, 17, 32), torch.randn(128, 8)
    timings = {}
    for n in candidates:
        torch.set_num_threads(n)
        for step in range(25):
            if step == 5:
                start = time.perf_counter()
            optimizer.zero_grad()
            ((net(x) - y) ** 2).mean().backward()
            optimizer.step()
        timings[n] = (time.perf_counter() - start) / 20
    torch.set_rng_state(rng_state)

    best = min(timings.values())
    threads = min(n for n, t in timings.items() if t <= best * 1.05)
    cache[key] = {'threads': threads, 'ms_per_step': {str(n): round(t * 1000, 2) for n, t in timings.items()}}
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass
    return threads


def configure_torch_threads(threads=None):
    import torch
    hw = detect()
    if threads is None:
        threads = default_threads(hw)
        if 'NIR_THREADS' not in os.environ and not usable_gpus(hw) and os.environ.get('NIR_AUTOTUNE', '1') != '0':
            threads = autotune_threads(hw)
    torch.set_num_threads(threads)
    return threads


def device():
    import torch
    kind = os.environ.get('NIR_DEVICE', '').lower()
    if kind == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    gpus = usable_gpus()
    if gpus:
        best = max(gpus, key=lambda g: g.free_mb)
        torch.cuda.set_device(best.index)
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda', best.index)
    return torch.device('cpu')
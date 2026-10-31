"""
Замер скорости обучения CNN на этом компьютере и выбор режима вычислений: число потоков PyTorch,
закрепление потоков за P-ядрами, устройство (процессор или видеокарта Intel, если стоит PyTorch с XPU).

Запуск:  python bench_train.py        (около 5 минут; ноутбук включить в зарядку, остальные программы закрыть)

Каждый режим проверяется в отдельном процессе, потому что число потоков и закрепление нельзя поменять
после import torch. Замер идёт на настоящей модели из config.py (с учётом tuned/cnn.json) и настоящей обучающей
выборке, шаги обучения те же, что в trainer.train_model. Лучший режим записывается в tuned/hw_train.json,
оттуда его читает hw.py при запуске GUI и main.py. Вернуться к значениям по умолчанию - удалить этот файл.
"""

import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "tuned", "hw_train.json")

# (потоки, закрепление). Логические процессоры 0-3 - два P-ядра с Hyper-Threading, 4-7 - четыре E-ядра.
# Первый режим - прежняя настройка (2 потока без закрепления), с ним сравниваются остальные.
CANDIDATES = [(2, ""), (2, "0,2"), (3, ""), (4, ""), (4, "0,1,2,3"), (6, ""), (8, "")]
TRAIN_SUBSET, VAL_SUBSET = 2048, 512        # часть выборки: шаги те же, замер короче
WARMUP_EPOCHS, TIMED_EPOCHS = 1, 3
RECHECK = 3                                 # сколько лучших режимов перепроверить (ноутбук нагревается)


def child():
    """Один режим: обучает несколько эпох на части выборки и печатает время эпохи."""
    import hw
    hw.prepare()
    import numpy as np
    import torch
    hw.configure_torch_threads()
    from config import SEED
    from trainer import build_datasets, get_device, train_model

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    X, Y, Xv, Yv, groups = build_datasets()
    step = max(1, len(X) // TRAIN_SUBSET)
    data = (X[::step][:TRAIN_SUBSET], Y[::step][:TRAIN_SUBSET], Xv[:VAL_SUBSET], Yv[:VAL_SUBSET],
            groups[:VAL_SUBSET])

    stamps = []
    train_model(data=data, callback=lambda epoch, val_loss: stamps.append(time.perf_counter()),
                verbose=False, max_epochs=WARMUP_EPOCHS + TIMED_EPOCHS, patience=10 ** 6,
                log=lambda text: None)
    laps = [b - a for a, b in zip(stamps[WARMUP_EPOCHS - 1:], stamps[WARMUP_EPOCHS:])]

    # проход обучения стоит примерно втрое дороже проверки на тот же объём (прямой + обратный ход против прямого)
    scale = (3 * len(X) + len(Xv)) / (3 * len(data[0]) + len(data[2]))
    xpu = getattr(torch, "xpu", None)
    try:
        xpu_ok = bool(xpu is not None and xpu.is_available())
    except Exception:
        xpu_ok = False
    print("RESULT " + json.dumps({"est_full_epoch": statistics.median(laps) * scale,
                                  "threads": torch.get_num_threads(),
                                  "device": get_device().type, "xpu": xpu_ok}), flush=True)


def run_child(threads, pin, device):
    env = dict(os.environ, NIR_THREADS=str(threads), NIR_PIN=pin or "none", NIR_DEVICE=device,
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("KMP_AFFINITY", None)
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, cwd=HERE,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=1200)
    except subprocess.TimeoutExpired:
        return {"error": "превышено время ожидания"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {"error": tail[-1] if tail else f"код возврата {proc.returncode}"}


def label(key):
    threads, pin, device = key
    where = "видеокарта Intel (XPU)" if device == "xpu" else "процессор"
    return f"{where}, потоков {threads}, закрепление {pin or 'нет'}"


def pick_best(runs):
    """runs: {(потоки, закрепление, устройство): [время полной эпохи, ...]} -> ключ самого быстрого режима."""
    return min(runs, key=lambda key: min(runs[key]))


def main():
    if "--child" in sys.argv:
        child()
        return

    print("Замер обучения CNN: каждый режим - отдельный процесс, 1 прогревочная и 3 замерочные эпохи.\n")
    runs = {}
    xpu_available = False

    def measure(key):
        result = run_child(*key)
        if "error" in result:
            print(f"  {label(key)}: ошибка ({result['error']})", flush=True)
            return False
        if key[2] == "xpu" and result.get("device") != "xpu":
            print(f"  {label(key)}: недоступно, обучение ушло на {result.get('device')}", flush=True)
            return False
        runs.setdefault(key, []).append(result["est_full_epoch"])
        print(f"  {label(key)}: {result['est_full_epoch']:.1f} с на полную эпоху", flush=True)
        return result.get("xpu", False)

    for threads, pin in CANDIDATES:
        xpu_available = measure((threads, pin, "cpu")) or xpu_available
    if xpu_available:
        measure((2, "", "xpu"))
    if not runs:
        print("\nНи один режим не удалось замерить, tuned/hw_train.json не изменён.")
        return

    print("\nПерепроверка лучших режимов (в обратном порядке):")
    ranked = sorted(runs, key=lambda key: min(runs[key]))
    for key in reversed(ranked[:RECHECK]):
        measure(key)

    best = pick_best(runs)
    best_time = min(runs[best])
    base_key = (CANDIDATES[0][0], CANDIDATES[0][1], "cpu")
    print("\nИтог (лучшее время полной эпохи):")
    for key in sorted(runs, key=lambda k: min(runs[k])):
        speedup = f"{min(runs[base_key]) / min(runs[key]):.2f}x" if base_key in runs else "-"
        print(f"  {min(runs[key]):7.1f} с  {speedup:>6}  {label(key)}{'   <- выбран' if key == best else ''}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"threads": best[0], "pin": best[1], "device": best[2],
                   "est_epoch_sec": round(best_time, 2)}, f, indent=2)
    print(f"\nЗаписано: {OUT_FILE}")
    print(f"Оценка: 300 эпох - примерно {best_time * 300 / 60:.0f} мин.")


if __name__ == "__main__":
    main()
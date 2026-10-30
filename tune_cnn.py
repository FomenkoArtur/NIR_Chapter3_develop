"""
Автонастройка гиперпараметров CNN: Optuna (TPE + Hyperband), критерий - замкнутый контур.
Параллельно: несколько процессов-воркеров делят один GPU и один Optuna-study (SQLite).
Запуск: python tune_cnn.py [число_испытаний] [число_воркеров]   ->  tuned/cnn.json
Запускать ПОСЛЕ tune_pid.py.
"""

import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import optuna

import config as C

SEED_CNN_TUNE = 150000
MAX_EP, PATIENCE_EP = 200, 25
STORAGE_URL = 'sqlite:///' + os.path.join('tuned', 'cnn_study.db')

MANUAL = {'k1': 5, 'k2': 3, 'k3': 3, 'c1': 32, 'c2': 64, 'c3': 64, 'fc': 128,
          'pool': 8, 'dropout': 0.0, 'LR': 5e-4, 'WEIGHT_DECAY': 1e-8, 'BATCH_SIZE': 128}


def make_storage():
    return optuna.storages.RDBStorage(
        STORAGE_URL, engine_kwargs={'connect_args': {'timeout': 120}})   # ожидание блокировки SQLite


def make_pruner():
    return optuna.pruners.HyperbandPruner(min_resource=10, max_resource=MAX_EP, reduction_factor=3)


def split(p):
    model_hp = {k: p[k] for k in ('k1', 'k2', 'k3', 'c1', 'c2', 'c3', 'fc', 'pool', 'dropout')}
    train_hp = {k: p[k] for k in ('LR', 'WEIGHT_DECAY', 'BATCH_SIZE')}
    return model_hp, train_hp


def worker(wid, n_trials):
    import torch
    from episode import CNNPolicy
    from trainer import build_datasets, train_model
    from tune_pid import build_cases, evaluate

    torch.set_num_threads(1)                    # без перегрузки CPU между воркерами
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(C.SEED)
    tag = f"[w{wid}]"

    data = build_datasets()
    cases, quiet = build_cases(SEED_CNN_TUNE)

    def objective(trial):
        p = {
            'k1': trial.suggest_categorical('k1', [3, 5, 7]),
            'k2': trial.suggest_categorical('k2', [3, 5, 7]),
            'k3': trial.suggest_categorical('k3', [3, 5, 7]),
            'c1': trial.suggest_categorical('c1', [16, 32, 64]),
            'c2': trial.suggest_categorical('c2', [32, 64, 128]),
            'c3': trial.suggest_categorical('c3', [32, 64, 128]),
            'fc': trial.suggest_categorical('fc', [64, 128, 256]),
            'pool': trial.suggest_categorical('pool', [4, 8, 16]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'LR': trial.suggest_float('LR', 1e-4, 3e-3, log=True),
            'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', 1e-8, 1e-2, log=True),
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [64, 128, 256]),
        }
        model_hp, train_hp = split(p)
        trial.set_user_attr('receptive_field', p['k1'] + p['k2'] + p['k3'] - 2)
        t_trial = time.time()
        print(f"{tag} trial {trial.number} старт: {p}", flush=True)

        def cb(epoch, val_loss):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"{tag} trial {trial.number} ОТСЕЧЁН на эпохе {epoch} "
                      f"(t={time.time() - t_trial:.1f} c)", flush=True)
                raise optuna.TrialPruned()

        torch.manual_seed(C.SEED)
        model = train_model(model_hp, train_hp, data=data, callback=cb, verbose=False,
                            max_epochs=MAX_EP, patience=PATIENCE_EP)      # вернёт модель на CPU
        t_train = time.time() - t_trial
        cost = evaluate(lambda: CNNPolicy(model), cases, quiet)
        cost_out = cost if np.isfinite(cost) else 1e3
        print(f"{tag} trial {trial.number} финиш: cost={cost_out:.4f}, "
              f"обучение {t_train:.1f} c, всего {time.time() - t_trial:.1f} c", flush=True)
        return cost_out

    study = optuna.load_study(
        study_name='cnn', storage=make_storage(),
        sampler=optuna.samplers.TPESampler(seed=C.SEED + wid, constant_liar=True),
        pruner=make_pruner())
    study.optimize(objective, n_trials=n_trials)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs('tuned', exist_ok=True)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=C.SEED, constant_liar=True),
        pruner=make_pruner(), storage=make_storage(),
        study_name='cnn', load_if_exists=True)
    print(f"[tune_cnn] Study 'cnn': уже есть испытаний {len(study.trials)}; "
          f"новых {n_trials}, воркеров {n_workers}", flush=True)
    if len(study.trials) == 0:
        study.enqueue_trial(MANUAL)             # ручная конфигурация - первое испытание

    per_worker = math.ceil(n_trials / n_workers)
    ctx = mp.get_context('spawn')               # spawn обязателен при работе с CUDA
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(i, per_worker)) for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"[tune_cnn] Оптимизация завершена за {time.time() - t0:.1f} c", flush=True)

    study = optuna.load_study(study_name='cnn', storage=make_storage())
    best = study.best_trial
    base = study.trials[0].value
    print(f"Ручная конфигурация: cost = {base}")
    print(f"Лучшая: cost = {best.value:.4f}, рецептивное поле = {best.user_attrs['receptive_field']}")
    print(best.params)

    model_hp, train_hp = split(best.params)
    with open(os.path.join('tuned', 'cnn.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': model_hp, 'train': train_hp, 'cost': best.value,
                   'receptive_field': best.user_attrs['receptive_field']}, f, indent=2)
    print("Сохранено в tuned/cnn.json")


if __name__ == '__main__':
    main()
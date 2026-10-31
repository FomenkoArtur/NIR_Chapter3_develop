"""
Генерация выборок (fastsim) и обучение CNN (данные в памяти устройства, без DataLoader).

train_model - единственная реализация цикла обучения: её вызывают main.py, кнопка «Обучить регулятор»
(gui/workers.py) и замер bench_train.py. Поэтому режим вычислений, подобранный под компьютер (hw.py),
действует везде одинаково, а результат обучения не зависит от того, откуда оно запущено.
"""

import time
from copy import deepcopy

import torch
import torch.optim as optim

import hw
from config import (MAX_EPOCHS, PATIENCE, TRAIN_HP, ACTIVE_W, SEED_TEACHER_START, SEED_CRITICAL_START,
                    CRITICAL_SCENARIOS, QUIET_BAND, FA_MARGIN, FA_W)
from fastsim import critical_arrays, train_arrays, training_profiles
from model import CNNController


def get_device():
    return torch.device(hw.pick_device())


def weighted_mse(pred, target, active_w=ACTIVE_W):
    quiet = (target.abs() <= QUIET_BAND).float()
    err = (pred - target) ** 2 * (1.0 + active_w * (1.0 - quiet))
    false_alarm = (torch.relu(pred.abs() - FA_MARGIN) ** 2 * quiet).mean()
    return err.mean() + FA_W * false_alarm


def iterate_batches(X, Y, batch_size):
    order = torch.randperm(len(X))
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], Y[idx]


def generate_train_set():
    print("Формирование обучающей выборки...")
    X, Y = train_arrays()
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def generate_critical_set():
    print("Формирование тестовой выборки...")
    X, Y, groups = critical_arrays()
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), groups


def check_sets(X_train, Y_train, X_test, Y_test):
    train_seeds = set(SEED_TEACHER_START + i for i in range(1, len(training_profiles()) + 1))
    test_seeds = set(SEED_CRITICAL_START + i for i in range(1, len(CRITICAL_SCENARIOS) + 1))
    assert not (train_seeds & test_seeds), "Зёрна пересекаются"

    train_keys = {row.tobytes() for row in X_train.numpy()}
    duplicates = sum(row.tobytes() in train_keys for row in X_test.numpy())
    assert duplicates == 0, f"Пересечений окон: {duplicates}"

    eps_train = X_train[:, 0, :].abs().max(dim=1).values
    eps_test = X_test[:, 0, :].abs().max(dim=1).values
    eps_max_train = eps_train.max().item()
    outside = (eps_test > eps_max_train).float().mean().item()
    sat_train = (Y_train.abs().max(dim=1).values > 0.99).float().mean().item()
    sat_test = (Y_test.abs().max(dim=1).values > 0.99).float().mean().item()

    n_train, n_test = len(X_train), len(X_test)
    print(f"Выборки: обучение {n_train} / тест {n_test}")
    print(f"  max|eps|: обучение {eps_max_train:.3f}, тест {eps_test.max().item():.3f}; вне обучения: {outside:.1%}")
    print(f"  насыщение (|u| > 0.99): обучение {sat_train:.1%}, тест {sat_test:.1%}")


def build_datasets():
    X_train, Y_train = generate_train_set()
    X_test, Y_test, groups = generate_critical_set()
    check_sets(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test, groups


def generate_training_data():
    X_train, Y_train, X_test, Y_test, _ = build_datasets()
    return X_train, Y_train, X_test, Y_test


def mse_by_scenario(model, X_test, Y_test, groups):
    model.eval()
    dev = next(model.parameters()).device
    with torch.inference_mode():
        err = ((model(X_test.to(dev)) - Y_test.to(dev)) ** 2).mean(dim=1).cpu().numpy()
    return {name: float(err[groups == name].mean()) for name in CRITICAL_SCENARIOS}


def weights_fingerprint(model):
    return sum(p.detach().abs().sum().item() for p in model.parameters())


def train_model(model_hp=None, train_hp=None, data=None, callback=None, verbose=True,
                max_epochs=None, patience=None, on_epoch=None, should_stop=None, log=print):
    """
    Обучение (на GPU, если выбран; на этом ноутбуке - на процессоре). Возвращает модель на CPU - остальной код
    (CNNPolicy и т.д.) работает без изменений.

    callback(epoch, val_loss)               - как раньше (эпохи с нуля), для tune-скриптов и замеров;
    on_epoch(epoch, train_loss, val_loss)   - после каждой эпохи, номер эпохи с единицы (кривые обучения в GUI);
    should_stop()                           - опрашивается перед каждым батчем; True - остановить обучение
                                              (недообученная эпоха отбрасывается, возвращаются лучшие веса);
    log(text)                               - куда писать сообщения (print или сигнал GUI).
    Если остановка пришла до конца первой эпохи, возвращается None.
    """
    hp = dict(TRAIN_HP)
    hp.update(train_hp or {})
    max_epochs = max_epochs or MAX_EPOCHS
    patience = patience or PATIENCE

    X_train, Y_train, X_val, Y_val, groups = data if data is not None else build_datasets()

    dev = get_device()
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    Xt, Yt = X_train.to(dev), Y_train.to(dev)
    Xv, Yv = X_val.to(dev), Y_val.to(dev)
    n = len(Xt)
    bs = int(hp['BATCH_SIZE'])

    model = CNNController(model_hp).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=hp['LR'], weight_decay=hp['WEIGHT_DECAY'],
                           fused=(dev.type == 'cuda'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    if verbose:
        log(f"Обучение CNN на {dev}...")

    best_val_loss = float('inf')
    bad_epochs = 0
    best_state = None
    first_by_scenario = None

    for epoch in range(max_epochs):
        t_epoch = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=dev)
        Xp, Yp = Xt[perm], Yt[perm]                   # одна перестановка за эпоху вместо двух выборок на батч
        total = torch.zeros((), device=dev)          # накопление на устройстве, без .item() в цикле
        interrupted = False
        for i in range(0, n, bs):
            if should_stop is not None and should_stop():
                interrupted = True
                break
            xb, yb = Xp[i:i + bs], Yp[i:i + bs]
            if len(xb) < 2:                           # BatchNorm не работает на батче из 1 элемента
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.detach() * len(xb)
        if interrupted:
            log("Обучение остановлено пользователем.")
            break
        train_loss = total.item() / n                 # единственная синхронизация за эпоху (вместе с val)

        model.eval()
        with torch.inference_mode():
            val_loss = weighted_mse(model(Xv), Yv).item()
        scheduler.step(val_loss)

        if verbose:
            if epoch == 0:
                first_by_scenario = mse_by_scenario(model, X_val, Y_val, groups)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log(f"  эпоха {epoch + 1}: train = {train_loss:.6f}, val = {val_loss:.6f}, "
                    f"lr = {optimizer.param_groups[0]['lr']:.2e}, {time.perf_counter() - t_epoch:.1f} с/эпоха")

        if callback is not None:
            callback(epoch, val_loss)
        if on_epoch is not None:
            on_epoch(epoch + 1, train_loss, val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"Ранний останов на эпохе {epoch + 1}, val MSE = {best_val_loss:.6f}")
                break

    if best_state is None:
        return None
    model.load_state_dict(best_state)

    if verbose and first_by_scenario is not None:
        final_by_scenario = mse_by_scenario(model, X_val, Y_val, groups)
        log("\nMSE на критических ситуациях:")
        for name in CRITICAL_SCENARIOS:
            first, final = first_by_scenario[name], final_by_scenario[name]
            log(f"  {name}: {first:.6f} -> {final:.6f} ({final / first - 1:+.0%})")
        log("Обучение завершено.")

    model = model.cpu().eval()
    del Xt, Yt, Xv, Yv
    return model
"""
Генерация выборок (fastsim) и обучение CNN (данные в памяти устройства, без DataLoader).
"""
import hw
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy

from config import (MAX_EPOCHS, PATIENCE, TRAIN_HP, ACTIVE_W, SEED_TEACHER_START, SEED_CRITICAL_START,
                    CRITICAL_SCENARIOS)
from fastsim import critical_arrays, train_arrays, training_profiles
from model import CNNController


def get_device():
    return hw.device()


def weighted_mse(pred, target, active_w=ACTIVE_W):
    err = (pred - target) ** 2
    active = (target.abs().max(dim=1, keepdim=True).values > 0.1).float()
    w = 1.0 + active_w * active
    return (err * w).mean()


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
                max_epochs=None, patience=None):
    """
    Обучение на GPU (если есть). Возвращает модель на CPU - остальной код (CNNPolicy и т.д.)
    работает без изменений.
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
        print(f"Обучение CNN на {dev}...")

    best_val_loss = float('inf')
    bad_epochs = 0
    best_state = None
    first_by_scenario = None

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        Xp, Yp = Xt[perm], Yt[perm]                   # одна перестановка за эпоху вместо двух выборок на батч
        total = torch.zeros((), device=dev)          # накопление на устройстве, без .item() в цикле
        for i in range(0, n, bs):
            xb, yb = Xp[i:i + bs], Yp[i:i + bs]
            if len(xb) < 2:                           # BatchNorm не работает на батче из 1 элемента
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.detach() * len(xb)
        train_loss = total.item() / n                 # единственная синхронизация за эпоху (вместе с val)

        model.eval()
        with torch.inference_mode():
            val_loss = weighted_mse(model(Xv), Yv).item()
        scheduler.step(val_loss)

        if verbose:
            if epoch == 0:
                first_by_scenario = mse_by_scenario(model, X_val, Y_val, groups)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  эпоха {epoch + 1}: train = {train_loss:.6f}, val = {val_loss:.6f}, lr = {optimizer.param_groups[0]['lr']:.2e}")

        if callback is not None:
            callback(epoch, val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Ранний останов на эпохе {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        final_by_scenario = mse_by_scenario(model, X_val, Y_val, groups)
        print("\nMSE на критических ситуациях:")
        for name in CRITICAL_SCENARIOS:
            first, final = first_by_scenario[name], final_by_scenario[name]
            print(f"  {name}: {first:.6f} -> {final:.6f} ({final / first - 1:+.0%})")
        print("Обучение завершено.")

    model = model.cpu().eval()
    del Xt, Yt, Xv, Yv
    return model
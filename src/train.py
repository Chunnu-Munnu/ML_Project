"""Train one model.

    python -m src.train --model dsae --split random
    python -m src.train --model dsae --split song --augment

The loss is  gamma * reconstruction + (1 - gamma) * cross-entropy  (Equation 3
of the paper). A model that has no decoder is trained on cross-entropy alone,
and one that has no classifier on reconstruction alone.

--augment is our addition: instead of the paper's 20 fixed clips per song, every
epoch is fitted on new random crops of the training songs (see data.CropSampler).
Because the network can then no longer memorise its training clips, it tolerates
and needs a larger learning rate, so --augment also raises the default from 1e-4
to 1e-3 (src.ablation compares all four combinations).
"""
import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import CropSampler, Dataset, load_dataset, song_split
from .models import build_model


def loss_terms(out, x, y):
    recon = F.mse_loss(out.recon, x) if out.recon is not None else None
    ce = F.cross_entropy(out.logits, y) if out.logits is not None else None
    return recon, ce


def combine(recon, ce, gamma: float):
    if recon is None:
        return ce
    if ce is None:
        return recon
    return gamma * recon + (1 - gamma) * ce


@torch.no_grad()
def evaluate_split(model, X, y, gamma: float) -> dict:
    model.eval()
    out = model(X)
    recon, ce = loss_terms(out, X, y)
    stats = {"loss": combine(recon, ce, gamma).item()}
    if recon is not None:
        stats["recon"] = recon.item()
    if out.logits is not None:
        stats["acc"] = (out.logits.argmax(1) == y).float().mean().item()
    return stats


def train_model(name: str, data: Dataset, cfg: C.TrainConfig, verbose: bool = True,
                sampler: CropSampler | None = None):
    """Return the trained model and its per-epoch history.

    Without a sampler the model is fitted on the fixed training clips of `data`, as in
    the paper. With one, every epoch is fitted on freshly drawn random crops instead.
    The history is measured on the fixed training and dev clips in both cases.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(name, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    X = {k: torch.from_numpy(v) for k, v in data.X.items()}
    y = {k: torch.from_numpy(v) for k, v in data.y.items()}
    X_fit, y_fit = X["train"], y["train"]
    tag = name if sampler is None else f"{name}+crops"
    history = []
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if sampler is not None:
            X_fit, y_fit = (torch.from_numpy(a) for a in sampler.sample())
        n = len(X_fit)
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, cfg.batch_size):
            batch = perm[i:i + cfg.batch_size]
            out = model(X_fit[batch])
            loss = combine(*loss_terms(out, X_fit[batch], y_fit[batch]), cfg.gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        row = {"epoch": epoch}
        for split in ("train", "dev"):
            for key, value in evaluate_split(model, X[split], y[split], cfg.gamma).items():
                row[f"{split}_{key}"] = value
        history.append(row)
        if verbose and (epoch % 50 == 0 or epoch == 1):
            shown = "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch")
            print(f"[{tag}] epoch {epoch:4d}  {shown}  ({time.time() - start:.0f}s)")

    model.eval()
    return model, history


def save_checkpoint(model, name: str, data: Dataset, cfg: C.TrainConfig, history: list,
                    run: str | None = None):
    """Write models/<run>_<split>.pt and results/history_<run>_<split>.json.

    `name` is the architecture; `run` names the files and defaults to the architecture.
    """
    run = run or name
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": name,
        "state_dict": model.state_dict(),
        "scaler": asdict(data.scaler),
        "split": data.split_mode,
        "split_seed": C.SPLIT_SEED,
        "config": asdict(cfg),
    }, C.model_path(run, data.split_mode))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / f"history_{run}_{data.split_mode}.json", "w") as f:
        json.dump(history, f)


def main():
    defaults = C.TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["two_layer", "autoencoder", "dsae"], default="dsae")
    p.add_argument("--split", choices=["random", "song"], default="random")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=None,
                   help=f"learning rate (default: {defaults.lr:g}, or {C.AUGMENT_LR:g} with --augment)")
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--augment", action="store_true",
                   help="fit every epoch on new random crops of the training songs (needs --split song "
                        "and the audio in data/genres); the run is saved as <model>_aug")
    args = p.parse_args()
    if args.augment and args.split != "song":
        p.error("--augment needs --split song: crops are cut from whole training songs, "
                "and in the random split every song also has clips in the test set")

    lr = args.lr if args.lr is not None else C.AUGMENT_LR if args.augment else defaults.lr
    cfg = C.TrainConfig(args.epochs, args.batch_size, lr, args.gamma, args.dropout, args.seed, args.augment)
    data = load_dataset(args.split)   # the split is fixed; --seed only changes the training run
    sampler = CropSampler(song_split("train"), data.scaler, cfg.seed) if args.augment else None
    run = args.model + C.AUGMENT_SUFFIX if args.augment else args.model
    model, history = train_model(args.model, data, cfg, sampler=sampler)
    save_checkpoint(model, args.model, data, cfg, history, run)
    print(f"saved {C.model_path(run, args.split)}")


if __name__ == "__main__":
    main()

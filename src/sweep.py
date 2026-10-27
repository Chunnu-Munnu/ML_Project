"""Sweep the loss weight gamma of the Deep Softmax Autoencoder.

    python -m src.sweep --split song --gammas 0 0.5 0.9 0.99 1

gamma = 1 is pure reconstruction (the classifier head is never trained) and
gamma = 0 is pure classification through the 64-dim bottleneck. Each run writes
one JSON file to results/sweep/, which `python -m src.evaluate` turns into a figure.
"""
import argparse
import json

from . import config as C
from .data import load_dataset
from .evaluate import linear_probe
from .inference import forward
from .train import train_model


def run(split: str, gamma: float, seed: int, epochs: int) -> dict:
    cfg = C.TrainConfig(epochs=epochs, gamma=gamma, seed=seed)
    data = load_dataset(split)
    model, history = train_model("dsae", data, cfg, verbose=False)
    out = {part: forward(model, data.X[part]) for part in data.X}
    return {
        "gamma": gamma,
        "split": split,
        "accuracy": {part: float((out[part]["logits"].argmax(1) == data.y[part]).mean()) for part in data.X},
        "reconstruction_mse": {part: float(((out[part]["recon"] - data.X[part]) ** 2).mean()) for part in data.X},
        "probe_test": linear_probe(out["train"]["z"], data.y["train"], out["test"]["z"], data.y["test"]),
        "best_dev_acc": max(r["dev_acc"] for r in history),
    }


def main():
    defaults = C.TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["random", "song"], default="song")
    p.add_argument("--gammas", type=float, nargs="+", default=[0, 0.5, 0.9, 0.99, 1])
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--seed", type=int, default=defaults.seed)
    args = p.parse_args()

    C.SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for gamma in args.gammas:
        result = run(args.split, gamma, args.seed, args.epochs)
        (C.SWEEP_DIR / f"gamma_{gamma:g}_{args.split}.json").write_text(json.dumps(result, indent=2))
        print(f"gamma={gamma:g}  test acc {result['accuracy']['test']:.3f}  "
              f"probe {result['probe_test']:.3f}  recon mse {result['reconstruction_mse']['test']:.3f}")


if __name__ == "__main__":
    main()

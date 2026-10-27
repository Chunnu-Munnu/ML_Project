"""Ablation of our training change: random crops x learning rate, for both classifiers.

    python -m src.ablation

Trains the Two Layer Neural Network and the Deep Softmax Autoencoder on the song-level
split four times each: with the paper's fixed clips and with random crops, at the
paper's learning rate (1e-4) and at the larger one we use with crops (1e-3). This
separates the effect of the crops from the effect of the learning rate. No model is
saved; the scores go to results/ablation.json, which `python -m src.evaluate` turns
into a table.
"""
import argparse
import json

from . import config as C
from .data import CropSampler, load_dataset, song_split
from .evaluate import song_vote_accuracy
from .inference import forward
from .train import train_model


def run(name: str, augment: bool, lr: float, epochs: int, seed: int) -> dict:
    cfg = C.TrainConfig(epochs=epochs, lr=lr, seed=seed, augment=augment)
    data = load_dataset("song")
    sampler = CropSampler(song_split("train"), data.scaler, seed) if augment else None
    model, history = train_model(name, data, cfg, verbose=False, sampler=sampler)
    logits = {part: forward(model, data.X[part])["logits"] for part in data.X}
    return {
        "model": name,
        "augment": augment,
        "lr": lr,
        "accuracy": {part: float((logits[part].argmax(1) == data.y[part]).mean()) for part in data.X},
        "song_vote_test": song_vote_accuracy(data.y["test"], logits["test"], data.song["test"]),
        "best_dev_acc": max(r["dev_acc"] for r in history),
    }


def main():
    defaults = C.TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--seed", type=int, default=defaults.seed)
    args = p.parse_args()

    results = []
    for name in ("two_layer", "dsae"):
        for augment in (False, True):
            for lr in (defaults.lr, C.AUGMENT_LR):
                results.append(run(name, augment, lr, args.epochs, args.seed))
                a = results[-1]["accuracy"]
                print(f"{name:9s} {'random crops' if augment else 'fixed clips ':12s} lr={lr:g}  "
                      f"train {a['train']:.3f}  dev {a['dev']:.3f}  test {a['test']:.3f}  "
                      f"song vote {results[-1]['song_vote_test']:.3f}")
                C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                C.ABLATION_FILE.write_text(json.dumps(results, indent=2))   # keep what is finished so far


if __name__ == "__main__":
    main()

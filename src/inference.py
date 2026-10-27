"""Load a trained model and run it on audio. Shared by the evaluation, the demo and the CLI.

    python -m src.inference path/to/song.wav
"""
import argparse
from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from .data import Scaler, clip_and_pool, load_audio
from .models import build_model


def load_model(run: str, split: str):
    """Return (model in evaluation mode, the scaler fitted on its training clips).

    `run` is the name the checkpoint was saved under: an architecture such as "dsae",
    or "dsae_aug" for the run trained with random-crop augmentation.
    """
    path = C.model_path(run, split)
    if not path.exists():
        name, augment = run.removesuffix(C.AUGMENT_SUFFIX), " --augment" * run.endswith(C.AUGMENT_SUFFIX)
        raise FileNotFoundError(
            f"{path} not found. Train it with `python -m src.train --model {name} --split {split}{augment}`.")
    ckpt = torch.load(path, weights_only=True)
    # checkpoints written before SPLIT_SEED existed used the training seed for the split
    if ckpt.get("split_seed", ckpt["config"]["seed"]) != C.SPLIT_SEED:
        raise ValueError(f"{path} was trained on a different data split than SPLIT_SEED={C.SPLIT_SEED}; "
                         "its test set would overlap with its training set.")
    model = build_model(ckpt["model"], ckpt["config"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Scaler(**ckpt["scaler"])


@torch.no_grad()
def forward(model, X: np.ndarray) -> dict:
    """Run a model on scaled inputs and return its outputs as numpy arrays."""
    out = model(torch.from_numpy(X))
    return {k: v.numpy() for k, v in out._asdict().items() if v is not None}


@dataclass
class Prediction:
    clips: np.ndarray        # (n, 500) pooled clips in input units
    probs: np.ndarray        # (n, 4) genre probabilities per clip
    encodings: np.ndarray    # (n, 64) bottleneck encodings
    recon: np.ndarray        # (n, 500) reconstructions in input units

    @property
    def song_probs(self) -> np.ndarray:
        return self.probs.mean(axis=0)

    @property
    def genre(self) -> str:
        return C.GENRES[int(self.song_probs.argmax())]


class GenrePredictor:
    """A trained Deep Softmax Autoencoder plus the preprocessing it expects."""

    def __init__(self, run: str = C.BEST_RUN, split: str = "song"):
        self.model, self.scaler = load_model(run, split)

    def predict(self, signal: np.ndarray) -> Prediction:
        clips = clip_and_pool(signal)
        if len(clips) == 0:
            raise ValueError(f"audio is shorter than one clip ({C.CLIP_SAMPLES / C.SAMPLE_RATE:.1f} s)")
        out = forward(self.model, self.scaler.transform(clips))
        return Prediction(
            clips=clips,
            probs=torch.softmax(torch.from_numpy(out["logits"]), dim=1).numpy(),
            encodings=out["z"],
            recon=self.scaler.inverse(out["recon"]),
        )

    def predict_file(self, source) -> Prediction:
        return self.predict(load_audio(source))


def main():
    p = argparse.ArgumentParser(description="Predict the genre of an audio file.")
    p.add_argument("audio")
    p.add_argument("--run", choices=["dsae", "dsae_aug"], default=C.BEST_RUN,
                   help="which trained Deep Softmax Autoencoder to use")
    p.add_argument("--split", choices=["random", "song"], default="song",
                   help="the split that run was trained on")
    args = p.parse_args()

    pred = GenrePredictor(args.run, args.split).predict_file(args.audio)
    print(f"{len(pred.clips)} clips analysed -> {pred.genre}")
    for genre, prob in zip(C.GENRES, pred.song_probs):
        print(f"  {genre:10s} {prob:6.1%}  {'#' * round(40 * prob)}")


if __name__ == "__main__":
    main()

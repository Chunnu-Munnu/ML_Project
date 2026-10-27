"""Latent interpolation: walk in a straight line between two encodings.

    python -m src.morph song_a.wav song_b.wav

The paper leaves this as future work ("interpolate components in our encodings to
interpret the latent space"). Every point  z = (1 - a) * z_A + a * z_B  on the line
between two 64-dim encodings is a valid input for both heads of the Deep Softmax
Autoencoder, so it can be decoded into a waveform and classified. Watching the genre
probabilities along the walk shows whether the latent space is smooth, and the genres
met on the way show which genres the model has placed between which.
"""
import argparse
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from . import config as C


@dataclass
class Morph:
    alphas: np.ndarray       # (steps,) 0 is clip A, 1 is clip B
    encodings: np.ndarray    # (steps, 64) points on the line
    probs: np.ndarray        # (steps, 4) genre probabilities at each point
    recon: np.ndarray        # (steps, 500) decoded waveform at each point, scaled units


@torch.no_grad()
def classify(model, points: np.ndarray, space: str = "latent") -> np.ndarray:
    """Genre probabilities of points of shape (..., d).

    space="latent": the points are 64-dim encodings and go straight to the classifier head.
    space="input":  the points are scaled 500-dim inputs and go through the whole model.
    """
    points = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
    logits = model.classifier(points) if space == "latent" else model(points).logits
    return torch.softmax(logits, dim=-1).numpy()


def line(a: np.ndarray, b: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced points from a to b. Inputs of shape (..., d) give (..., steps, d)."""
    alphas = np.linspace(0, 1, steps, dtype=np.float32)
    t = alphas[:, None]
    return alphas, (1 - t) * a[..., None, :] + t * b[..., None, :]


@torch.no_grad()
def morph(model, z_a: np.ndarray, z_b: np.ndarray, steps: int = 21) -> Morph:
    """Decode and classify `steps` points on the line from one encoding to another."""
    alphas, z = line(np.asarray(z_a, dtype=np.float32), np.asarray(z_b, dtype=np.float32), steps)
    return Morph(alphas, z, classify(model, z), model.decoder(torch.from_numpy(z)).numpy())


def typical_clip(encodings: np.ndarray) -> int:
    """Index of the clip whose encoding lies closest to the song's mean encoding."""
    return int(np.linalg.norm(encodings - encodings.mean(axis=0), axis=1).argmin())


def transitions(model, points: np.ndarray, y: np.ndarray, space: str = "latent",
                n_pairs: int = 1000, steps: int = 21, seed: int = 0) -> dict:
    """Walk between random clips of every pair of genres and summarise what happens.

    With space="latent" the walk is a straight line between two encodings. With
    space="input" it is a straight line between two inputs, which is a plain cross-fade
    of the two pieces of audio and serves as the reference. Only clips the model
    classifies correctly are used as end points, so every walk starts in genre A and
    ends in genre B. The same seed draws the same pairs of clips in both spaces.

    For each pair "A -> B" this returns
        curves:    (steps, 4) genre probabilities averaged over all walks
        direct:    share of walks on which the top genre is only ever A or B
        via:       for each other genre, the share of walks on which it becomes the top genre
        monotone:  share of walks on which the probability of B never drops by more
                   than one percentage point from one step to the next
    """
    rng = np.random.default_rng(seed)
    correct = classify(model, points, space).argmax(axis=1) == y
    genres = range(len(C.GENRES))
    result = {}
    for a, b in combinations(genres, 2):
        starts, ends = (np.flatnonzero(correct & (y == g)) for g in (a, b))
        if len(starts) == 0 or len(ends) == 0:
            continue
        _, paths = line(points[rng.choice(starts, n_pairs)], points[rng.choice(ends, n_pairs)], steps)
        probs = classify(model, paths, space)                   # (n_pairs, steps, 4)
        top = probs.argmax(axis=-1)
        result[f"{C.GENRES[a]} -> {C.GENRES[b]}"] = {
            "curves": probs.mean(axis=0).tolist(),
            "direct": float(np.isin(top, (a, b)).all(axis=1).mean()),
            "via": {C.GENRES[k]: float((top == k).any(axis=1).mean()) for k in genres if k not in (a, b)},
            "monotone": float((np.diff(probs[:, :, b], axis=1) >= -0.01).all(axis=1).mean()),
        }
    return result


def main():
    from .inference import GenrePredictor   # imported here: inference is not needed by the functions above

    p = argparse.ArgumentParser(description="Morph between the most typical clips of two audio files.")
    p.add_argument("audio_a")
    p.add_argument("audio_b")
    p.add_argument("--steps", type=int, default=11)
    args = p.parse_args()

    predictor = GenrePredictor()
    a, b = (predictor.predict_file(f).encodings for f in (args.audio_a, args.audio_b))
    m = morph(predictor.model, a[typical_clip(a)], b[typical_clip(b)], args.steps)
    print("position  " + "  ".join(f"{g:>9s}" for g in C.GENRES) + "  top genre")
    for alpha, probs in zip(m.alphas, m.probs):
        print(f"{alpha:8.0%}  " + "  ".join(f"{q:9.1%}" for q in probs) + f"  {C.GENRES[int(probs.argmax())]}")


if __name__ == "__main__":
    main()

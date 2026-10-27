"""Similar-song search: compare songs by the encodings the model gives them.

    python -m src.retrieval path/to/song.wav

The paper plans to "evaluate our embeddings in an extrinsic task". This is one: if the
64-dim encodings capture musical style, songs of the same genre should lie close to
each other, so the nearest neighbours of a new song should share its genre.

A song's fingerprint is the mean of its clip encodings, centred on the library mean
and scaled to unit length. Two songs are compared with the dot product of their
fingerprints (cosine similarity). Centring removes the large offset that all encodings
share (the tanh outputs average about 0.8 in magnitude), so that the cosine compares
what differs between songs rather than what they have in common.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C


def song_means(features: np.ndarray, song: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average clip features per song. Returns (sorted song ids, (n_songs, d) means)."""
    ids = np.unique(song)
    return ids, np.stack([features[song == s].mean(axis=0) for s in ids])


def fingerprints(means: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Centre and scale song means to unit length, so a dot product is a cosine."""
    v = np.atleast_2d(means) - center
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def similarity(query_means: np.ndarray, library_means: np.ndarray) -> np.ndarray:
    """(n_queries, n_library) cosine similarities, centred on the library mean."""
    center = library_means.mean(axis=0)
    return fingerprints(query_means, center) @ fingerprints(library_means, center).T


def ranking_metrics(sim: np.ndarray, library_y: np.ndarray, query_y: np.ndarray, k: int = 5) -> dict:
    """How often the most similar library songs share the query's genre.

    precision_at_1: the single nearest song has the same genre
    precision_at_k: share of the k nearest songs with the same genre
    map:            mean average precision over the whole ranked library
    """
    order = np.argsort(-sim, axis=1)
    relevant = library_y[order] == query_y[:, None]
    precision = np.cumsum(relevant, axis=1) / (np.arange(relevant.shape[1]) + 1)
    average_precision = (precision * relevant).sum(axis=1) / np.maximum(relevant.sum(axis=1), 1)
    return {"precision_at_1": float(relevant[:, 0].mean()),
            "precision_at_k": float(relevant[:, :k].mean()),
            "map": float(average_precision.mean())}


@dataclass
class Match:
    name: str            # "jazz/jazz.00012.wav", relative to data/genres
    genre: str
    similarity: float


@dataclass
class SongIndex:
    """A searchable library of songs, one mean encoding per song."""
    means: np.ndarray    # (n_songs, 64) mean clip encoding of every library song
    labels: np.ndarray   # (n_songs,) genre index
    names: np.ndarray    # (n_songs,) file name relative to data/genres

    def search(self, clip_encodings: np.ndarray, k: int = 5) -> list[Match]:
        """The k library songs most similar to a song given by its clip encodings."""
        sim = similarity(clip_encodings.mean(axis=0), self.means)[0]
        return [Match(str(self.names[i]), C.GENRES[int(self.labels[i])], float(sim[i]))
                for i in np.argsort(-sim)[:k]]

    def save(self, path: Path):
        np.savez(path, means=self.means, labels=self.labels, names=self.names)

    @classmethod
    def load(cls, path: Path) -> "SongIndex":
        with np.load(path) as d:
            return cls(d["means"], d["labels"], d["names"])


def index_path(split: str = "song") -> Path:
    return C.MODELS_DIR / f"song_index_{split}.npz"


def main():
    from .inference import GenrePredictor   # imported here: only the CLI needs the model

    p = argparse.ArgumentParser(description="List the library songs most similar to an audio file.")
    p.add_argument("audio")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    pred = GenrePredictor().predict_file(args.audio)
    print(f"predicted genre: {pred.genre}")
    for rank, m in enumerate(SongIndex.load(index_path()).search(pred.encodings, args.k), 1):
        print(f"  {rank}. {m.name:28s} {m.genre:10s} similarity {m.similarity:+.3f}")


if __name__ == "__main__":
    main()

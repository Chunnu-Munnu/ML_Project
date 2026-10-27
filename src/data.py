"""Raw audio -> 500-dimensional examples, with no hand-crafted features.

Each song is cut into non-overlapping clips of 20,000 samples and every clip is
average-pooled with a window of 40, exactly as described in the paper.
"""
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config as C


def load_audio(source, allow_headerless: bool = False) -> np.ndarray:
    """Read an audio file (path or file-like object) as mono float32 at 22.05 kHz."""
    try:
        signal, sr = sf.read(source, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        if not allow_headerless:
            raise
        # jazz.00054 ships without a WAV header in the Hugging Face mirror of
        # GTZAN; the payload is still plain 16-bit PCM, so read it as such.
        signal, sr = sf.read(source, dtype="float32", always_2d=True, format="RAW",
                             samplerate=C.SAMPLE_RATE, channels=1, subtype="PCM_16")
    signal = signal.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        g = gcd(C.SAMPLE_RATE, sr)
        signal = resample_poly(signal, C.SAMPLE_RATE // g, sr // g).astype(np.float32)
    return signal


def pool(signal: np.ndarray) -> np.ndarray:
    """Average every POOL_SIZE consecutive samples: 22,050 Hz becomes about 551 Hz."""
    n = len(signal) // C.POOL_SIZE
    return signal[: n * C.POOL_SIZE].reshape(n, C.POOL_SIZE).mean(axis=1)


def clip_and_pool(signal: np.ndarray, max_clips: int | None = None) -> np.ndarray:
    """Cut a waveform into clips and average-pool each one to INPUT_DIM values."""
    n_clips = len(signal) // C.CLIP_SAMPLES
    if max_clips is not None:
        n_clips = min(n_clips, max_clips)
    return pool(signal[: n_clips * C.CLIP_SAMPLES]).reshape(n_clips, C.INPUT_DIM)


def list_songs(genres_dir: Path = C.GENRES_DIR) -> list[tuple[int, Path]]:
    """(genre index, file) for every song; a song's position in this list is its id."""
    songs = []
    for label, genre in enumerate(C.GENRES):
        files = sorted((genres_dir / genre).glob("*.wav")) + sorted((genres_dir / genre).glob("*.au"))
        if not files:
            raise FileNotFoundError(
                f"No audio found in {genres_dir / genre}. Run `python scripts/download_data.py` first.")
        songs.extend((label, f) for f in files)
    return songs


def find_twins(X: np.ndarray, song: np.ndarray, max_shift: float = 0.25, threshold: float = 0.8) -> np.ndarray:
    """For every song, the index of the same recording stored under another name, or -1.

    GTZAN holds a number of recordings twice, some of them shifted by a fraction of a
    second (Sturm, 2013). Two songs count as twins when their loudness envelopes, taken
    in steps of 45 ms, correlate above `threshold` at a shift of at most `max_shift` seconds.
    In the four genres used here the 30 twin pairs score 0.85 or more at shifts of up to
    0.18 s, and no other pair exceeds 0.77.
    """
    block = 25                                             # pooled values per envelope step
    ids = np.unique(song)
    env = np.stack([np.abs(X[song == s]).reshape(-1, block).mean(axis=1) for s in ids])
    env = env - env.mean(axis=1, keepdims=True)
    env /= np.maximum(np.linalg.norm(env, axis=1, keepdims=True), 1e-12)
    n = env.shape[1]
    best = np.full((len(ids), len(ids)), -1.0)
    max_lag = round(max_shift * C.SAMPLE_RATE / (C.POOL_SIZE * block))
    for lag in range(-max_lag, max_lag + 1):               # correlate the overlapping parts
        best = np.maximum(best, env[:, max(lag, 0):n + min(lag, 0)] @ env[:, max(-lag, 0):n + min(-lag, 0)].T)
    np.fill_diagonal(best, -1.0)
    return np.where(best.max(axis=1) > threshold, ids[best.argmax(axis=1)], -1)


def build_dataset(genres_dir: Path = C.GENRES_DIR, cache: Path = C.CACHE_FILE):
    """Return (X, y, song, names, twin) for the four genres, building the cache on first use.

    X: (8000, 500) float32, y: (8000,) genre index, song: (8000,) song index.
    names: (400,) file of every song relative to genres_dir, indexed by song index.
    twin:  (400,) index of a duplicate of each song, or -1 (see find_twins).
    """
    keys = ("X", "y", "song", "names", "twin")
    if cache.exists():
        with np.load(cache) as d:
            if set(keys) <= set(d.files):      # caches written by older versions lack some arrays
                return tuple(d[k] for k in keys)

    X, y, song, names = [], [], [], []
    for song_id, (label, f) in enumerate(list_songs(genres_dir)):
        clips = clip_and_pool(load_audio(f, allow_headerless=True), C.CLIPS_PER_SONG)
        X.append(clips)
        y.extend([label] * len(clips))
        song.extend([song_id] * len(clips))
        names.append(f"{C.GENRES[label]}/{f.name}")
    X = np.concatenate(X).astype(np.float32)
    y, song, names = np.array(y, dtype=np.int64), np.array(song, dtype=np.int64), np.array(names)
    twin = find_twins(X, song)
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(cache.stem + ".partial.npz")   # never leave a half-written cache behind
    np.savez_compressed(partial, X=X, y=y, song=song, names=names, twin=twin)
    partial.replace(cache)
    return X, y, song, names, twin


def split_indices(y: np.ndarray, song: np.ndarray, mode: str, seed: int, twin: np.ndarray | None = None):
    """Indices for train / dev / test.

    random: clips are shuffled individually (the paper's protocol). Clips from
            one song can land in both train and test.
    song:   whole songs are assigned to one split, 75 / 12-13 / 12-13 per genre,
            so the test set only contains songs the model has never heard. Songs
            that have a twin (`twin[song index] >= 0`) always go to the training
            set; otherwise one copy could be trained on and the other tested.
    """
    rng = np.random.default_rng(seed)
    n_train, n_dev, _ = C.SPLIT_SIZES
    if mode == "random":
        perm = rng.permutation(len(y))
        return perm[:n_train], perm[n_train:n_train + n_dev], perm[n_train + n_dev:]
    if mode == "song":
        parts = ([], [], [])
        for label in np.unique(y):
            songs = rng.permutation(np.unique(song[y == label]))
            if twin is not None:
                songs = np.concatenate([songs[twin[songs] >= 0], songs[twin[songs] < 0]])
            n_tr = round(len(songs) * 0.75)
            n_dv = (len(songs) - n_tr + label % 2) // 2   # alternate the odd song out
            for part, ids in zip(parts, (songs[:n_tr], songs[n_tr:n_tr + n_dv], songs[n_tr + n_dv:])):
                part.append(np.flatnonzero(np.isin(song, ids)))
        return tuple(rng.permutation(np.concatenate(p)) for p in parts)
    raise ValueError(f"unknown split mode: {mode}")


def song_split(part: str, seed: int = C.SPLIT_SEED) -> list[tuple[int, Path]]:
    """(genre index, file) of the songs in one part of the song-level split."""
    _, y, song, names, twin = build_dataset()
    idx = split_indices(y, song, "song", seed, twin)[("train", "dev", "test").index(part)]
    ids, first = np.unique(song[idx], return_index=True)
    return [(int(label), C.GENRES_DIR / str(names[s])) for s, label in zip(ids, y[idx][first])]


def held_out_songs(seed: int = C.SPLIT_SEED) -> list[tuple[int, Path]]:
    """The test songs of the song-level split: the model has never heard them."""
    return song_split("test", seed)


@dataclass
class Scaler:
    """One global mean and standard deviation, fitted on the training clips.

    A single scalar pair (rather than one per feature) keeps the relative
    loudness of clips intact while bringing the inputs to unit scale.
    """
    mean: float
    std: float

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        return cls(float(X.mean()), float(X.std()))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean


class CropSampler:
    """Random-crop augmentation: new clips from the training songs for every epoch.

    The paper cuts each song into 20 fixed clips, so the network meets the same 6,000
    waveforms in every epoch and can simply memorise them. Each call to sample() instead
    cuts 20 windows per song at random positions anywhere in its 30 seconds, so no clip
    is seen twice while the number of gradient steps per epoch stays the same.

    Crops start on a multiple of POOL_SIZE samples, which lets every song be pooled once.
    """

    def __init__(self, songs: list[tuple[int, Path]], scaler: Scaler, seed: int):
        self.labels = np.array([label for label, _ in songs], dtype=np.int64)
        self.pooled = [pool(load_audio(f, allow_headerless=True)) for _, f in songs]
        self.scaler = scaler
        self.rng = np.random.default_rng(seed)

    def sample(self, clips_per_song: int = C.CLIPS_PER_SONG) -> tuple[np.ndarray, np.ndarray]:
        """Scaled crops of shape (n_songs * clips_per_song, INPUT_DIM) and their genres."""
        crops = [p[t:t + C.INPUT_DIM] for p in self.pooled
                 for t in self.rng.integers(len(p) - C.INPUT_DIM + 1, size=clips_per_song)]
        return self.scaler.transform(np.stack(crops)), np.repeat(self.labels, clips_per_song)


@dataclass
class Dataset:
    """Scaled splits plus the bookkeeping needed for song-level evaluation."""
    X: dict          # split name -> (n, 500) float32, scaled
    y: dict          # split name -> (n,) int64
    song: dict       # split name -> (n,) int64
    scaler: Scaler
    split_mode: str
    names: np.ndarray   # (n_songs,) "genre/file.wav" of every song index


def load_dataset(split_mode: str = "random", seed: int = C.SPLIT_SEED) -> Dataset:
    """Build, split and scale the data. The split depends only on `seed`, never on
    the training seed, so every model is tested on the same held-out examples."""
    X, y, song, names, twin = build_dataset()
    idx = dict(zip(("train", "dev", "test"), split_indices(y, song, split_mode, seed, twin)))
    scaler = Scaler.fit(X[idx["train"]])
    return Dataset(
        X={k: scaler.transform(X[i]) for k, i in idx.items()},
        y={k: y[i] for k, i in idx.items()},
        song={k: song[i] for k, i in idx.items()},
        scaler=scaler,
        split_mode=split_mode,
        names=names,
    )

"""Fast checks that need neither the dataset nor trained weights.

    python -m pytest
"""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from src import config as C
from src.data import CropSampler, Dataset, Scaler, clip_and_pool, find_twins, pool, split_indices
from src.inference import forward, load_model
from src.models import build_model
from src.morph import classify, line, morph, transitions, typical_clip
from src.retrieval import SongIndex, ranking_metrics, similarity, song_means
from src.train import combine, loss_terms, save_checkpoint, train_model


def fake_labels():
    """Same layout as the real dataset: 4 genres x 100 songs x 20 clips."""
    song = np.repeat(np.arange(400), C.CLIPS_PER_SONG)
    return song // 100, song


def fake_dataset(n_train: int = 64, n_other: int = 32) -> Dataset:
    """Random numbers in the shape of the real data, enough to run a training step."""
    rng = np.random.default_rng(0)
    sizes = {"train": n_train, "dev": n_other, "test": n_other}
    return Dataset(
        X={k: rng.normal(size=(n, C.INPUT_DIM)).astype(np.float32) for k, n in sizes.items()},
        y={k: rng.integers(0, len(C.GENRES), n) for k, n in sizes.items()},
        song={k: np.arange(n) for k, n in sizes.items()},
        scaler=Scaler(0.0, 1.0), split_mode="song", names=np.array([]))


# ---------------------------------------------------------------- data
def test_clip_and_pool_shape_and_values():
    signal = np.arange(2 * C.CLIP_SAMPLES + 123, dtype=np.float32)
    clips = clip_and_pool(signal)
    assert clips.shape == (2, C.INPUT_DIM)
    # the first pooled value is the mean of the first 40 raw samples
    assert clips[0, 0] == pytest.approx(signal[:C.POOL_SIZE].mean())
    assert clip_and_pool(signal, max_clips=1).shape == (1, C.INPUT_DIM)
    assert clip_and_pool(signal[:100]).shape == (0, C.INPUT_DIM)
    # a clip is nothing but a stretch of the pooled song
    assert np.array_equal(clips[1], pool(signal)[C.INPUT_DIM:2 * C.INPUT_DIM])


@pytest.mark.parametrize("mode", ["random", "song"])
def test_splits_partition_the_data(mode):
    y, song = fake_labels()
    parts = split_indices(y, song, mode, seed=0)
    assert tuple(len(p) for p in parts) == C.SPLIT_SIZES
    assert len(np.unique(np.concatenate(parts))) == len(y)


def test_song_split_keeps_songs_together():
    y, song = fake_labels()
    train, dev, test = (set(song[p]) for p in split_indices(y, song, "song", seed=0))
    assert not (train & dev or train & test or dev & test)
    # every genre is equally represented in the training set
    assert np.bincount(y[split_indices(y, song, "song", seed=0)[0]]).tolist() == [1500] * 4


def test_twins_are_found():
    rng = np.random.default_rng(0)
    song = np.repeat(np.arange(6), C.CLIPS_PER_SONG)
    # six "songs" of noise whose loudness rises and falls in a pattern of their own
    loudness = np.repeat(rng.uniform(0.1, 1.0, size=(6, 40)), 250, axis=1)
    waves = rng.normal(size=loudness.shape) * loudness
    waves[4] = np.roll(waves[1], 25)             # song 4 is song 1 again, 45 ms late
    twin = find_twins(waves.reshape(-1, C.INPUT_DIM).astype(np.float32), song)
    assert twin.tolist() == [-1, 4, -1, -1, 1, -1]


def test_song_split_keeps_twins_in_training():
    y, song = fake_labels()
    twin = np.full(400, -1)
    for a, b in ((10, 20), (150, 160), (399, 301)):
        twin[a], twin[b] = b, a
    parts = split_indices(y, song, "song", seed=0, twin=twin)
    assert tuple(len(p) for p in parts) == C.SPLIT_SIZES
    assert set(np.flatnonzero(twin >= 0)) <= set(song[parts[0]])
    assert np.bincount(y[parts[0]]).tolist() == [1500] * 4
    # without twins the split is the one we had before
    plain, no_twins = split_indices(y, song, "song", seed=0), split_indices(y, song, "song", seed=0, twin=np.full(400, -1))
    assert all(np.array_equal(a, b) for a, b in zip(plain, no_twins))


def test_scaler_round_trip():
    X = np.random.default_rng(0).normal(3, 5, size=(50, C.INPUT_DIM)).astype(np.float32)
    scaler = Scaler.fit(X)
    Z = scaler.transform(X)
    assert Z.mean() == pytest.approx(0, abs=1e-4) and Z.std() == pytest.approx(1, abs=1e-4)
    assert np.allclose(scaler.inverse(Z), X, atol=1e-4)


def test_crop_sampler_cuts_random_windows_of_each_song(tmp_path):
    # two "songs" that are ramps, so a crop reveals the position it was cut from
    songs = []
    for label, slope in ((0, 1.0), (2, -1.0)):
        path = tmp_path / f"song{label}.wav"
        sf.write(path, slope * np.linspace(0, 1, 3 * C.CLIP_SAMPLES, dtype=np.float32), C.SAMPLE_RATE, subtype="FLOAT")
        songs.append((label, path))
    sampler = CropSampler(songs, Scaler(0.0, 1.0), seed=0)
    X, y = sampler.sample(clips_per_song=5)
    assert X.shape == (10, C.INPUT_DIM) and y.tolist() == [0] * 5 + [2] * 5
    for crop, pooled in zip(X, np.repeat(np.arange(2), 5)):
        whole = sampler.pooled[pooled]
        start = int(np.abs(whole - crop[0]).argmin())
        assert np.allclose(crop, whole[start:start + C.INPUT_DIM])
    assert not np.array_equal(X, sampler.sample(clips_per_song=5)[0])   # new crops on every call


# ---------------------------------------------------------------- models and training
@pytest.mark.parametrize("name,has_logits,has_recon", [
    ("two_layer", True, False), ("autoencoder", False, True), ("dsae", True, True)])
def test_model_outputs(name, has_logits, has_recon):
    x = torch.randn(8, C.INPUT_DIM)
    out = build_model(name)(x)
    assert (out.logits is not None) == has_logits and (out.recon is not None) == has_recon
    if has_logits:
        assert out.logits.shape == (8, len(C.GENRES))
    if has_recon:
        assert out.recon.shape == x.shape and out.z.shape == (8, C.ENCODING_DIM)


def test_loss_is_the_weighted_sum_from_the_paper():
    x, y = torch.randn(8, C.INPUT_DIM), torch.randint(0, 4, (8,))
    recon, ce = loss_terms(build_model("dsae").eval()(x), x, y)
    assert combine(recon, ce, 0.9).item() == pytest.approx(0.9 * recon.item() + 0.1 * ce.item())
    assert combine(None, ce, 0.9) is ce and combine(recon, None, 0.9) is recon


def test_training_fits_on_fresh_crops_when_given_a_sampler():
    data = fake_dataset()
    cfg = C.TrainConfig(epochs=3, batch_size=32)
    _, history = train_model("dsae", data, cfg, verbose=False)
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    calls = []

    class Sampler:
        def sample(self):
            calls.append(1)
            return data.X["dev"], data.y["dev"]

    _, augmented = train_model("dsae", data, cfg, verbose=False, sampler=Sampler())
    assert len(calls) == cfg.epochs
    assert augmented[-1]["train_loss"] != history[-1]["train_loss"]   # it was fitted on other data


def test_checkpoint_round_trip_and_split_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    data = fake_dataset()
    cfg = C.TrainConfig(epochs=1, batch_size=32, augment=True)
    model, history = train_model("dsae", data, cfg, verbose=False)
    save_checkpoint(model, "dsae", data, cfg, history, run="dsae_aug")

    loaded, scaler = load_model("dsae_aug", "song")
    assert scaler == data.scaler
    assert np.allclose(forward(loaded, data.X["test"])["logits"], forward(model, data.X["test"])["logits"])
    with pytest.raises(FileNotFoundError, match="--augment"):
        load_model("two_layer_aug", "song")
    # a model trained on another split must not be evaluated on this one
    monkeypatch.setattr(C, "SPLIT_SEED", C.SPLIT_SEED + 1)
    with pytest.raises(ValueError, match="different data split"):
        load_model("dsae_aug", "song")


# ---------------------------------------------------------------- latent interpolation
class GenresOnALine(torch.nn.Module):
    """A toy classifier head: the four genres sit at 0, 1, 2 and 3 on a single axis."""

    def forward(self, z):
        return -20 * (z - torch.arange(4.0)) ** 2


def test_line_and_typical_clip():
    alphas, points = line(np.zeros(3, dtype=np.float32), np.full(3, 2, dtype=np.float32), steps=5)
    assert alphas.tolist() == [0, 0.25, 0.5, 0.75, 1] and points[:, 0].tolist() == [0, 0.5, 1, 1.5, 2]
    assert line(np.zeros((7, 3)), np.ones((7, 3)), steps=5)[1].shape == (7, 5, 3)
    assert typical_clip(np.array([[0.0, 0], [1, 1], [10, 10]])) == 1


def test_morph_ends_at_the_two_clips():
    model = build_model("dsae").eval()
    z = forward(model, np.random.default_rng(0).normal(size=(2, C.INPUT_DIM)).astype(np.float32))["z"]
    m = morph(model, z[0], z[1], steps=7)
    assert m.probs.shape == (7, len(C.GENRES)) and m.recon.shape == (7, C.INPUT_DIM)
    assert np.allclose(m.probs.sum(axis=1), 1, atol=1e-5)
    assert np.allclose(m.encodings[[0, -1]], z, atol=1e-6)
    assert np.allclose(m.probs[[0, -1]], classify(model, z), atol=1e-5)


def test_transitions_find_the_genres_in_between():
    model = SimpleNamespace(classifier=GenresOnALine())
    y = np.repeat(np.arange(4), 5)
    points = y[:, None].astype(np.float32)      # every clip sits exactly on its genre
    walks = transitions(model, points, y, n_pairs=20, steps=21)
    assert len(walks) == 6
    for pair, t in walks.items():
        a, b = (C.GENRES.index(g) for g in pair.split(" -> "))
        between = {C.GENRES[k] for k in range(a + 1, b)}
        assert t["direct"] == (0.0 if between else 1.0)
        assert {g for g, share in t["via"].items() if share > 0} == between
        assert t["monotone"] == 1.0
        assert np.allclose(np.sum(t["curves"], axis=1), 1, atol=1e-5)


# ---------------------------------------------------------------- similar-song search
def test_song_means():
    ids, means = song_means(np.array([[1.0], [3.0], [10.0]]), np.array([5, 5, 2]))
    assert ids.tolist() == [2, 5] and means.ravel().tolist() == [10.0, 2.0]


def test_ranking_metrics_on_a_known_ranking():
    # ranked library genres for the single query of genre 0: 0, 1, 1, 0
    sim = np.array([[0.9, 0.1, 0.8, 0.2]])
    m = ranking_metrics(sim, library_y=np.array([0, 0, 1, 1]), query_y=np.array([0]), k=2)
    assert m["precision_at_1"] == 1.0 and m["precision_at_k"] == 0.5
    assert m["map"] == pytest.approx((1 / 1 + 2 / 4) / 2)


def test_song_index_finds_the_same_song_first(tmp_path):
    rng = np.random.default_rng(0)
    means = rng.normal(size=(12, C.ENCODING_DIM)).astype(np.float32)
    index = SongIndex(means, labels=np.arange(12) % 4, names=np.array([f"song{i}.wav" for i in range(12)]))
    sim = similarity(means, means)
    assert np.allclose(np.diag(sim), 1, atol=1e-5) and (sim.argmax(axis=1) == np.arange(12)).all()

    clips_of_song_7 = means[7] + 0.01 * rng.normal(size=(20, C.ENCODING_DIM))
    matches = index.search(clips_of_song_7, k=3)
    assert [m.name for m in matches][0] == "song7.wav" and matches[0].genre == C.GENRES[7 % 4]
    assert matches[0].similarity > matches[1].similarity > matches[2].similarity

    index.save(tmp_path / "index.npz")
    again = SongIndex.load(tmp_path / "index.npz")
    assert [m.name for m in again.search(clips_of_song_7, k=3)] == [m.name for m in matches]

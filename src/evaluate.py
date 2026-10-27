"""Evaluate the trained models and regenerate every table and figure.

    python -m src.evaluate

Writes results/metrics.json, results/summary.md and results/figures/*.png, plus the
two kinds of file the demo reads: models/latent_map_<run>_<split>.npz and
models/song_index_song.npz.
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import plots
from .data import Dataset, load_dataset
from .inference import forward, load_model
from .models import DISPLAY_NAMES, SHORT_NAMES
from .morph import transitions
from .retrieval import SongIndex, index_path, ranking_metrics, similarity, song_means

SPLITS = ("random", "song")
SPLIT_NAMES = {"random": "random clip split (paper)", "song": "song-level split (ours)"}
# Every run evaluated on each split. Random-crop training only exists for the song split.
RUNS = {"random": ("two_layer", "autoencoder", "dsae"),
        "song": ("two_layer", "two_layer_aug", "autoencoder", "dsae", "dsae_aug")}
FIGURE_RUNS = ("two_layer", "autoencoder", "dsae", "dsae_aug")   # runs that get a panel in the figures
RETRIEVAL_K = 5

# Numbers reported in the paper, for side-by-side comparison.
PAPER = {
    "accuracy": {
        "two_layer": {"train": 0.520, "dev": 0.381, "test": 0.364},
        "dsae": {"train": 0.949, "dev": 0.641, "test": 0.653},
    },
    "per_class": {
        "classical": {"precision": 0.783, "recall": 0.775, "f1": 0.779},
        "jazz": {"precision": 0.606, "recall": 0.627, "f1": 0.616},
        "metal": {"precision": 0.515, "recall": 0.554, "f1": 0.534},
        "pop": {"precision": 0.670, "recall": 0.608, "f1": 0.638},
    },
}


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    labels = list(range(len(C.GENRES)))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "accuracy": float((y == pred).mean()),
        "per_class": {g: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i, g in enumerate(C.GENRES)},
        "confusion": confusion_matrix(y, pred, labels=labels, normalize="true").tolist(),
    }


def song_vote_accuracy(y: np.ndarray, logits: np.ndarray, song: np.ndarray) -> float:
    """Average the clip probabilities of each song and classify the song once."""
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    songs = np.unique(song)
    votes = np.array([probs[song == s].mean(axis=0).argmax() for s in songs])
    truth = np.array([y[song == s][0] for s in songs])
    return float((votes == truth).mean())


def linear_probe(train_feats, y_train, test_feats, y_test) -> float:
    """Test accuracy of logistic regression on fixed features: how linearly
    readable is the genre from this representation?"""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(train_feats, y_train)
    return float(clf.score(test_feats, y_test))


def loudness(X: np.ndarray) -> np.ndarray:
    """A single hand-made feature per clip: log of its RMS amplitude."""
    return np.log(X.std(axis=1, keepdims=True) + 1e-6)


def encoding_label(run: str) -> str:
    return f"{SHORT_NAMES[run]} encoding (64-dim)"


def retrieval_metrics(data: Dataset, outputs: dict) -> tuple[dict, SongIndex]:
    """Similar-song search: the training songs are the library, and every song the
    models never trained on (dev and test) is used once as a query.

    Returns the scores of each representation and the library of the best run.
    """
    unseen = ("dev", "test")
    query_song = np.concatenate([data.song[p] for p in unseen])
    query_y = np.concatenate([data.y[p] for p in unseen])
    library_ids, first = np.unique(data.song["train"], return_index=True)
    library_y = data.y["train"][first]
    query_y = query_y[np.unique(query_song, return_index=True)[1]]

    def per_song(features: dict) -> tuple[np.ndarray, np.ndarray]:
        queries = np.concatenate([features[p] for p in unseen])
        return song_means(queries, query_song)[1], song_means(features["train"], data.song["train"])[1]

    # loudness is one number per song, so "similar" simply means "about as loud"
    q, lib = per_song({p: loudness(data.X[p]) for p in data.X})
    sims = {"loudness (1-dim, hand-made)": -np.abs(q - lib.T)}
    for run in RUNS[data.split_mode]:
        if "z" in outputs[run]["train"]:
            sims[encoding_label(run)] = similarity(*per_song({p: outputs[run][p]["z"] for p in data.X}))
    sims[f"{SHORT_NAMES[C.BEST_RUN]} logits (4-dim)"] = similarity(
        *per_song({p: outputs[C.BEST_RUN][p]["logits"] for p in data.X}))

    best_library = song_means(outputs[C.BEST_RUN]["train"]["z"], data.song["train"])[1]
    metrics = {"k": RETRIEVAL_K, "n_library": int(len(library_y)), "n_queries": int(len(query_y)),
               "scores": {name: ranking_metrics(sim, library_y, query_y, RETRIEVAL_K) for name, sim in sims.items()}}
    return metrics, SongIndex(best_library, library_y, data.names[library_ids])


def evaluate_split(data: Dataset) -> tuple[dict, dict]:
    """Return (metrics, outputs) for every run trained on this split."""
    split = data.split_mode
    models = {run: load_model(run, split)[0] for run in RUNS[split]}
    outputs = {run: {part: forward(model, data.X[part]) for part in data.X} for run, model in models.items()}
    classifiers = [run for run in RUNS[split] if "logits" in outputs[run]["test"]]
    autoencoders = [run for run in RUNS[split] if "recon" in outputs[run]["test"]]
    metrics = {"n": {part: int(len(v)) for part, v in data.y.items()},
               "accuracy": {}, "per_class": {}, "confusion": {}, "reconstruction_mse": {}, "best_dev": {}}

    for run in classifiers:
        metrics["accuracy"][run] = {}
        for part in data.X:
            m = classification_metrics(data.y[part], outputs[run][part]["logits"].argmax(1))
            metrics["accuracy"][run][part] = m["accuracy"]
            if part == "test":
                metrics["per_class"][run] = m["per_class"]
                metrics["confusion"][run] = m["confusion"]
        history = json.loads((C.RESULTS_DIR / f"history_{run}_{split}.json").read_text())
        best = max(history, key=lambda r: r["dev_acc"])
        metrics["best_dev"][run] = {"epoch": best["epoch"], "dev_acc": best["dev_acc"]}

    for run in autoencoders:
        metrics["reconstruction_mse"][run] = {
            part: float(((outputs[run][part]["recon"] - data.X[part]) ** 2).mean()) for part in data.X}

    probes = {
        "loudness (1-dim, hand-made)": (loudness(data.X["train"]), loudness(data.X["test"])),
        "raw input (500-dim)": (data.X["train"], data.X["test"]),
        **{encoding_label(run): (outputs[run]["train"]["z"], outputs[run]["test"]["z"]) for run in autoencoders},
    }
    metrics["linear_probe_test"] = {
        k: linear_probe(tr, data.y["train"], te, data.y["test"]) for k, (tr, te) in probes.items()}

    if split == "song":   # these three need whole songs that the models have never heard
        metrics["song_vote_test"] = {
            run: song_vote_accuracy(data.y["test"], outputs[run]["test"]["logits"], data.song["test"])
            for run in classifiers}
        metrics["n_test_songs"] = int(len(np.unique(data.song["test"])))
        best = models[C.BEST_RUN]
        metrics["interpolation"] = {
            "latent": transitions(best, outputs[C.BEST_RUN]["test"]["z"], data.y["test"], "latent"),
            "input": transitions(best, data.X["test"], data.y["test"], "input")}
        metrics["retrieval"], index = retrieval_metrics(data, outputs)
        index.save(index_path(split))
    return metrics, outputs


def make_figures(data: Dataset, metrics: dict, outputs: dict):
    split = data.split_mode
    y = data.y["test"]
    shown = [run for run in FIGURE_RUNS if run in outputs]
    classifiers = [run for run in shown if "logits" in outputs[run]["test"]]
    autoencoders = [run for run in shown if "recon" in outputs[run]["test"]]
    plots.pca_panels([("Raw 500-dim input", data.X["train"], data.X["test"])], y, f"pca_raw_{split}")
    plots.pca_panels(
        [(f"{SHORT_NAMES[run]}: 64-dim encoding", outputs[run]["train"]["z"], outputs[run]["test"]["z"])
         for run in autoencoders], y, f"pca_encodings_{split}")
    plots.pca_panels(
        [(f"{SHORT_NAMES[run]}: 4-dim logits", outputs[run]["train"]["logits"], outputs[run]["test"]["logits"])
         for run in classifiers], y, f"pca_logits_{split}")
    plots.accuracy_curves(
        {SHORT_NAMES[run]: json.loads((C.RESULTS_DIR / f"history_{run}_{split}.json").read_text())
         for run in classifiers}, f"accuracy_curves_{split}")
    for run in classifiers:
        if run != "two_layer":
            plots.confusion_matrix(metrics["confusion"][run], f"{SHORT_NAMES[run]}, test set",
                                   f"confusion_{run}_{split}")
    plots.reconstructions(
        data.scaler.inverse(data.X["test"]),
        {SHORT_NAMES[run]: data.scaler.inverse(outputs[run]["test"]["recon"]) for run in autoencoders},
        y, f"reconstructions_{split}")

    if "interpolation" in metrics:
        walks = metrics["interpolation"]
        plots.transition_curves(walks["latent"], f"interpolation_{split}")
        plots.dot_comparison(
            [(pair.replace(" -> ", " to "), {space: walks[space][pair]["monotone"] for space in walks})
             for pair in walks["latent"]],
            {"latent": ("walk in the latent space", plots.TRAIN_COLOR),
             "input": ("cross-fade of the two inputs", plots.DEV_COLOR)},
            "Walks on which the target genre gains steadily", f"interpolation_steady_{split}",
            xlabel="share of walks", xlim=(0, 1.08))
    if "retrieval" in metrics:
        r = metrics["retrieval"]
        plots.dot_comparison(
            [(name, {"top1": s["precision_at_1"], "topk": s["precision_at_k"]}) for name, s in r["scores"].items()],
            {"top1": ("the most similar song", plots.TRAIN_COLOR),
             "topk": (f"the {r['k']} most similar songs", plots.DEV_COLOR)},
            f"Similar-song search: results sharing the query's genre ({r['n_queries']} queries)",
            f"retrieval_{split}", chance=1 / len(C.GENRES), xlabel="share of results", xlim=(0.15, 1.0))


def make_overview_figures(all_metrics: dict):
    """Figures that put both splits side by side."""
    columns = {"random": (SPLIT_NAMES["random"], plots.TRAIN_COLOR), "song": (SPLIT_NAMES["song"], plots.DEV_COLOR)}
    runs = [run for run in RUNS["song"] if run in all_metrics["song"]["accuracy"]]
    plots.dot_comparison(
        [(SHORT_NAMES[run], {"paper": PAPER["accuracy"].get(run, {}).get("test"),
                               **{s: all_metrics[s]["accuracy"].get(run, {}).get("test") for s in SPLITS}})
         for run in runs],
        {"paper": ("reported in the paper", plots.MUTED), **columns},
        "Test accuracy of the classifiers", "accuracy_overview", chance=1 / len(C.GENRES))
    plots.dot_comparison(
        [(name, {s: all_metrics[s]["linear_probe_test"].get(name) for s in SPLITS})
         for name in all_metrics["song"]["linear_probe_test"]],
        columns, "Linear probe: how much genre each representation exposes", "linear_probe",
        chance=1 / len(C.GENRES))
    sweep = sorted((json.loads(f.read_text()) for f in C.SWEEP_DIR.glob("gamma_*_song.json")),
                   key=lambda r: r["gamma"])
    if sweep:
        plots.gamma_sweep(sweep, "gamma_sweep_song")
    return sweep


def save_latent_map(data: Dataset, outputs: dict, run: str):
    """Store the 2-D PCA of a run's encodings so the demo can place new clips on it."""
    points, var, pca = plots.project(outputs[run]["train"]["z"], outputs[run]["test"]["z"])
    np.savez(C.MODELS_DIR / f"latent_map_{run}_{data.split_mode}.npz", mean=pca.mean_,
             components=pca.components_, explained=var, points=points.astype(np.float32), labels=data.y["test"])


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def summary_markdown(all_metrics: dict) -> str:
    sweep, ablation = all_metrics["sweep"], all_metrics["ablation"]
    lines = ["# Results", "", "Generated by `python -m src.evaluate`. All numbers are from the final epoch.", ""]
    lines += ["## Classification accuracy", "",
              "| Model | Split | Train | Dev | Test |", "|---|---|---|---|---|"]
    for run in RUNS["song"]:
        if run not in all_metrics["song"]["accuracy"]:
            continue
        if run in PAPER["accuracy"]:
            a = PAPER["accuracy"][run]
            lines.append(f"| {DISPLAY_NAMES[run]} | paper (reported) | {pct(a['train'])} | {pct(a['dev'])} | {pct(a['test'])} |")
        for split in SPLITS:
            a = all_metrics[split]["accuracy"].get(run)
            if a:
                lines.append(f"| {DISPLAY_NAMES[run]} | {SPLIT_NAMES[split]} | {pct(a['train'])} | {pct(a['dev'])} | {pct(a['test'])} |")
    for split in SPLITS:
        m = all_metrics[split]
        for run in m["per_class"]:
            if run.startswith("two_layer"):
                continue
            lines += ["", f"## {DISPLAY_NAMES[run]} per genre, test set, {SPLIT_NAMES[split]}", "",
                      "| Genre | Precision | Recall | F1 | Paper F1 |", "|---|---|---|---|---|"]
            for g in C.GENRES:
                c = m["per_class"][run][g]
                lines.append(f"| {g} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {PAPER['per_class'][g]['f1']:.3f} |")
        lines += ["", f"## Linear probe on fixed features, test accuracy, {SPLIT_NAMES[split]}", "",
                  "| Features | Accuracy |", "|---|---|"]
        lines += [f"| {k} | {pct(v)} |" for k, v in m["linear_probe_test"].items()]
        lines += ["", f"## Reconstruction error (MSE on scaled inputs), {SPLIT_NAMES[split]}", "",
                  "| Model | Train | Dev | Test |", "|---|---|---|---|"]
        for run, r in m["reconstruction_mse"].items():
            lines.append(f"| {DISPLAY_NAMES[run]} | {r['train']:.3f} | {r['dev']:.3f} | {r['test']:.3f} |")
        if "song_vote_test" in m:
            lines += ["", f"## Whole-song prediction by averaging clip probabilities ({m['n_test_songs']} unseen test songs)", "",
                      "| Model | Clip accuracy | Song accuracy |", "|---|---|---|"]
            for run, v in m["song_vote_test"].items():
                lines.append(f"| {DISPLAY_NAMES[run]} | {pct(m['accuracy'][run]['test'])} | {pct(v)} |")
        if "interpolation" in m:
            walks = m["interpolation"]
            lines += ["", f"## Latent interpolation with the {DISPLAY_NAMES[C.BEST_RUN]}, {SPLIT_NAMES[split]}", "",
                      "1,000 walks per pair of genres between correctly classified test clips. "
                      "Direct: the top genre is only ever one of the two end genres. "
                      "Steady: the probability of the target genre never drops by more than one point.", "",
                      "| Walk | Direct (latent) | Direct (input cross-fade) | Steady (latent) | Steady (input cross-fade) | Most common detour (latent) |",
                      "|---|---|---|---|---|---|"]
            for pair, t in walks["latent"].items():
                u = walks["input"][pair]
                via, share = max(t["via"].items(), key=lambda item: item[1])
                detour = f"{via} on {pct(share)} of walks" if share >= 0.05 else "none"
                lines.append(f"| {pair.replace(' -> ', ' to ')} | {pct(t['direct'])} | {pct(u['direct'])} | "
                             f"{pct(t['monotone'])} | {pct(u['monotone'])} | {detour} |")
            mean = lambda space, key: float(np.mean([t[key] for t in walks[space].values()]))
            lines.append(f"| **average** | {pct(mean('latent', 'direct'))} | {pct(mean('input', 'direct'))} | "
                         f"{pct(mean('latent', 'monotone'))} | {pct(mean('input', 'monotone'))} | |")
        if "retrieval" in m:
            r = m["retrieval"]
            lines += ["", f"## Similar-song search, {SPLIT_NAMES[split]}", "",
                      f"Library: the {r['n_library']} training songs. Queries: the {r['n_queries']} dev and test songs. "
                      f"A result counts as correct when it has the query's genre; chance is {pct(1 / len(C.GENRES))}.", "",
                      f"| Song fingerprint built from | Top-1 | Precision@{r['k']} | Mean average precision |", "|---|---|---|---|"]
            for name, s in r["scores"].items():
                lines.append(f"| {name} | {pct(s['precision_at_1'])} | {pct(s['precision_at_k'])} | {s['map']:.3f} |")
    if sweep:
        lines += ["", f"## Loss weight gamma, Deep Softmax Autoencoder, {SPLIT_NAMES['song']}", "",
                  "| gamma | Test accuracy | Linear probe on encoding | Test reconstruction MSE |", "|---|---|---|---|"]
        for r in sweep:
            lines.append(f"| {r['gamma']:g} | {pct(r['accuracy']['test'])} | {pct(r['probe_test'])} | "
                         f"{r['reconstruction_mse']['test']:.3f} |")
    if ablation:
        lines += ["", f"## Ablation: random crops and learning rate, {SPLIT_NAMES['song']}", "",
                  "| Model | Training clips | Learning rate | Train | Dev | Test | Whole-song test |",
                  "|---|---|---|---|---|---|---|"]
        for r in ablation:
            a = r["accuracy"]
            lines.append(f"| {DISPLAY_NAMES[r['model']]} | {'random crops (ours)' if r['augment'] else 'fixed (paper)'} | "
                         f"{r['lr']:g} | {pct(a['train'])} | {pct(a['dev'])} | {pct(a['test'])} | {pct(r['song_vote_test'])} |")
    return "\n".join(lines) + "\n"


def main():
    all_metrics = {}
    for split in SPLITS:
        data = load_dataset(split)
        metrics, outputs = evaluate_split(data)
        make_figures(data, metrics, outputs)
        for run in outputs:
            if run.startswith("dsae"):
                save_latent_map(data, outputs, run)
        all_metrics[split] = metrics
    all_metrics["paper"] = PAPER
    all_metrics["sweep"] = make_overview_figures(all_metrics)
    all_metrics["ablation"] = json.loads(C.ABLATION_FILE.read_text()) if C.ABLATION_FILE.exists() else []

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    summary = summary_markdown(all_metrics)
    (C.RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

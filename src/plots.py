"""Figures for the report, the slides and the demo.

One visual language throughout: each genre keeps the same colour and marker
everywhere, grids and axes stay faint, and text is never drawn in a data colour.
The four genre colours were checked for colour-blind separation on all pairs.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from . import config as C

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
GENRE_COLORS = {"classical": "#2a78d6", "jazz": "#eb6834", "metal": "#4a3aa7", "pop": "#1baf7a"}
GENRE_MARKERS = {"classical": "o", "jazz": "s", "metal": "^", "pop": "D"}
TRAIN_COLOR, DEV_COLOR = "#2a78d6", "#eb6834"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def save(fig, name: str):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def genre_legend(fig, y: float = 1.0):
    handles = [plt.Line2D([], [], linestyle="", marker=GENRE_MARKERS[g], markersize=6,
                          markerfacecolor=GENRE_COLORS[g], markeredgecolor=SURFACE, label=g)
               for g in C.GENRES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(C.GENRES),
               handletextpad=0.2, columnspacing=1.6)


def project(train_feats: np.ndarray, feats: np.ndarray):
    """Fit a 2-component PCA on the training features and project `feats`."""
    pca = PCA(n_components=2, random_state=0).fit(train_feats)
    return pca.transform(feats), pca.explained_variance_ratio_, pca


def scatter_genres(ax, points: np.ndarray, y: np.ndarray, title: str, var=None):
    # draw the genres in interleaved layers so that none sits systematically on top
    layers = 8
    for layer in range(layers):
        for label, g in enumerate(C.GENRES):
            idx = np.flatnonzero(y == label)[layer::layers]
            ax.scatter(points[idx, 0], points[idx, 1], s=13, marker=GENRE_MARKERS[g], color=GENRE_COLORS[g],
                       edgecolors=SURFACE, linewidths=0.35, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("PC 1" + (f" ({var[0]:.0%} of variance)" if var is not None else ""))
    ax.set_ylabel("PC 2" + (f" ({var[1]:.0%})" if var is not None else ""))
    ax.tick_params(length=0)


def pca_panels(panels, y, name: str):
    """panels: list of (title, train_features, plotted_features)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.3), squeeze=False)
    for ax, (title, train_feats, feats) in zip(axes[0], panels):
        points, var, _ = project(train_feats, feats)
        scatter_genres(ax, points, y, title, var)
    genre_legend(fig)
    fig.tight_layout()
    return save(fig, name)


def accuracy_curves(histories: dict, name: str):
    """histories: display name -> list of per-epoch rows with train_acc / dev_acc."""
    fig, axes = plt.subplots(1, len(histories), figsize=(3.6 * len(histories), 2.9), sharey=True, squeeze=False)
    for ax, (title, rows) in zip(axes[0], histories.items()):
        epochs = [r["epoch"] for r in rows]
        # when the two curves end close together, move their labels apart
        close = abs(rows[-1]["train_acc"] - rows[-1]["dev_acc"]) < 0.05
        train_on_top = rows[-1]["train_acc"] >= rows[-1]["dev_acc"]
        for key, color, label in (("train_acc", TRAIN_COLOR, "train"), ("dev_acc", DEV_COLOR, "dev")):
            values = [r[key] for r in rows]
            nudge = 0 if not close else 6 if (key == "train_acc") == train_on_top else -6
            ax.plot(epochs, values, color=color, linewidth=2, solid_capstyle="round", label=label)
            ax.annotate(f"{label} {values[-1]:.0%}", (epochs[-1], values[-1]), xytext=(5, nudge),
                        textcoords="offset points", va="center", color=INK_2, fontsize=8.5)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.2, 1.0)
        ax.set_xlim(0, epochs[-1] * 1.22)
        ax.set_xticks(np.linspace(0, epochs[-1], 6))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(loc="upper left", ncol=2, handlelength=1.2)
    fig.tight_layout()
    return save(fig, name)


def confusion_matrix(matrix: np.ndarray, title: str, name: str):
    """Row-normalised confusion matrix; rows are true genres."""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.pcolormesh(matrix, cmap=BLUES, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i in range(len(C.GENRES)):
        for j in range(len(C.GENRES)):
            ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color=SURFACE if matrix[i, j] > 0.5 else INK, fontsize=9)
    ticks = np.arange(len(C.GENRES)) + 0.5
    ax.set_xticks(ticks, C.GENRES)
    ax.set_yticks(ticks, C.GENRES)
    ax.invert_yaxis()
    ax.set_xlabel("predicted genre")
    ax.set_ylabel("true genre")
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return save(fig, name)


def reconstructions(originals: np.ndarray, recons: dict, y: np.ndarray, name: str):
    """One pooled clip per genre with each model's reconstruction, in input units."""
    fig, axes = plt.subplots(len(recons), len(C.GENRES), figsize=(7.4, 1.5 * len(recons) + 0.5),
                             sharex=True, sharey="col", squeeze=False)
    t = np.arange(originals.shape[1]) * C.POOL_SIZE / C.SAMPLE_RATE
    for row, (model_name, rec) in enumerate(recons.items()):
        for col, genre in enumerate(C.GENRES):
            ax = axes[row][col]
            i = int(np.flatnonzero(y == col)[0])
            ax.plot(t, originals[i], color=AXIS, linewidth=1.6, label="input clip")
            ax.plot(t, rec[i], color=GENRE_COLORS[genre], linewidth=1.0, label="reconstruction")
            ax.tick_params(length=0)
            if row == 0:
                ax.set_title(genre)
            if row == len(recons) - 1:
                ax.set_xlabel("seconds")
            if col == 0:
                ax.set_ylabel(model_name, fontsize=8)
    handles = [plt.Line2D([], [], color=AXIS, linewidth=1.6, label="input clip (500 pooled samples)"),
               plt.Line2D([], [], color=INK_2, linewidth=1.0, label="reconstruction from the 64-dim encoding")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    return save(fig, name)


def dot_comparison(rows: list, columns: dict, title: str, name: str, chance: float | None = None,
                   xlabel: str = "test accuracy", xlim: tuple = (0.15, 1.0)):
    """Dumbbell chart: one row per model, one dot per column (e.g. per split).

    rows: list of (label, {column_key: value}); columns: column_key -> (legend label, colour).
    """
    fig, ax = plt.subplots(figsize=(6.2, 0.46 * len(rows) + 1.3))
    for i, (label, values) in enumerate(rows):
        present = [values[k] for k in columns if values.get(k) is not None]
        ax.plot([min(present), max(present)], [i, i], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        for k, (_, color) in columns.items():
            if values.get(k) is None:
                continue
            ax.scatter(values[k], i, s=70, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=2)
        ax.annotate(f"{max(present):.0%}", (max(present), i), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8.5)
        if max(present) - min(present) > 0.03:
            ax.annotate(f"{min(present):.0%}", (min(present), i), xytext=(-8, 0), textcoords="offset points",
                        va="center", ha="right", color=INK_2, fontsize=8.5)
    if chance is not None:
        ax.axvline(chance, color=AXIS, linewidth=0.8)
        ax.annotate("chance", (chance, len(rows) - 0.5), xytext=(4, 3), textcoords="offset points",
                    va="bottom", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=24)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=7, markerfacecolor=color,
                          markeredgecolor=SURFACE, label=label) for label, color in columns.values()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(columns),
              handletextpad=0.2, columnspacing=1.4, borderaxespad=0.1)
    fig.tight_layout()
    return save(fig, name)


def transition_curves(transitions: dict, name: str):
    """One panel per pair of genres: average genre probabilities on the walk from A to B.

    transitions: "A -> B" -> {curves, direct, via}, as returned by morph.transitions.
    The two end genres are drawn solid, the other two dashed.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4), sharey=True)
    for ax, (pair, t) in zip(axes.ravel(), transitions.items()):
        a, b = pair.split(" -> ")
        curves = np.asarray(t["curves"])
        alphas = np.linspace(0, 1, len(curves))
        for k, g in enumerate(C.GENRES):
            ends = g in (a, b)
            ax.plot(alphas, curves[:, k], color=GENRE_COLORS[g], linewidth=2 if ends else 1.3,
                    linestyle="-" if ends else (0, (3, 2)))
        via, share = max(t["via"].items(), key=lambda item: item[1])
        note = f"direct {t['direct']:.0%}" + (f", via {via} {share:.0%}" if share >= 0.05 else "")
        ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center", va="top", color=INK_2, fontsize=8)
        ax.set_title(f"{a} to {b}")
        ax.set_xticks([0, 0.5, 1], [a, "halfway", b])
        for tick, align in zip(ax.get_xticklabels(), ("left", "center", "right")):
            tick.set_horizontalalignment(align)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(length=0)
    for row in axes:
        row[0].set_ylabel("average probability")
    handles = [plt.Line2D([], [], color=GENRE_COLORS[g], linewidth=2, label=g) for g in C.GENRES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(C.GENRES),
               handlelength=1.4, columnspacing=1.6)
    fig.tight_layout()
    return save(fig, name)


def gamma_sweep(runs: list, name: str):
    """Genre accuracy and reconstruction error as the loss weight gamma goes from 0 to 1.

    runs: the JSON records written by src.sweep. The gammas are spaced evenly on the
    x axis because the interesting values (0.9, 0.99, 1) lie very close together.
    """
    runs = sorted(runs, key=lambda r: r["gamma"])
    x = np.arange(len(runs))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    series = (("classifier head", [r["accuracy"]["test"] for r in runs], TRAIN_COLOR, "o"),
              ("linear probe on the 64-dim encoding", [r["probe_test"] for r in runs], DEV_COLOR, "s"))
    for label, values, color, marker in series:
        left.plot(x, values, color=color, linewidth=2, marker=marker, markersize=6.5,
                  markeredgecolor=SURFACE, label=label)
    left.axhline(1 / len(C.GENRES), color=AXIS, linewidth=0.8)
    left.annotate("chance", (x[-1], 1 / len(C.GENRES)), xytext=(0, 3), textcoords="offset points",
                  ha="right", color=MUTED, fontsize=8)
    left.set_title("Genre accuracy on the test set", pad=24)
    left.set_ylim(0.2, max(0.7, max(max(values) for _, values, _, _ in series) + 0.05))
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    left.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.4,
                columnspacing=1.4, borderaxespad=0.1)
    right.plot(x, [r["reconstruction_mse"]["test"] for r in runs], color=INK_2, linewidth=2,
               marker="o", markersize=6.5, markeredgecolor=SURFACE)
    right.set_title("Reconstruction error on the test set (MSE)", pad=24)
    for ax in (left, right):
        ax.set_xticks(x, [f"{r['gamma']:g}" for r in runs])
        ax.set_xlabel("gamma, the weight on reconstruction")
        ax.tick_params(length=0)
    fig.tight_layout()
    return save(fig, name)

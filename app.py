"""Live demo: play a song and watch the Deep Softmax Autoencoder encode it.

    streamlit run app.py
"""
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import resample_poly

from src import config as C
from src.data import held_out_songs, load_audio
from src.inference import GenrePredictor, Prediction
from src.models import DISPLAY_NAMES
from src.morph import Morph, morph, typical_clip
from src.plots import AXIS, GENRE_COLORS, GRID, INK, INK_2, MUTED, SURFACE
from src.retrieval import SongIndex, index_path

CLIP_SECONDS = C.CLIP_SAMPLES / C.SAMPLE_RATE
GENRE_SCALE = alt.Scale(domain=list(C.GENRES), range=[GENRE_COLORS[g] for g in C.GENRES])
GENRE_SHAPES = alt.Scale(domain=list(C.GENRES), range=["circle", "square", "triangle-up", "diamond"])
BLUE_RAMP = ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BEST = (C.BEST_RUN, "song")
MODELS = {   # label in the sidebar -> (run, split)
    "Random crops, song-level split: our most accurate model on unseen songs": BEST,
    "Fixed clips, song-level split: test songs are never heard in training": ("dsae", "song"),
    "Fixed clips, random clip split: the paper's protocol": ("dsae", "random"),
}
SPLIT_LABELS = {
    "song": "Song-level split: test songs are never heard in training",
    "random": "Random clip split: the paper's protocol",
}

st.set_page_config(page_title="Genre encodings from raw audio", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def get_predictor(run: str, split: str) -> GenrePredictor:
    return GenrePredictor(run, split)


@st.cache_data(show_spinner=False)
def read_song(path: str) -> np.ndarray:
    return load_audio(path, allow_headerless=True)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes) -> np.ndarray:
    return load_audio(io.BytesIO(data))


@st.cache_data
def get_latent_map(run: str, split: str) -> dict:
    with np.load(C.MODELS_DIR / f"latent_map_{run}_{split}.npz") as d:
        return {k: d[k] for k in d.files}


@st.cache_resource
def get_song_index() -> SongIndex:
    return SongIndex.load(index_path())


@st.cache_data
def get_held_out_songs() -> list:
    return [(C.GENRES[label], str(path)) for label, path in held_out_songs()]


def file_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def as_audio(clips: np.ndarray) -> np.ndarray:
    """Pooled clips brought back to the original sample rate so that they can be played."""
    return resample_poly(clips.ravel(), C.POOL_SIZE, 1).astype(np.float32)


# ---------------------------------------------------------------- charts
def styled(chart):
    return (chart.configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK_2,
                            titleColor=INK_2, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12, titleFontSize=12,
                              titleFontWeight="normal"))


def probability_bars(probs: np.ndarray, title: str):
    df = pd.DataFrame({"genre": C.GENRES, "probability": probs})
    base = alt.Chart(df).encode(
        y=alt.Y("genre:N", sort=list(C.GENRES), title=None, axis=alt.Axis(ticks=False, domain=False)),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title=title,
                axis=alt.Axis(format="%", tickCount=5)))
    bars = base.mark_bar(size=18, cornerRadiusEnd=4).encode(
        color=alt.Color("genre:N", scale=GENRE_SCALE, legend=None),
        tooltip=[alt.Tooltip("genre:N"), alt.Tooltip("probability:Q", format=".1%")])
    labels = base.mark_text(align="left", dx=6, color=INK_2, fontSize=12).encode(
        text=alt.Text("probability:Q", format=".0%"))
    return styled((bars + labels).properties(height=170))


def timeline(pred: Prediction):
    n = len(pred.probs)
    df = pd.DataFrame({
        "start": np.repeat(np.arange(n) * CLIP_SECONDS, len(C.GENRES)),
        "end": np.repeat((np.arange(n) + 1) * CLIP_SECONDS, len(C.GENRES)),
        "genre": np.tile(C.GENRES, n),
        "probability": pred.probs.ravel(),
    })
    chart = alt.Chart(df).mark_rect(stroke=SURFACE, strokeWidth=1.5).encode(
        x=alt.X("start:Q", title="position in the song (seconds)", scale=alt.Scale(nice=False)),
        x2="end:Q",
        y=alt.Y("genre:N", sort=list(C.GENRES), title=None, axis=alt.Axis(ticks=False, domain=False)),
        color=alt.Color("probability:Q", scale=alt.Scale(domain=[0, 1], range=BLUE_RAMP),
                        legend=alt.Legend(format="%", title="probability", gradientLength=110)),
        tooltip=[alt.Tooltip("genre:N"), alt.Tooltip("start:Q", format=".1f", title="clip starts at (s)"),
                 alt.Tooltip("probability:Q", format=".1%")])
    return styled(chart.properties(height=170))


def project(encodings: np.ndarray, lmap: dict) -> np.ndarray:
    """Place 64-dim encodings on the stored 2-D PCA map."""
    return (np.atleast_2d(encodings) - lmap["mean"]) @ lmap["components"].T


def reference_clips(lmap: dict):
    """The held-out clips of the test set as a faint backdrop for the latent maps."""
    df = pd.DataFrame({"x": lmap["points"][:, 0], "y": lmap["points"][:, 1],
                       "genre": [C.GENRES[i] for i in lmap["labels"]]})
    var = lmap["explained"]
    return alt.Chart(df).mark_point(filled=True, size=26, opacity=0.4).encode(
        x=alt.X("x:Q", title=f"PC 1 ({var[0]:.0%} of variance)"),
        y=alt.Y("y:Q", title=f"PC 2 ({var[1]:.0%})"),
        color=alt.Color("genre:N", scale=GENRE_SCALE, legend=alt.Legend(title="held-out clips", orient="top")),
        shape=alt.Shape("genre:N", scale=GENRE_SHAPES, legend=None),
        tooltip=[alt.Tooltip("genre:N", title="held-out clip")])


def latent_space(pred: Prediction, lmap: dict):
    xy = project(pred.encodings, lmap)
    song = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                         "starts at (s)": np.arange(len(xy)) * CLIP_SECONDS,
                         "predicted": [C.GENRES[i] for i in pred.probs.argmax(1)],
                         "confidence": pred.probs.max(1)})
    foreground = alt.Chart(song).mark_point(filled=True, size=110, color=INK, stroke=SURFACE,
                                            strokeWidth=1.5, opacity=1).encode(
        x="x:Q", y="y:Q",
        tooltip=[alt.Tooltip("starts at (s):Q", format=".1f"), alt.Tooltip("predicted:N"),
                 alt.Tooltip("confidence:Q", format=".0%")])
    return styled((reference_clips(lmap) + foreground).properties(height=400))


def waveforms(seconds: np.ndarray, signals: dict, colors: list):
    """Overlaid waveforms with a crosshair; signals maps a legend label to its samples."""
    wide = pd.DataFrame({"seconds": seconds, **signals})
    long = wide.melt("seconds", var_name="signal", value_name="amplitude")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6, strokeJoin="round").encode(
        x=alt.X("seconds:Q", title="seconds within the clip", scale=alt.Scale(nice=False)),
        y=alt.Y("amplitude:Q", title="amplitude"),
        color=alt.Color("signal:N", scale=alt.Scale(domain=list(signals), range=colors),
                        legend=alt.Legend(title=None, orient="top")))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["seconds"], empty=False)
    crosshair = alt.Chart(wide).mark_rule(color=AXIS).encode(
        x="seconds:Q",
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("seconds:Q", format=".3f")] + [alt.Tooltip(f"{k}:Q", format=".3f") for k in signals]
    ).add_params(nearest)
    return styled((lines + crosshair).properties(height=400))


def morph_curves(m: Morph, step: int):
    df = pd.DataFrame({"position": np.repeat(m.alphas, len(C.GENRES)),
                       "genre": np.tile(C.GENRES, len(m.alphas)),
                       "probability": m.probs.ravel()})
    lines = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("position:Q", title="position on the walk from song A to song B",
                axis=alt.Axis(format="%", tickCount=5)),
        y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), title="genre probability",
                axis=alt.Axis(format="%")),
        color=alt.Color("genre:N", scale=GENRE_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("genre:N"), alt.Tooltip("position:Q", format=".0%"),
                 alt.Tooltip("probability:Q", format=".1%")])
    here = alt.Chart(pd.DataFrame({"position": [float(m.alphas[step])]})).mark_rule(
        color=INK, strokeWidth=1.5, strokeDash=[4, 3]).encode(x="position:Q")
    return styled((lines + here).properties(height=300))


def morph_map(m: Morph, step: int, lmap: dict):
    xy = project(m.encodings, lmap)
    path = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "position": m.alphas,
                         "top genre": [C.GENRES[i] for i in m.probs.argmax(1)]})
    line_ = alt.Chart(path).mark_line(color=INK, strokeWidth=2).encode(x="x:Q", y="y:Q", order="position:Q")
    ends = alt.Chart(path.iloc[[0, -1]].assign(label=["A", "B"])).mark_text(
        color=INK, fontSize=14, fontWeight="bold", dy=-14).encode(x="x:Q", y="y:Q", text="label:N")
    here = alt.Chart(path.iloc[[step]]).mark_point(filled=True, size=160, color=INK, stroke=SURFACE,
                                                    strokeWidth=2, opacity=1).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("position:Q", format=".0%"), alt.Tooltip("top genre:N")])
    return styled((reference_clips(lmap) + line_ + ends + here).properties(height=400))


def song_map(index: SongIndex, matches: list, pred: Prediction, lmap: dict):
    xy = project(index.means, lmap)
    library = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "song": index.names,
                            "genre": [C.GENRES[i] for i in index.labels]})
    found = library[library["song"].isin([m.name for m in matches])]
    q = project(pred.encodings.mean(axis=0), lmap)
    var = lmap["explained"]
    songs = alt.Chart(library).mark_point(filled=True, size=45, opacity=0.55).encode(
        x=alt.X("x:Q", title=f"PC 1 ({var[0]:.0%} of variance)"),
        y=alt.Y("y:Q", title=f"PC 2 ({var[1]:.0%})"),
        color=alt.Color("genre:N", scale=GENRE_SCALE, legend=alt.Legend(title="library songs", orient="top")),
        shape=alt.Shape("genre:N", scale=GENRE_SHAPES, legend=None),
        tooltip=[alt.Tooltip("song:N"), alt.Tooltip("genre:N")])
    rings = alt.Chart(found).mark_point(size=260, color=INK, strokeWidth=2, opacity=1).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("song:N", title="match"), alt.Tooltip("genre:N")])
    query = alt.Chart(pd.DataFrame({"x": q[:, 0], "y": q[:, 1], "song": ["this song"]})).mark_point(
        filled=True, size=190, shape="cross", color=INK, stroke=SURFACE, strokeWidth=1.5, opacity=1).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("song:N")])
    return styled((songs + rings + query).properties(height=420))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Audio")
    have_dataset = C.GENRES_DIR.exists()
    sources = ["Held-out GTZAN song", "Upload a file"] if have_dataset else ["Upload a file"]
    source = st.radio("Source", sources, label_visibility="collapsed")
    signal, true_genre, title = None, None, None
    if source == "Held-out GTZAN song":
        songs = get_held_out_songs()
        genre = st.selectbox("Genre", C.GENRES)
        path = st.selectbox("Song", [p for g, p in songs if g == genre], format_func=file_name)
        signal, true_genre, title = read_song(path), genre, file_name(path)
    else:
        upload = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3", "au"])
        if upload is not None:
            try:
                signal, title = read_upload(upload.getvalue()), upload.name
            except Exception as error:  # unreadable or unsupported audio
                st.error(f"Could not read this file: {error}")

    st.header("Model")
    run, split = MODELS[st.radio("Trained on", list(MODELS), label_visibility="collapsed")]
    if split == "random" and source == "Held-out GTZAN song":
        st.caption("With the random clip split the model has usually trained on other clips "
                   "of this same song.")
    st.caption("The Genre morph and Similar songs tabs always use the first model.")

# ---------------------------------------------------------------- page
st.title("Genre encodings from raw audio")
st.caption("A Deep Softmax Autoencoder squeezes every 0.9 s of raw waveform into 64 numbers, "
           "then reconstructs the audio and predicts the genre from those 64 numbers alone. "
           "No spectrograms, no MFCCs.")

demo_tab, morph_tab, similar_tab, results_tab = st.tabs(
    ["Live demo", "Genre morph", "Similar songs", "Project results"])
ready = signal is not None and len(signal) >= C.CLIP_SAMPLES


def explain_missing_song():
    if signal is None:
        st.info("Choose a song in the sidebar to begin.")
    else:
        st.error(f"This audio is shorter than one clip ({CLIP_SECONDS:.1f} s).")


with demo_tab:
    if not ready:
        explain_missing_song()
    else:
        pred = get_predictor(run, split).predict(signal)
        votes = np.bincount(pred.probs.argmax(1), minlength=len(C.GENRES))

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader(title)
            st.audio(signal, sample_rate=C.SAMPLE_RATE)
            st.caption(f"{len(signal) / C.SAMPLE_RATE:.0f} s of audio cut into {len(pred.clips)} clips of "
                       f"{C.CLIP_SAMPLES:,} samples. Each clip is average-pooled to {C.INPUT_DIM} numbers.")
            with st.expander("Hear what the network hears"):
                st.caption(f"The pooled input: one number for every {C.POOL_SIZE} samples, so only sound below "
                           f"about {C.SAMPLE_RATE / C.POOL_SIZE / 2:.0f} Hz survives. That is the bass, "
                           "the kick drum and the overall loudness.")
                st.audio(as_audio(pred.clips), sample_rate=C.SAMPLE_RATE)
                st.caption(f"The same input rebuilt by the decoder from {C.ENCODING_DIM} numbers per clip.")
                st.audio(as_audio(pred.recon), sample_rate=C.SAMPLE_RATE)
            st.markdown("**Clip by clip**")
            st.altair_chart(timeline(pred), width="stretch", theme=None)
        with right:
            st.metric("Predicted genre", pred.genre,
                      help="The genre with the highest probability once the clips are averaged.")
            st.caption(f"{votes.max()} of {len(pred.clips)} clips vote {C.GENRES[int(votes.argmax())]}.")
            if true_genre is not None:
                if pred.genre == true_genre:
                    st.success(f"Correct. GTZAN labels this song {true_genre}.", icon=":material/check_circle:")
                else:
                    st.error(f"Wrong. GTZAN labels this song {true_genre}.", icon=":material/cancel:")
            st.altair_chart(probability_bars(pred.song_probs, "average probability over all clips"),
                            width="stretch", theme=None)

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Where this song lands in the 64-dim latent space**")
            st.caption("Black dots are this song's clips. Coloured marks are 1,000 held-out clips, "
                       "projected to two dimensions with PCA.")
            st.altair_chart(latent_space(pred, get_latent_map(run, split)), width="stretch", theme=None)
        with right:
            st.markdown("**What the 64 numbers keep of one clip**")
            clip = st.slider("Clip", 1, len(pred.clips), 1) if len(pred.clips) > 1 else 1
            seconds = np.arange(C.INPUT_DIM) * C.POOL_SIZE / C.SAMPLE_RATE
            st.altair_chart(waveforms(seconds, {"input clip": pred.clips[clip - 1],
                                                "reconstruction": pred.recon[clip - 1]}, [MUTED, "#2a78d6"]),
                            width="stretch", theme=None)

with morph_tab:
    st.caption("Latent interpolation, which the paper leaves as future work. We take the most typical clip of "
               "two songs, walk in a straight line between their 64-dim encodings, and at every step ask the "
               "decoder for a waveform and the classifier for a genre.")
    if not ready:
        explain_missing_song()
    elif not have_dataset:
        st.info("This tab picks its second song from the GTZAN test songs. "
                "Run `python scripts/download_data.py` to fetch them.")
    else:
        predictor = get_predictor(*BEST)
        a = predictor.predict(signal)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(f"**Song A:** {title}")
            st.caption(f"Chosen in the sidebar. Predicted {a.genre}.")
        with right:
            songs = get_held_out_songs()
            default = (C.GENRES.index(true_genre) + 2) % len(C.GENRES) if true_genre else len(C.GENRES) - 1
            genre_b = st.selectbox("Genre of song B", C.GENRES, index=default)
            path_b = st.selectbox("Song B", [p for g, p in songs if g == genre_b], format_func=file_name)
        b = predictor.predict(read_song(path_b))
        m = morph(predictor.model, a.encodings[typical_clip(a.encodings)], b.encodings[typical_clip(b.encodings)])
        position = st.slider("Position on the walk from A (0%) to B (100%)", 0, 100, 50, step=5, format="%d%%")
        step = position // 5
        top = int(m.probs[step].argmax())
        visited = [C.GENRES[i] for i in pd.unique(m.probs.argmax(1))]

        left, right = st.columns(2, gap="large")
        with left:
            st.metric(f"Top genre {position}% of the way", f"{C.GENRES[top]} ({m.probs[step, top]:.0%})")
            st.altair_chart(morph_curves(m, step), width="stretch", theme=None)
            st.caption("Genres met along the whole walk: " + " → ".join(visited) + ".")
        with right:
            st.markdown("**The walk on the latent map**")
            st.altair_chart(morph_map(m, step, get_latent_map(*BEST)), width="stretch", theme=None)

        st.markdown("**The decoder's waveform at this position**")
        seconds = np.arange(C.INPUT_DIM) * C.POOL_SIZE / C.SAMPLE_RATE
        decoded = predictor.scaler.inverse(m.recon)
        st.altair_chart(waveforms(seconds, {"decoded at A": decoded[0], "decoded at B": decoded[-1],
                                            f"decoded {position}% of the way": decoded[step]},
                                  [AXIS, MUTED, "#2a78d6"]).properties(height=260),
                        width="stretch", theme=None)

with similar_tab:
    st.caption("Similar-song search: a use of the encodings beyond classification. Every song gets a "
               "fingerprint, the average of its clip encodings, and the library of 300 training songs is "
               "ranked by how closely their fingerprints point the same way.")
    if not ready:
        explain_missing_song()
    elif not index_path().exists():
        st.info("Run `python -m src.evaluate` to build the song library.")
    else:
        query = get_predictor(*BEST).predict(signal)
        index = get_song_index()
        matches = index.search(query.encodings, k=5)
        wanted = true_genre or query.genre
        hits = sum(m.genre == wanted for m in matches)
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown(f"**Most similar to** {title}")
            st.caption(f"{hits} of the 5 are {wanted}, " +
                       ("the genre GTZAN gives this song." if true_genre else "the genre predicted for this song."))
            for rank, match in enumerate(matches, 1):
                st.markdown(f"{rank}. **{file_name(match.name)}**, {match.genre}, "
                            f"similarity {match.similarity:.3f}")
                if have_dataset:
                    st.audio(read_song(str(C.GENRES_DIR / match.name)), sample_rate=C.SAMPLE_RATE)
        with right:
            st.markdown("**The library as the model sees it**")
            st.caption("One mark per library song, placed by its fingerprint. The cross is this song "
                       "and the rings are its five matches.")
            st.altair_chart(song_map(index, matches, query, get_latent_map(*BEST)), width="stretch", theme=None)

with results_tab:
    metrics_file = C.RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        st.info("Run `python -m src.evaluate` to generate the results.")
    else:
        metrics = json.loads(metrics_file.read_text())
        shown = st.radio("Evaluation protocol", list(SPLIT_LABELS), format_func=SPLIT_LABELS.get, horizontal=True)
        m = metrics[shown]

        st.markdown("**Classification accuracy**")
        rows = []
        for name, accuracy in m["accuracy"].items():
            if name in metrics["paper"]["accuracy"]:
                rows.append({"Model": DISPLAY_NAMES[name], "Source": "paper", **{
                    k.capitalize(): f"{v:.1%}" for k, v in metrics["paper"]["accuracy"][name].items()}})
            rows.append({"Model": DISPLAY_NAMES[name], "Source": "this project", **{
                k.capitalize(): f"{v:.1%}" for k, v in accuracy.items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("**How much genre can a linear classifier read from each representation?**")
        st.dataframe(pd.DataFrame([{"Features": k, "Test accuracy": f"{v:.1%}"}
                                   for k, v in m["linear_probe_test"].items()]), hide_index=True, width="stretch")

        if "retrieval" in m:
            r = m["retrieval"]
            st.markdown(f"**Similar-song search: {r['n_queries']} unseen songs against "
                        f"{r['n_library']} library songs**")
            st.dataframe(pd.DataFrame([
                {"Song fingerprint built from": k, "Top-1 same genre": f"{s['precision_at_1']:.1%}",
                 f"Precision@{r['k']}": f"{s['precision_at_k']:.1%}", "Mean average precision": f"{s['map']:.3f}"}
                for k, s in r["scores"].items()]), hide_index=True, width="stretch")

        figures = [
            ("accuracy_overview", "Test accuracy against the paper"),
            (f"accuracy_curves_{shown}", "Accuracy during training"),
            (f"interpolation_{shown}", "Latent interpolation: average genre probabilities on the walk between two genres"),
            (f"interpolation_steady_{shown}", "Latent walks compared with cross-fading the audio"),
            (f"retrieval_{shown}", "Similar-song search"),
            (f"pca_encodings_{shown}", "PCA of the 64-dim encodings of test clips"),
            (f"pca_logits_{shown}", "PCA of the 4-dim logits of test clips"),
            (f"confusion_{C.BEST_RUN}_{shown}", f"Confusion matrix of the {DISPLAY_NAMES[C.BEST_RUN]}"),
            (f"confusion_dsae_{shown}", "Confusion matrix of the Deep Softmax Autoencoder"),
            ("linear_probe", "Linear probes on both splits"),
            (f"gamma_sweep_{shown}", "Effect of the loss weight gamma"),
            (f"reconstructions_{shown}", "Reconstructions of one test clip per genre"),
            (f"pca_raw_{shown}", "PCA of the raw input"),
        ]
        for name, caption in figures:
            path = C.FIGURES_DIR / f"{name}.png"
            if path.exists():
                st.markdown(f"**{caption}**")
                st.image(str(path))

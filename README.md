# Latent Feature Extraction for Musical Genres from Raw Audio

UE24CS352A Machine Learning mini-project, PES University.
Team: Amogh Sharma (PES1UG24CS053) and Amartya Sinha (PES1UG24CS051).

A re-implementation and extension of the Stanford CS229 project by Sawhney, Vasavada and Wang (2018):
[report](https://cs229.stanford.edu/proj2018/report/20.pdf),
[poster](https://cs229.stanford.edu/proj2018/poster/20.pdf).

The task is to learn an encoding of musical genre straight from the raw waveform, with no
spectrograms or MFCCs. Every 0.9 s clip of audio is average-pooled to 500 numbers. A Deep Softmax
Autoencoder (DSAE) squeezes those into a 64-dim encoding, and from that encoding alone it both
reconstructs the clip and predicts one of four GTZAN genres: classical, jazz, metal, pop.

## Results at a glance

Test accuracy on 0.9 s clips (chance is 25%). Full tables are in `results/summary.md`.

| Model | Paper reports | Our run, paper's split | Our run, unseen songs | Unseen songs, with our random crops |
|---|---|---|---|---|
| Two Layer Neural Network | 36.4% | 47.0% | 40.5% | 66.8% |
| Deep Softmax Autoencoder | 65.3% | 56.3% | 52.2% | 66.8% |

- The paper shuffles clips, so clips of one song land in both training and test data. With whole
  songs held out, and GTZAN's duplicate recordings kept out of the test set, accuracy is lower.
- With the paper's 20 fixed clips per song the networks memorise: about 95% on training clips against
  50% on dev. A logistic regression on one hand-made number, clip loudness, gets 62%.
- Our random-crop training fixes this and beats the paper's reported accuracy on a stricter test.

## What we added to the paper

| Addition | Where |
|---|---|
| **Genre morph**: latent interpolation between two songs, with a measure of how smooth the latent space is. Future work in the paper. | `src/morph.py`, demo tab |
| **Similar-song search**: nearest neighbours in the encoding space as an external test of the encodings. Future work in the paper. | `src/retrieval.py`, demo tab |
| Random-crop training: new random clips of every training song in each epoch | `src/data.py` (`CropSampler`), `--augment` |
| Song-level split that keeps GTZAN's 30 duplicated recordings out of dev and test | `src/data.py` (`find_twins`, `split_indices`) |
| Linear probes, including a loudness baseline, on every representation | `src/evaluate.py` |
| Gamma sweep and a crops x learning-rate ablation | `src/sweep.py`, `src/ablation.py` |
| Streamlit demo and 20 unit tests | `app.py`, `tests/` |

What the two new features show: the latent space is a chain, classical → jazz → metal → pop, which
is the order of average loudness; and the nearest library song shares the genre of an unseen query
68% of the time.

## Setup

Python 3.11 or newer. No GPU is needed; every model trains on a laptop CPU in under five minutes.

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

The download is the GTZAN archive, about 1.2 GB. Only the four genres used here are unpacked into
`data/genres` (400 songs, about 500 MB). `data/processed/gtzan4.npz` already holds the pooled clips,
so training without `--augment` and the whole evaluation also run without the download. The demo and
`--augment` need the audio.

## Reproduce everything

```bash
# the paper's three models, on the paper's split and on ours
python -m src.train --model two_layer   --split random
python -m src.train --model autoencoder --split random
python -m src.train --model dsae        --split random
python -m src.train --model two_layer   --split song
python -m src.train --model autoencoder --split song
python -m src.train --model dsae        --split song

# our training change: random crops (song split only)
python -m src.train --model two_layer --split song --augment
python -m src.train --model dsae      --split song --augment

# optional experiments, each about 20 minutes
python -m src.sweep       # loss weight gamma from 0 to 1
python -m src.ablation    # random crops x learning rate

# all tables, figures and the files the demo needs
python -m src.evaluate
```

Every random choice is seeded, so the same command gives the same numbers (a different number of CPU
threads can change the last digit). `python -m src.evaluate` writes
`results/summary.md`, `results/metrics.json` and `results/figures/*.png`.

## Demo

```bash
streamlit run app.py
```

Four tabs:

- **Live demo.** Pick a held-out song or upload a file. Shows the predicted genre, the prediction clip
  by clip, where the clips land in the latent space, the decoder's reconstruction, and lets you hear
  the pooled audio the network actually receives.
- **Genre morph.** Walks in a straight line between the encodings of two songs and shows how the genre
  probabilities and the decoded waveform change on the way.
- **Similar songs.** Ranks the 300 library songs by how close their encodings are to the chosen song.
- **Project results.** The tables and figures from `results/`.

The same things work from the command line:

```bash
python -m src.inference path/to/song.wav
python -m src.morph song_a.wav song_b.wav
python -m src.retrieval path/to/song.wav
```

## Tests

```bash
python -m pytest
```

20 tests, about 15 seconds. They need neither the dataset nor trained weights.

## Layout

| Path | What it holds |
|---|---|
| `src/config.py` | Paths and every hyperparameter, marked as from the paper or ours |
| `src/data.py` | Audio loading, clipping, pooling, duplicate detection, the two splits, scaling, random crops |
| `src/models.py` | Two Layer Neural Network, Vanilla Autoencoder, Deep Softmax Autoencoder |
| `src/train.py` | One training loop for all models; the weighted loss of the paper |
| `src/inference.py` | Loading a checkpoint and predicting a whole audio file |
| `src/evaluate.py` | All metrics, tables and figures |
| `src/plots.py` | Every figure, in one visual style |
| `src/sweep.py`, `src/ablation.py` | The gamma sweep and the crops x learning-rate ablation |
| `src/morph.py` | Latent interpolation |
| `src/retrieval.py` | Similar-song search |
| `app.py` | Streamlit demo |
| `scripts/download_data.py` | Fetches and unpacks GTZAN |
| `tests/test_pipeline.py` | Unit tests |
| `writeup/Writeup.pdf` | The two-page write-up |
| `presentation/Presentation.pptx` | The slide deck |
| `data/` | Audio (not in git) and the cached clips |
| `models/` | Trained checkpoints, latent maps and the song library for the demo |
| `results/` | Metrics (`summary.md`, `metrics.json`), training histories, figures |

## References

1. A. Sawhney, V. Vasavada, W. Wang. Latent Feature Extraction for Musical Genres from Raw Audio. Stanford CS229, 2018.
2. G. Tzanetakis, P. Cook. Musical Genre Classification of Audio Signals. IEEE Transactions on Speech and Audio Processing, 2002.
3. B. L. Sturm. The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. arXiv:1306.1461, 2013.

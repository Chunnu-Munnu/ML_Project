"""Paths and hyperparameters shared by every stage of the pipeline.

Values marked "paper" come straight from Sawhney, Vasavada & Wang (CS229, 2018).
Values marked "ours" are choices we had to make where the paper is silent.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GENRES_DIR = DATA_DIR / "genres"
CACHE_FILE = DATA_DIR / "processed" / "gtzan4.npz"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SWEEP_DIR = RESULTS_DIR / "sweep"
ABLATION_FILE = RESULTS_DIR / "ablation.json"

# --- data (paper) ---------------------------------------------------------
GENRES = ("classical", "jazz", "metal", "pop")
SAMPLE_RATE = 22050
CLIP_SAMPLES = 20000          # ~0.9 s of raw audio, "in R^20,000"
POOL_SIZE = 40                # average pooling window
INPUT_DIM = CLIP_SAMPLES // POOL_SIZE   # 500
CLIPS_PER_SONG = 20           # 400 songs x 20 clips = 8000 examples
SPLIT_SIZES = (6000, 1000, 1000)        # train / dev / test
SPLIT_SEED = 229              # ours: decides which clips / songs are train, dev and test

# --- architecture (paper, Figure 2) ---------------------------------------
ENCODER_SIZES = (256, 192, 128)
ENCODING_DIM = 64
CLASSIFIER_SIZES = (32, 16)
BASELINE_HIDDEN = 128


@dataclass
class TrainConfig:
    epochs: int = 500         # paper (Figure 3 runs to 500)
    batch_size: int = 512     # paper
    lr: float = 1e-4          # paper, Adam
    gamma: float = 0.9        # paper: weight on reconstruction loss
    dropout: float = 0.1      # paper: keep probability 0.9
    seed: int = 229           # ours: weight initialisation and batch order
    augment: bool = False     # ours: random-crop augmentation, see data.CropSampler


AUGMENT_SUFFIX = "_aug"       # a run trained with --augment is saved as e.g. dsae_aug
AUGMENT_LR = 1e-3             # ours: learning rate with random crops, chosen on the dev set (src.ablation)
BEST_RUN = "dsae_aug"         # song-split run used by the demo and the latent-space analyses


def model_path(run: str, split: str) -> Path:
    return MODELS_DIR / f"{run}_{split}.pt"

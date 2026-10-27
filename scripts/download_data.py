"""Download GTZAN and unpack the four genres used in this project into data/genres.

    python scripts/download_data.py

The archive (about 1.2 GB) comes from the Hugging Face mirror of the GTZAN Genre
Collection (Tzanetakis & Cook, 2002). Only classical, jazz, metal and pop are
extracted, about 500 MB on disk.
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
ARCHIVE = C.DATA_DIR / "genres.tar.gz"


def download():
    if ARCHIVE.exists():
        print(f"{ARCHIVE} already exists, skipping download")
        return
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = ARCHIVE.with_suffix(".part")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        print(f"\rdownloading {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(URL, part, reporthook=progress)
    part.rename(ARCHIVE)
    print()


def extract():
    wanted = {f"genres/{g}/" for g in C.GENRES}
    count = 0
    with tarfile.open(ARCHIVE) as tar:
        for member in tar:
            name = Path(member.name).name
            # skip macOS resource forks ("._file") that ship inside the archive
            if not member.isfile() or name.startswith("."):
                continue
            if any(member.name.startswith(prefix) for prefix in wanted):
                tar.extract(member, C.DATA_DIR, filter="data")
                count += 1
    print(f"extracted {count} files to {C.GENRES_DIR}")


if __name__ == "__main__":
    download()
    extract()

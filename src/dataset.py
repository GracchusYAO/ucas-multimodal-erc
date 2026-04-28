"""MELD dataset parsing utilities.

TODO:
- Parse official MELD train/dev/test metadata CSV files.
- Attach utterance-level media paths.
- Keep dialogue and utterance ids for context batching.
"""

from pathlib import Path


EMOTION2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

ID2EMOTION = {value: key for key, value in EMOTION2ID.items()}

MELD_METADATA = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

MELD_MEDIA_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}


def get_meld_paths(data_root: str | Path = "data/meld") -> dict[str, dict[str, Path]]:
    """Return expected metadata and media paths for each MELD split."""
    root = Path(data_root)
    return {
        split: {
            "metadata": root / metadata_name,
            "media_dir": root / MELD_MEDIA_DIRS[split],
        }
        for split, metadata_name in MELD_METADATA.items()
    }

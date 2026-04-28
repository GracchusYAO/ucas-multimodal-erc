"""MELD 数据读取工具。

这个文件只做一件事：把官方 CSV 里的每条 utterance 读出来，并拼好对应
视频路径。后续文本/音频/视觉特征提取都依赖这里返回的顺序。
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# 7 类情绪的 id 顺序固定住，训练、评估和画混淆矩阵都用这一套。
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

# 官方 raw 包里三个 split 的视频目录名不同，所以集中写在这里。
MELD_MEDIA_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}

MELD_SPLITS = tuple(MELD_METADATA)


@dataclass(frozen=True)
class MELDUtterance:
    """一条 MELD utterance。

    注意：media_exists 不存在时也保留这条样本。dev 里官方缺了
    dia110_utt7.mp4，后面音频/视觉特征用零向量处理即可。
    """

    split: str
    sr_no: int
    dialogue_id: int
    utterance_id: int
    text: str
    speaker: str
    emotion: str
    emotion_id: int
    sentiment: str
    season: int
    episode: int
    start_time: str
    end_time: str
    media_path: Path
    media_exists: bool

    @property
    def key(self) -> str:
        return f"{self.split}:dia{self.dialogue_id}_utt{self.utterance_id}"

    @property
    def media_filename(self) -> str:
        return f"dia{self.dialogue_id}_utt{self.utterance_id}.mp4"


@dataclass(frozen=True)
class MELDDialogue:
    """按 dialogue 分组后的样本，给后面的 BiGRU context 用。"""

    split: str
    dialogue_id: int
    utterances: tuple[MELDUtterance, ...]

    @property
    def key(self) -> str:
        return f"{self.split}:dia{self.dialogue_id}"


def validate_split(split: str) -> str:
    split = split.lower()
    if split not in MELD_METADATA:
        raise ValueError(f"Unknown split: {split}. Expected one of {MELD_SPLITS}.")
    return split


def get_meld_paths(data_root: str | Path = "data/meld") -> dict[str, dict[str, Path]]:
    """返回每个 split 的 CSV 路径和视频目录路径。"""
    root = Path(data_root)
    return {
        split: {
            "metadata": root / csv_name,
            "media_dir": root / MELD_MEDIA_DIRS[split],
        }
        for split, csv_name in MELD_METADATA.items()
    }


def expected_media_path(
    split: str,
    dialogue_id: int | str,
    utterance_id: int | str,
    data_root: str | Path = "data/meld",
) -> Path:
    """按官方命名规则拼视频路径：dia{Dialogue_ID}_utt{Utterance_ID}.mp4。"""
    split = validate_split(split)
    return (
        Path(data_root)
        / MELD_MEDIA_DIRS[split]
        / f"dia{int(dialogue_id)}_utt{int(utterance_id)}.mp4"
    )


def load_meld_split(
    split: str,
    data_root: str | Path = "data/meld",
) -> list[MELDUtterance]:
    """读取一个 split，返回按 CSV 原顺序排列的 utterance 列表。"""
    split = validate_split(split)
    root = Path(data_root)
    csv_path = root / MELD_METADATA[split]
    media_dir = root / MELD_MEDIA_DIRS[split]

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {csv_path}")
    if not media_dir.exists():
        raise FileNotFoundError(f"Missing media directory: {media_dir}")

    utterances: list[MELDUtterance] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            emotion = row["Emotion"].strip().lower()
            if emotion not in EMOTION2ID:
                raise ValueError(f"Unknown emotion label: {emotion}")

            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])
            media_path = expected_media_path(split, dialogue_id, utterance_id, root)

            utterances.append(
                MELDUtterance(
                    split=split,
                    sr_no=int(row["Sr No."]),
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                    text=row["Utterance"],
                    speaker=row["Speaker"].strip(),
                    emotion=emotion,
                    emotion_id=EMOTION2ID[emotion],
                    sentiment=row["Sentiment"].strip().lower(),
                    season=int(row["Season"]),
                    episode=int(row["Episode"]),
                    start_time=row["StartTime"].strip(),
                    end_time=row["EndTime"].strip(),
                    media_path=media_path,
                    media_exists=media_path.exists(),
                )
            )

    return utterances


def load_meld_splits(
    splits: tuple[str, ...] = MELD_SPLITS,
    data_root: str | Path = "data/meld",
) -> dict[str, list[MELDUtterance]]:
    return {split: load_meld_split(split, data_root) for split in splits}


def group_by_dialogue(utterances: list[MELDUtterance]) -> list[MELDDialogue]:
    """按 Dialogue_ID 分组，并在每个 dialogue 内按 Utterance_ID 排序。"""
    grouped: dict[tuple[str, int], list[MELDUtterance]] = defaultdict(list)
    for item in utterances:
        # 保留官方 Dialogue_ID，不重新编号。train 中 Dialogue_ID=60 本来就不存在。
        grouped[(item.split, item.dialogue_id)].append(item)

    dialogues: list[MELDDialogue] = []
    for (split, dialogue_id), items in grouped.items():
        dialogues.append(
            MELDDialogue(
                split=split,
                dialogue_id=dialogue_id,
                utterances=tuple(sorted(items, key=lambda x: x.utterance_id)),
            )
        )
    return sorted(dialogues, key=lambda x: x.dialogue_id)


def summarize_split(split: str, data_root: str | Path = "data/meld") -> dict[str, object]:
    """简单 sanity check：数量、标签分布、缺视频、额外视频。"""
    split = validate_split(split)
    root = Path(data_root)
    utterances = load_meld_split(split, root)
    dialogues = group_by_dialogue(utterances)

    expected_media = {item.media_filename for item in utterances}
    actual_media = {
        path.name
        for path in (root / MELD_MEDIA_DIRS[split]).glob("*.mp4")
        if not path.name.startswith("._")
    }

    dialogue_ids = sorted({dialogue.dialogue_id for dialogue in dialogues})
    gaps = [
        item
        for item in range(dialogue_ids[0], dialogue_ids[-1] + 1)
        if item not in dialogue_ids
    ]

    return {
        "split": split,
        "utterances": len(utterances),
        "dialogues": len(dialogues),
        "dialogue_id_gaps": gaps,
        "missing_media": [item.media_filename for item in utterances if not item.media_exists],
        "extra_media": sorted(actual_media - expected_media),
        "emotion_counts": dict(sorted(Counter(item.emotion for item in utterances).items())),
        "sentiment_counts": dict(
            sorted(Counter(item.sentiment for item in utterances).items())
        ),
    }


def print_summary(summary: dict[str, object]) -> None:
    """把 summarize_split 的结果打印成人能看的格式。"""
    missing = summary["missing_media"]
    extra = summary["extra_media"]
    print(f"{summary['split']}:")
    print(f"  utterances: {summary['utterances']}")
    print(f"  dialogues: {summary['dialogues']}")
    print(f"  dialogue_id_gaps: {summary['dialogue_id_gaps']}")
    print(f"  missing_media: {len(missing)}")
    if missing:
        print(f"  missing_examples: {missing[:5]}")
    print(f"  extra_media: {len(extra)}")
    if extra:
        print(f"  extra_examples: {extra[:5]}")
    print(f"  emotions: {summary['emotion_counts']}")
    print(f"  sentiments: {summary['sentiment_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MELD metadata and media files.")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    args = parser.parse_args()

    for split in args.split or MELD_SPLITS:
        print_summary(summarize_split(split, args.data_root))


if __name__ == "__main__":
    main()

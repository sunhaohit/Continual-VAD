import bisect
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data as data

import utils


DATASET_ROOT_CANDIDATES = [
    "./dataset",
    os.path.join(os.path.dirname(__file__), "dataset"),
]


def _path_label(path: str) -> str:
    return os.path.basename(str(path).rstrip(os.sep)) if path else ""


def normalize_dataset_name(name: str) -> str:
    n = str(name).strip().lower().replace("_", "-")
    if n in ("ucfcrime", "ucf-crime"):
        return "ucf-crime"
    if n in ("shanghaitech", "shanghai-tech"):
        return "shanghaitech"
    return n


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _resolve_dataset_root(args, dataset_key: str) -> str:
    candidates = [getattr(args, "dataset_path", "")] + DATASET_ROOT_CANDIDATES
    for root in candidates:
        if _exists(os.path.join(root, dataset_key)):
            return root
    for root in candidates:
        if _exists(root):
            return root
    return getattr(args, "dataset_path", "")


def _resolve_feature_root(args, dataset_key: str, dataset_root: str) -> str:
    explicit = getattr(args, "feature_path", "")
    if explicit:
        return explicit
    folder = "UCFClipFeatures" if dataset_key == "ucf-crime" else "SH_clip_video_features"
    return os.path.join(dataset_root, dataset_key, folder)


def _resolve_split_file(dataset_root: str, dataset_key: str, preferred: str, fallbacks: List[str]) -> str:
    if _exists(preferred):
        return preferred
    split_dir = os.path.join(dataset_root, dataset_key)
    for name in [preferred] + fallbacks:
        if not name:
            continue
        candidate = os.path.join(split_dir, name)
        if _exists(candidate):
            return candidate
    return os.path.join(split_dir, preferred or fallbacks[0])


def _read_split_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _safe_load_feature(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def _feature_file(feature_root: str, video_name: str, feature_suffix: str) -> str:
    suffix = feature_suffix if feature_suffix.startswith(".") else "." + feature_suffix
    return os.path.join(feature_root, video_name + suffix)


def _clip_count(num_steps: int, clip_len: int, stride: int) -> int:
    if num_steps <= clip_len:
        return 1
    stride = max(1, int(stride))
    last_start = max(0, num_steps - clip_len)
    return last_start // stride + 1


class _ConcatTestDataset(data.Dataset):
    def __init__(self, args):
        self.cross_clip = int(args.cross_clip)
        self.use_cross = bool(getattr(args, "concat_test_use_cross", False))
        test_feature_path = getattr(args, "ucf_test_npy", "")
        if not test_feature_path:
            test_feature_path = os.path.join("dataset", "ucfcrime", "Concat_test_10.npy")
        print("Using test npy:", _path_label(test_feature_path))
        self.con_test = np.load(test_feature_path)
        print("Concat test shape:", self.con_test.shape)
        print("Concat test use_cross:", self.use_cross, "cross_clip:", self.cross_clip)

    def __getitem__(self, index):
        if not self.use_cross or self.cross_clip <= 1:
            return np.asarray(self.con_test[index], dtype=np.float32)
        data_item = np.expand_dims(self.con_test[index], 1)
        total = len(self.con_test)
        for k in range(1, self.cross_clip):
            src_idx = index + k if (index + k) < total else index
            data_item = np.concatenate((data_item, np.expand_dims(self.con_test[src_idx], 1)), axis=1)
        return np.asarray(data_item, dtype=np.float32)

    def __len__(self):
        return len(self.con_test)


class _SingleVideoTestDataset(data.Dataset):
    def __init__(self, args):
        self.clip_len = int(getattr(args, "clip_len", 10))
        self.clip_stride = max(1, int(getattr(args, "test_clip_stride", 1)))
        self.dataset_key = normalize_dataset_name(args.dataset_name)
        self.dataset_root = _resolve_dataset_root(args, self.dataset_key)
        self.feature_root = _resolve_feature_root(args, self.dataset_key, self.dataset_root)
        self.feature_suffix = getattr(args, "feature_suffix", ".npy")

        preferred_split = getattr(args, "test_split_file", "test_split_10crop.txt")
        split_path = _resolve_split_file(
            self.dataset_root,
            self.dataset_key,
            preferred_split,
            ["test_split_10crop.txt", "test_split.txt"],
        )
        self.testlist = _read_split_list(split_path)

        self.videos: List[Dict[str, object]] = []
        self.cum_counts: List[int] = []
        total = 0
        missing = 0
        for name in self.testlist:
            path = _feature_file(self.feature_root, name, self.feature_suffix)
            if not _exists(path):
                missing += 1
                continue
            feat_shape = np.load(path, mmap_mode="r").shape
            if len(feat_shape) == 0:
                continue
            feat_len = int(feat_shape[0])
            count = _clip_count(feat_len, self.clip_len, self.clip_stride)
            self.videos.append({"name": name, "path": path, "feat_len": feat_len, "count": count})
            total += count
            self.cum_counts.append(total)

        if not self.videos:
            raise FileNotFoundError(
                "No testing feature files found.\n"
                "dataset={}\nfeature_root={}\nsplit={}".format(
                    self.dataset_key,
                    _path_label(self.feature_root),
                    _path_label(split_path),
                )
            )
        if missing > 0:
            print("[test_loader] missing {} feature files from split (skipped).".format(missing))

        self._cache_video_idx: Optional[int] = None
        self._cache_feat: Optional[np.ndarray] = None
        print(
            "Single-video test dataset: videos={} clips={} | dataset={} | feature_root={} | stride={}".format(
                len(self.videos), len(self), self.dataset_key, _path_label(self.feature_root), self.clip_stride
            )
        )

    def _video_index(self, index: int) -> int:
        return bisect.bisect_right(self.cum_counts, index)

    def _load_video_feat(self, video_idx: int) -> np.ndarray:
        if self._cache_video_idx != video_idx or self._cache_feat is None:
            path = self.videos[video_idx]["path"]
            self._cache_feat = _safe_load_feature(path)
            self._cache_video_idx = video_idx
        return self._cache_feat

    def __getitem__(self, index: int):
        video_idx = self._video_index(index)
        prev_end = 0 if video_idx == 0 else self.cum_counts[video_idx - 1]
        local_idx = index - prev_end
        video = self.videos[video_idx]
        feat = self._load_video_feat(video_idx)
        feat_len = int(video["feat_len"])
        last_start = max(0, feat_len - self.clip_len)
        start = min(local_idx * self.clip_stride, last_start)
        clip = feat[start:start + self.clip_len]
        clip = utils.pad(clip, self.clip_len).astype(np.float32)
        return clip, video["name"], int(local_idx)

    def __len__(self):
        return self.cum_counts[-1] if self.cum_counts else 0


class Dataset_Con_all_feedback_XD(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        del is_normal
        del transform
        del test_mode
        data_mode = str(getattr(args, "data_mode", "single_video")).lower()
        if data_mode == "concat":
            self.impl = _ConcatTestDataset(args)
        else:
            self.impl = _SingleVideoTestDataset(args)

    def __getitem__(self, index):
        return self.impl[index]

    def __len__(self):
        return len(self.impl)


def _concat_feature_info(args, train: bool) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, ...]]]:
    attr = "ucf_train_npy" if train else "ucf_test_npy"
    fallback = "concat_UCF.npy" if train else "Concat_test_10.npy"
    npy_path = getattr(args, attr, "")
    if not npy_path:
        npy_path = os.path.join("./dataset", "ucfcrime", fallback)
    if not _exists(npy_path):
        return None, None, None
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim < 1:
        return None, npy_path, tuple(arr.shape)
    return int(arr.shape[-1]), npy_path, tuple(arr.shape)


def infer_feature_info(args, train=True) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, ...]]]:
    data_mode = str(getattr(args, "data_mode", "single_video")).lower()
    if data_mode == "concat":
        return _concat_feature_info(args, train=train)

    dataset_key = normalize_dataset_name(args.dataset_name)
    dataset_root = _resolve_dataset_root(args, dataset_key)
    feature_root = _resolve_feature_root(args, dataset_key, dataset_root)
    split_name = (
        getattr(args, "train_split_file", "train_split_10crop.txt")
        if train
        else getattr(args, "test_split_file", "test_split_10crop.txt")
    )
    split_path = _resolve_split_file(
        dataset_root,
        dataset_key,
        split_name,
        ["train_split_10crop.txt", "train_split.txt"] if train else ["test_split_10crop.txt", "test_split.txt"],
    )
    if not _exists(split_path):
        return None, None, None
    names = _read_split_list(split_path)
    suffix = getattr(args, "feature_suffix", ".npy")
    for name in names:
        path = _feature_file(feature_root, name, suffix)
        if _exists(path):
            arr = _safe_load_feature(path)
            return int(arr.shape[-1]), path, tuple(arr.shape)
    return None, None, None

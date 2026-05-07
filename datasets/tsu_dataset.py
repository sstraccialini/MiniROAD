"""Full-video OAD datasets.

Supports two annotation conventions:

  TSU / Smarthome
    * actions: [class_id, start_frame, end_frame]  (integer video frames)
    * duration: int  (total frames in the raw video)
    * frame_stride cfg key controls how feature indices map to frames
      (default 16, matching the I3D extraction stride)

  Charades
    * actions: [class_id, start_sec, end_sec]  (float seconds)
    * duration: float  (clip length in seconds)
    * feature-to-time mapping is derived from (T_features / duration_sec)

Both datasets share the same loading infrastructure and return the same
5-tuple  (features, labels, mask, video_id, duration)  so they work with
the same collate_fn, trainer and eval code.
"""

import difflib
import json
import os
import os.path as osp
import re

import numpy as np
import torch
import torch.utils.data as data

from datasets.dataset_builder import DATA_LAYERS


def _canonical_video_id(value):
    # Normalize IDs for robust matching across whitespace/case/separator variants.
    return re.sub(r'[^a-z0-9]', '', str(value).strip().lower())


def _load_feature_file(file_path):
    extension = osp.splitext(file_path)[1].lower()
    if extension == '.npy':
        array = np.load(file_path, mmap_mode='r')
    elif extension == '.npz':
        archive = np.load(file_path, mmap_mode='r')
        array = archive[archive.files[0]]
    elif extension == '.pt':
        array = torch.load(file_path, map_location='cpu')
        if isinstance(array, torch.Tensor):
            array = array.numpy()
    else:
        raise ValueError(f'Unsupported feature file extension: {extension}')
    return np.asarray(array)


def _normalize_feature_layout(features, expected_dim):
    """Squeeze spatial singleton dimensions and ensure shape is (T, D)."""
    features = np.asarray(features)

    # Squeeze all singleton spatial dimensions that are NOT the time axis.
    while features.ndim > 2:
        squeezed = False
        for axis in range(1, features.ndim):
            if features.shape[axis] == 1:
                features = np.squeeze(features, axis=axis)
                squeezed = True
                break
        if not squeezed:
            break

    if features.ndim != 2:
        raise ValueError(f'Unexpected feature shape after squeezing: {features.shape}')

    # Ensure feature dimension is in the last position.
    if features.shape[1] != expected_dim:
        if features.shape[0] == expected_dim:
            features = features.T
        else:
            raise ValueError(
                f'Expected feature dimension {expected_dim} not found in shape {features.shape}'
            )

    return features.astype(np.float32)


class _FullVideoBase(data.Dataset):
    """Shared loading, indexing and padding logic for full-video OAD datasets.

    Subclasses must implement ``_render_labels``.
    """

    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.training = mode == 'train'
        self.config_dir = cfg.get('_config_dir', None)
        self.strict_missing_features = cfg.get('strict_missing_features', False)
        self.feature_root = self._resolve_path(cfg['feature_root'], expect_dir=True)
        self.split_file = self._resolve_path(cfg['split_file'], expect_dir=False)
        self.t_max = cfg.get('T_max', 2500)
        self.num_classes = cfg['num_classes']
        self.feature_dim = cfg.get('input_dim', 1024)
        self.subset = 'training' if self.training else 'testing'
        self.feature_index = self._build_feature_index()

        with open(self.split_file, 'r', encoding='utf-8') as handle:
            annotations = json.load(handle)

        self.entries = [
            (video_id, record)
            for video_id, record in annotations.items()
            if record.get('subset') == self.subset
        ]
        self.entries.sort(key=lambda item: item[0])
        self._filter_missing_feature_entries()

    def __len__(self):
        return len(self.entries)

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, raw_path, expect_dir):
        if raw_path is None:
            raise ValueError(
                'A required path is None. '
                'Set feature_root / split_file in the config or via --feature_root.'
            )

        raw_path = osp.expanduser(raw_path)
        candidates = [raw_path]
        if not osp.isabs(raw_path) and self.config_dir:
            candidates.append(osp.join(self.config_dir, raw_path))

        for candidate in candidates:
            candidate_abs = osp.abspath(candidate)
            if expect_dir and osp.isdir(candidate_abs):
                return candidate_abs
            if not expect_dir and osp.isfile(candidate_abs):
                return candidate_abs

        target_type = 'directory' if expect_dir else 'file'
        raise FileNotFoundError(
            f'Could not resolve {target_type} path "{raw_path}". Tried: {candidates}'
        )

    # ------------------------------------------------------------------
    # Feature index
    # ------------------------------------------------------------------

    def _build_feature_index(self):
        index = {}
        self.feature_stems = []
        self.feature_index_canonical = {}
        for entry in os.scandir(self.feature_root):
            if not entry.is_file():
                continue
            extension = osp.splitext(entry.name)[1].lower()
            if extension not in {'.npy', '.npz', '.pt'}:
                continue
            stem = osp.splitext(entry.name)[0]
            self.feature_stems.append(stem)
            index[stem] = entry.path
            index[stem.lower()] = entry.path
            self.feature_index_canonical[_canonical_video_id(stem)] = entry.path
        return index

    def _resolve_feature_path(self, video_id):
        for key in [video_id, video_id.strip(), video_id.lower(), video_id.strip().lower()]:
            if key in self.feature_index:
                return self.feature_index[key]

        canonical_key = _canonical_video_id(video_id)
        if canonical_key in self.feature_index_canonical:
            return self.feature_index_canonical[canonical_key]

        canonical_candidates = [
            stem for stem in self.feature_stems
            if canonical_key and canonical_key in _canonical_video_id(stem)
        ]
        if len(canonical_candidates) == 1:
            return self.feature_index[canonical_candidates[0]]

        for ext in ('.npy', '.npz', '.pt'):
            candidate = osp.join(self.feature_root, f'{video_id}{ext}')
            if osp.exists(candidate):
                return candidate

        close_matches = difflib.get_close_matches(video_id, self.feature_stems, n=3, cutoff=0.7)
        raise FileNotFoundError(
            f'Could not find features for "{video_id}" under "{self.feature_root}". '
            f'Indexed {len(self.feature_stems)} files. '
            f'Closest: {close_matches}. Canonical contains: {canonical_candidates[:3]}.'
        )

    def _has_feature(self, video_id):
        for key in [video_id, video_id.strip(), video_id.lower(), video_id.strip().lower()]:
            if key in self.feature_index:
                return True
        canonical_key = _canonical_video_id(video_id)
        if canonical_key in self.feature_index_canonical:
            return True
        return any(
            canonical_key and canonical_key in _canonical_video_id(stem)
            for stem in self.feature_stems
        )

    def _filter_missing_feature_entries(self):
        filtered_entries, missing_ids = [], []
        for video_id, record in self.entries:
            if self._has_feature(video_id):
                filtered_entries.append((video_id, record))
            else:
                missing_ids.append(video_id)

        if missing_ids:
            msg = (
                f'[{self.__class__.__name__}-{self.subset}] '
                f'Missing feature files for {len(missing_ids)} videos. '
                f'Examples: {missing_ids[:10]}'
            )
            if self.strict_missing_features:
                raise FileNotFoundError(msg)
            print(f'{msg}. These samples will be skipped.')

        self.entries = filtered_entries

    # ------------------------------------------------------------------
    # Per-epoch hook (no-op for full-video datasets)
    # ------------------------------------------------------------------

    def _init_features(self):
        """No-op.

        THUMOS/TVSeries datasets rebuild their sliding-window cache here each
        epoch. Full-video datasets (TSU, Charades) return the padded video
        per __getitem__, so there is no per-epoch cache to refresh.
        """
        return

    # ------------------------------------------------------------------
    # Label rendering — implemented by subclasses
    # ------------------------------------------------------------------

    def _render_labels(self, actions, feature_length, record):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        video_id, record = self.entries[index]
        feature_path = self._resolve_feature_path(video_id)
        features = _load_feature_file(feature_path)
        features = _normalize_feature_layout(features, self.feature_dim)
        feature_length = min(features.shape[0], self.t_max)
        features = features[:feature_length]
        labels, mask = self._render_labels(record.get('actions', []), features.shape[0], record)

        padded_features = np.zeros((self.t_max, features.shape[1]), dtype=np.float32)
        padded_features[:features.shape[0]] = features

        return (
            torch.from_numpy(padded_features),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
            video_id,
            # Duration stored as float for compatibility with both datasets:
            # TSU → raw integer frames; Charades → float seconds.
            float(record.get('duration', 0)),
        )


# ======================================================================
# TSU / Smarthome  (actions in video frames, stride-based mapping)
# ======================================================================

@DATA_LAYERS.register('TSU')
class TSUDataset(_FullVideoBase):
    """TSU / Smarthome dataset.

    Actions are stored as [class_id, start_frame, end_frame] using raw video
    frame indices. Each I3D or CLIP feature corresponds to ``tsu_frame_stride``
    consecutive video frames (default 16 for both I3D and CLIP extractions).
    """

    def __init__(self, cfg, mode='train'):
        self.frame_stride = cfg.get('tsu_frame_stride', 16)
        super().__init__(cfg, mode)

    def _render_labels(self, actions, feature_length, record):
        sequence_length = min(feature_length, self.t_max)
        labels = np.zeros((self.t_max, self.num_classes), dtype=np.float32)
        mask = np.zeros((self.t_max,), dtype=np.float32)
        mask[:sequence_length] = 1.0

        for class_id, start_frame, end_frame in actions:
            start_index = max(0, int(start_frame // self.frame_stride))
            end_index = min(sequence_length - 1, int(end_frame // self.frame_stride))
            if end_index >= start_index:
                labels[start_index:end_index + 1, class_id] = 1.0

        return labels, mask


# ======================================================================
# Charades  (actions in seconds, time-based mapping)
# ======================================================================

@DATA_LAYERS.register('CHARADES')
class CharadesDataset(_FullVideoBase):
    """Charades dataset.

    Actions are stored as [class_id, start_sec, end_sec].  Features per second
    is derived on-the-fly as  T_features / duration_sec  so the mapping is
    exact regardless of the extraction frame rate.

    157 action classes, all valid (no background class to skip).
    """

    def _render_labels(self, actions, feature_length, record):
        sequence_length = min(feature_length, self.t_max)
        labels = np.zeros((self.t_max, self.num_classes), dtype=np.float32)
        mask = np.zeros((self.t_max,), dtype=np.float32)
        mask[:sequence_length] = 1.0

        duration_sec = float(record.get('duration', 0.0))
        if duration_sec <= 0.0 or feature_length <= 0:
            return labels, mask

        fps_feat = feature_length / duration_sec  # features per second

        for class_id, start_sec, end_sec in actions:
            start_index = max(0, int(start_sec * fps_feat))
            end_index = min(sequence_length - 1, int(end_sec * fps_feat))
            if end_index >= start_index:
                labels[start_index:end_index + 1, class_id] = 1.0

        return labels, mask

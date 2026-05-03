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
        raise ValueError(f'Unsupported TSU feature extension: {extension}')
    return np.asarray(array)


def _normalize_feature_layout(features, expected_dim):
    features = np.asarray(features)
    
    # Squeeze all singleton spatial dimensions (excluding the first dimension which is typically T or C)
    while features.ndim > 2:
        squeezed = False
        for axis in range(1, features.ndim):
            if features.shape[axis] == 1:
                features = np.squeeze(features, axis=axis)
                squeezed = True
                break
        if not squeezed:
            break
    
    # Verify we now have 2D
    if features.ndim != 2:
        raise ValueError(f'Unexpected TSU feature shape: {features.shape}')
    
    # Ensure feature dimension is in the last position
    if features.shape[1] != expected_dim:
        if features.shape[0] == expected_dim:
            features = features.T
        else:
            raise ValueError(f'Expected feature dimension {expected_dim} not found in shape {features.shape}')
    
    return features.astype(np.float32)


@DATA_LAYERS.register('TSU')
class TSUDataset(data.Dataset):

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
        self.frame_stride = cfg.get('tsu_frame_stride', 6)
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

    def _resolve_path(self, raw_path, expect_dir):
        if raw_path is None:
            raise ValueError('TSU path cannot be None. Please set this in the config or CLI arguments.')

        raw_path = osp.expanduser(raw_path)
        candidates = []
        if osp.isabs(raw_path):
            candidates.append(raw_path)
        else:
            candidates.append(raw_path)
            if self.config_dir:
                candidates.append(osp.join(self.config_dir, raw_path))

        for candidate in candidates:
            candidate_abs = osp.abspath(candidate)
            if expect_dir and osp.isdir(candidate_abs):
                return candidate_abs
            if not expect_dir and osp.isfile(candidate_abs):
                return candidate_abs

        target_type = 'directory' if expect_dir else 'file'
        raise FileNotFoundError(f'Could not resolve TSU {target_type} path "{raw_path}". Tried: {candidates}')

    def _build_feature_index(self):
        # Index available feature files once to avoid repeated glob calls in __getitem__.
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
        keys = [video_id, video_id.strip(), video_id.lower(), video_id.strip().lower()]
        for key in keys:
            if key in self.feature_index:
                return self.feature_index[key]

        canonical_key = _canonical_video_id(video_id)
        if canonical_key in self.feature_index_canonical:
            return self.feature_index_canonical[canonical_key]

        canonical_candidates = []
        for stem in self.feature_stems:
            canonical_stem = _canonical_video_id(stem)
            if canonical_key and canonical_key in canonical_stem:
                canonical_candidates.append(stem)
        if len(canonical_candidates) == 1:
            matched_stem = canonical_candidates[0]
            return self.feature_index[matched_stem]

        preferred = [
            osp.join(self.feature_root, f'{video_id}.npy'),
            osp.join(self.feature_root, f'{video_id}.npz'),
            osp.join(self.feature_root, f'{video_id}.pt'),
        ]
        for candidate in preferred:
            if osp.exists(candidate):
                return candidate

        sample_names = sorted(set(self.feature_stems))[:5]
        close_matches = difflib.get_close_matches(video_id, self.feature_stems, n=3, cutoff=0.7)
        raise FileNotFoundError(
            f'Could not find TSU features for "{video_id}" under "{self.feature_root}". '
            f'Indexed {len(self.feature_stems)} files. '
            f'Closest stems: {close_matches}. Canonical contains matches: {canonical_candidates[:3]}. '
            f'Sample stems: {sample_names}'
        )

    def _has_feature(self, video_id):
        keys = [video_id, video_id.strip(), video_id.lower(), video_id.strip().lower()]
        for key in keys:
            if key in self.feature_index:
                return True

        canonical_key = _canonical_video_id(video_id)
        if canonical_key in self.feature_index_canonical:
            return True

        for stem in self.feature_stems:
            canonical_stem = _canonical_video_id(stem)
            if canonical_key and canonical_key in canonical_stem:
                return True

        return False

    def _filter_missing_feature_entries(self):
        filtered_entries = []
        missing_video_ids = []
        for video_id, record in self.entries:
            if self._has_feature(video_id):
                filtered_entries.append((video_id, record))
            else:
                missing_video_ids.append(video_id)

        if missing_video_ids:
            missing_preview = missing_video_ids[:10]
            message = (
                f'[TSU-{self.subset}] Missing feature files for {len(missing_video_ids)} videos. '
                f'Examples: {missing_preview}'
            )
            if self.strict_missing_features:
                raise FileNotFoundError(message)
            print(f'{message}. These samples will be skipped.')

        self.entries = filtered_entries

    def _init_features(self):
        """No-op for TSU.

        THUMOS/TVSeries datasets rebuild their sliding-window cache here each
        epoch. TSU returns the full padded video per __getitem__, so there is
        no per-epoch feature resampling to perform. This method only exists so
        main.py can call trainloader.dataset._init_features() safely.
        """
        return

    def _render_labels(self, actions, feature_length):
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

    def __getitem__(self, index):
        video_id, record = self.entries[index]
        feature_path = self._resolve_feature_path(video_id)
        features = _load_feature_file(feature_path)
        features = _normalize_feature_layout(features, self.feature_dim)
        feature_length = min(features.shape[0], self.t_max)
        features = features[:feature_length]
        labels, mask = self._render_labels(record.get('actions', []), features.shape[0])

        padded_features = np.zeros((self.t_max, features.shape[1]), dtype=np.float32)
        padded_features[:features.shape[0]] = features

        return (
            torch.from_numpy(padded_features),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
            video_id,
            int(record.get('duration', 0)),
        )
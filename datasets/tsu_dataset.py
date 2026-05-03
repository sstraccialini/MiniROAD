import glob
import json
import os.path as osp

import numpy as np
import torch
import torch.utils.data as data

from datasets.dataset_builder import DATA_LAYERS


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
    if features.ndim == 4 and features.shape[-1] == 1 and features.shape[-2] == 1:
        features = np.squeeze(features, axis=(2, 3))
    if features.ndim == 3 and features.shape[-1] == 1:
        features = np.squeeze(features, axis=-1)
    if features.ndim != 2:
        raise ValueError(f'Unexpected TSU feature shape: {features.shape}')
    if features.shape[-1] != expected_dim and features.shape[0] == expected_dim:
        features = features.T
    return features.astype(np.float32)


@DATA_LAYERS.register('TSU')
class TSUDataset(data.Dataset):

    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.training = mode == 'train'
        self.feature_root = cfg['feature_root']
        self.split_file = cfg['split_file']
        self.t_max = cfg.get('T_max', 2500)
        self.num_classes = cfg['num_classes']
        self.feature_dim = cfg.get('input_dim', 1024)
        self.frame_stride = cfg.get('tsu_frame_stride', 6)
        self.subset = 'training' if self.training else 'testing'

        with open(self.split_file, 'r', encoding='utf-8') as handle:
            annotations = json.load(handle)

        self.entries = [
            (video_id, record)
            for video_id, record in annotations.items()
            if record.get('subset') == self.subset
        ]
        self.entries.sort(key=lambda item: item[0])

    def __len__(self):
        return len(self.entries)

    def _resolve_feature_path(self, video_id):
        preferred = [osp.join(self.feature_root, f'{video_id}.npy'),
                     osp.join(self.feature_root, f'{video_id}.npz'),
                     osp.join(self.feature_root, f'{video_id}.pt')]
        for candidate in preferred:
            if osp.exists(candidate):
                return candidate
        matches = glob.glob(osp.join(self.feature_root, f'{video_id}.*'))
        if matches:
            return matches[0]
        raise FileNotFoundError(f'Could not find TSU features for {video_id} under {self.feature_root}')

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
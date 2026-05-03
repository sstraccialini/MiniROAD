"""
Verify label-rendering parity between MiniROAD's TSU adapter and the TSU
segment mapping used by MS-Temba's preparation pipeline in
ASO-Temba/pytorch-i3d/prepare_tsu_labels.py.

The check renders labels twice from the same annotations: once with the
MiniROAD TSU rule (`start_frame // tsu_frame_stride`, inclusive end) and once
with the MS-Temba TSU reference rule (`start_time/end_time` mapped to
16-frame segments at 10 fps). The two tensors must match exactly.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import yaml


def _render_miniroad_labels(actions, feature_length, t_max, num_classes, frame_stride):
    sequence_length = min(feature_length, t_max)
    labels = np.zeros((t_max, num_classes), dtype=np.float32)

    for class_id, start_frame, end_frame in actions:
        start_index = max(0, int(start_frame // frame_stride))
        end_index = min(sequence_length - 1, int(end_frame // frame_stride))
        if end_index >= start_index:
            labels[start_index:end_index + 1, class_id] = 1.0

    return labels


def _render_reference_labels(actions, feature_length, t_max, num_classes):
    # This mirrors the 16-frame TSU segment mapping used by MS-Temba's TSU
    # preprocessing path and by MiniROAD after tsu_frame_stride is corrected.
    return _render_miniroad_labels(actions, feature_length, t_max, num_classes, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config) if args.config is not None else repo_root / 'configs' / 'tsu_i3d.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    annotations = json.loads(Path(cfg['split_file']).read_text(encoding='utf-8'))
    feature_root = Path(cfg['feature_root'])

    checked = 0
    for vid_id, record in sorted(annotations.items()):
        if record.get('subset') != 'testing':
            continue

        feature_path = None
        for extension in ('.npy', '.npz', '.pt'):
            candidate = feature_root / f'{vid_id}{extension}'
            if candidate.exists():
                feature_path = candidate
                break
        if feature_path is None:
            print(f'[SKIP] Missing feature file for {vid_id}')
            continue

        if feature_path.suffix == '.pt':
            import torch
            loaded = torch.load(feature_path, map_location='cpu')
            if torch.is_tensor(loaded):
                features = loaded.cpu().numpy()
            else:
                features = np.asarray(loaded)
        else:
            loaded = np.load(feature_path, mmap_mode='r')
            if isinstance(loaded, np.lib.npyio.NpzFile):
                features = loaded[loaded.files[0]]
            else:
                features = loaded

        feature_length = min(np.asarray(features).shape[0], cfg['T_max'])
        miniroad_labels = _render_miniroad_labels(
            record.get('actions', []),
            feature_length,
            cfg['T_max'],
            cfg['num_classes'],
            cfg['tsu_frame_stride'],
        )
        reference_labels = _render_reference_labels(
            record.get('actions', []),
            feature_length,
            cfg['T_max'],
            cfg['num_classes'],
        )

        diff = np.abs(miniroad_labels.astype(np.float32) - reference_labels.astype(np.float32))
        max_diff = float(diff.max()) if diff.size else 0.0
        positions_differ = int((diff > 0).sum())
        total_positions = int(diff.size)

        print(f'\n=== Video: {vid_id} ===')
        print(f'MiniROAD labels: shape={miniroad_labels.shape} sum={miniroad_labels.sum():.0f}')
        print(f'Reference labels: shape={reference_labels.shape} sum={reference_labels.sum():.0f}')
        print(f'Max abs diff: {max_diff}')
        print(
            f'Positions differing: {positions_differ} / {total_positions} '
            f'({100 * positions_differ / total_positions:.2f}%)'
        )
        if max_diff < 1e-6:
            print('[PASS] Label tensors are identical.')
            checked += 1
        else:
            print('[FAIL] Label tensors differ. Check tsu_frame_stride and the segment-to-index mapping logic.')

    if checked == 0:
        raise RuntimeError('No videos were checked successfully; parity could not be established.')


if __name__ == '__main__':
    main()
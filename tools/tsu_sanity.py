import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from criterions import build_criterion
from datasets import build_data_loader
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--feature_root', required=True)
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    cfg['feature_root'] = args.feature_root

    loader = build_data_loader(cfg, mode='train')
    features, labels, mask, video_ids, durations = next(iter(loader))
    print('features', tuple(features.shape), features.dtype)
    print('labels', tuple(labels.shape), labels.dtype)
    print('mask', tuple(mask.shape), mask.dtype, 'mask_sum', float(mask.sum().item()))
    print('active_classes', int((labels.sum(dim=1) > 0).any(dim=0).sum().item()))
    assert mask.sum().item() <= cfg['T_max']
    mask_bool = (mask == 0)
    padded_label_rows = labels[mask_bool]
    assert torch.all(padded_label_rows == 0), (
        f'Padded positions have non-zero labels: max={float(padded_label_rows.max().item()) if padded_label_rows.numel() else 0.0}'
    )

    with open(cfg['split_file'], 'r', encoding='utf-8') as handle:
        annotations = json.load(handle)
    first_video = video_ids[0]
    class_id, start_frame, end_frame = annotations[first_video]['actions'][0]
    frame_stride = cfg.get('tsu_frame_stride', 6)
    start_index = max(0, int(start_frame // frame_stride))
    end_index = min(cfg['T_max'] - 1, int(end_frame // frame_stride))
    print('first_action', [class_id, start_frame, end_frame], 'mapped_range', (start_index, end_index))
    assert torch.all(labels[0, start_index:end_index + 1, class_id] == 1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg, device)
    criterion = build_criterion(cfg, device)
    model.train()
    rgb_input = features[:1].to(device)
    # Keep the TSU runtime path consistent with train/eval: MiniROAD runs in
    # RGB-only mode here, but the API still expects a flow tensor.
    flow_input = rgb_input.new_zeros(rgb_input.size(0), rgb_input.size(1), rgb_input.size(2))
    target = labels[:1].to(device)
    step_mask = mask[:1].to(device)
    out_dict = model(rgb_input, flow_input)
    loss = criterion(out_dict, target, step_mask)
    loss.backward()
    print('single_step_loss', float(loss.item()))


if __name__ == '__main__':
    main()
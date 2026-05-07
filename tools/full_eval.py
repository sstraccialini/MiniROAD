"""Comprehensive single-command evaluation for MiniROAD on TSU.

Reports the same schema as MS-Temba's `vim/streaming_inference.py` so the two
models can be compared side-by-side. Block metrics are omitted because
MiniROAD has a single classification head.

Example
-------
    python tools/full_eval.py \\
        --config configs/tsu_i3d.yaml \\
        --checkpoint output/.../best_xx.xx.pth \\
        --feature_root ../ASO-Temba/data/tsu_features_i3d
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from datasets import build_data_loader  # noqa: E402
from model import build_model  # noqa: E402


# Duration unit handling
# TSU:     duration field is in raw video frames  → needs fps conversion
# Charades: duration field is already in seconds  → use directly
# Controlled by config keys:  duration_unit ('frames' | 'seconds'),  video_fps (default 25)
_DURATION_UNIT_FRAMES = 'frames'
_DURATION_UNIT_SECONDS = 'seconds'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feature_root', default=None)
    parser.add_argument('--stream_chunk_size', type=int, default=1,
                        help='0 disables the streaming demo.')
    parser.add_argument('--streaming_demo_n', type=int, default=2,
                        help='Number of videos to time in streaming mode.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    cfg['_config_dir'] = os.path.abspath(os.path.dirname(args.config))
    if args.feature_root is not None:
        cfg['feature_root'] = args.feature_root
    cfg.setdefault('no_rgb', False)
    cfg.setdefault('no_flow', True)
    return cfg


def offline_forward(model, features):
    # Mirrors MROAD.forward but skips the eval-time softmax so callers receive
    # raw logits (AP is rank-based, but BCE loss needs logits).
    x = model.layer1(features)
    h0 = torch.zeros(model.num_layers, features.size(0), model.hidden_dim,
                     device=features.device)
    ht, _ = model.gru(x, h0)
    ht = model.relu(ht)
    return model.f_classification(ht)


def streaming_forward(model, features, chunk_size):
    # Chunk-by-chunk causal forward. GRU hidden state is carried externally
    # across chunks; numerically identical to the offline forward.
    B, T, _ = features.shape
    h = torch.zeros(model.num_layers, B, model.hidden_dim, device=features.device)
    all_logits = []
    chunk_times = []
    first_chunk_s = None
    step = max(chunk_size, 1)
    for start in range(0, T, step):
        chunk = features[:, start:start + step, :]
        t0 = time.time()
        x = model.layer1(chunk)
        ht, h = model.gru(x, h)
        ht = model.relu(ht)
        logits = model.f_classification(ht)
        elapsed = time.time() - t0
        chunk_times.append(elapsed)
        if first_chunk_s is None:
            first_chunk_s = elapsed
        all_logits.append(logits)
    return torch.cat(all_logits, dim=1), chunk_times, first_chunk_s


def sampled_25(probs, labels, mask):
    # Charades-style 25-frame sampling, replicates ASO-Temba/vim/utils.py:sampled_25.
    valid_t = int(mask.sum())
    if valid_t <= 25:
        return None
    p1_ = probs[:valid_t]
    l1_ = labels[:valid_t]
    sc = valid_t / 25.0
    p1 = p1_[1::int(sc)][:25]
    l1 = l1_[1::int(sc)][:25]
    return p1, l1


def mean_ap_percent(preds_concat, labels_concat):
    # Per-class AP averaged over classes with non-zero AP, expressed as a
    # percentage. Matches MS-Temba's `sum(100*apm.value()) / nonzero_count`.
    K = preds_concat.shape[1]
    aps = []
    for k in range(K):
        if labels_concat[:, k].sum() <= 0:
            continue
        ap = average_precision_score(labels_concat[:, k], preds_concat[:, k])
        if ap > 0:
            aps.append(ap)
    return float(np.mean(aps)) * 100 if aps else 0.0


def _mean(rows, key):
    vals = [r[key] for r in rows
            if isinstance(r[key], float) and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float('nan')


def main():
    args = parse_args()
    cfg = load_config(args)

    device = torch.device(args.device)
    duration_unit = cfg.get('duration_unit', _DURATION_UNIT_SECONDS)
    video_fps = float(cfg.get('video_fps', 25.0))
    testloader = build_data_loader(cfg, mode='test')

    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_parameters = sum(p.numel() for p in model.parameters())

    full_preds_list, full_labels_list = [], []
    sampled_preds_list, sampled_labels_list = [], []
    per_video_loss_sum = 0.0
    num_videos = 0
    offline_rows = []
    stream_rows = []

    eval_start = time.time()
    with torch.no_grad():
        for batch in testloader:
            features, labels, mask, video_ids, durations = batch
            features = features.to(device)
            labels_gpu = labels.to(device)
            mask_gpu = mask.to(device)
            B, T, _ = features.shape
            raw_duration = float(durations[0])
            if duration_unit == _DURATION_UNIT_FRAMES:
                duration_s = raw_duration / video_fps if raw_duration > 0 else float('nan')
            else:
                duration_s = raw_duration if raw_duration > 0 else float('nan')

            t0 = time.time()
            logits = offline_forward(model, features)
            offline_elapsed = time.time() - t0

            probs = torch.sigmoid(logits) * mask_gpu.unsqueeze(2)

            bce = F.binary_cross_entropy_with_logits(logits, labels_gpu, reduction='none')
            bce = bce * mask_gpu.unsqueeze(2)
            mask_sum = mask_gpu.sum().clamp_min(1.0)
            per_video_loss_sum += (bce.sum() / mask_sum).item()
            num_videos += 1

            p_np = probs[0].cpu().numpy()
            l_np = labels[0].numpy()
            m_np = mask[0].numpy()
            full_preds_list.append(p_np)
            full_labels_list.append(l_np)

            sampled = sampled_25(p_np, l_np, m_np)
            if sampled is not None:
                sampled_preds_list.append(sampled[0])
                sampled_labels_list.append(sampled[1])

            offline_rows.append({
                'inference_time_s': offline_elapsed,
                'per_frame_latency_ms': offline_elapsed / T * 1000 if T > 0 else float('nan'),
                'real_time_factor': offline_elapsed / duration_s if np.isfinite(duration_s) and duration_s > 0 else float('nan'),
            })

            if args.stream_chunk_size > 0 and len(stream_rows) < args.streaming_demo_n:
                _, chunk_times, first_chunk_s = streaming_forward(
                    model, features, args.stream_chunk_size
                )
                stream_elapsed = sum(chunk_times)
                stream_rows.append({
                    'per_frame_latency_ms': stream_elapsed / T * 1000 if T > 0 else float('nan'),
                    'first_chunk_latency_ms': first_chunk_s * 1000,
                    'real_time_factor': stream_elapsed / duration_s if np.isfinite(duration_s) and duration_s > 0 else float('nan'),
                })

    eval_time_seconds = time.time() - eval_start

    full_preds = np.concatenate(full_preds_list, axis=0)
    full_labels = np.concatenate(full_labels_list, axis=0)
    per_frame_mAP = mean_ap_percent(full_preds, full_labels)

    if sampled_preds_list:
        sampled_preds = np.concatenate(sampled_preds_list, axis=0)
        sampled_labels = np.concatenate(sampled_labels_list, axis=0)
        sampled_mAP_25_value = mean_ap_percent(sampled_preds, sampled_labels)
    else:
        sampled_mAP_25_value = float('nan')

    val_loss = per_video_loss_sum / max(num_videos, 1)

    if cfg.get('no_flow', True) and not cfg.get('no_rgb', False):
        mode = 'rgb'
    elif cfg.get('no_rgb', False) and not cfg.get('no_flow', True):
        mode = 'flow'
    else:
        mode = 'rgb+flow'

    report = {
        'model_path':                       args.checkpoint,
        'dataset':                          cfg['data_name'].lower(),
        'mode':                             mode,
        'num_classes':                      cfg['num_classes'],
        'num_parameters':                   num_parameters,
        'eval_time_seconds':                round(eval_time_seconds, 4),
        'val_loss':                         round(val_loss, 4),
        'per_frame_mAP':                    round(per_frame_mAP, 4),
        'sampled_mAP_25':                   round(sampled_mAP_25_value, 4),
        'offline_avg_inference_time_s':     round(_mean(offline_rows, 'inference_time_s'), 4),
        'offline_avg_per_frame_latency_ms': round(_mean(offline_rows, 'per_frame_latency_ms'), 4),
        'offline_avg_real_time_factor':     round(_mean(offline_rows, 'real_time_factor'), 4),
        'stream_chunk_size':                args.stream_chunk_size,
        'stream_num_demo_videos':           len(stream_rows),
        'stream_avg_per_frame_latency_ms':  round(_mean(stream_rows, 'per_frame_latency_ms'), 4),
        'stream_avg_first_chunk_latency_ms':round(_mean(stream_rows, 'first_chunk_latency_ms'), 4),
        'stream_avg_real_time_factor':      round(_mean(stream_rows, 'real_time_factor'), 4),
    }

    width = max(len(k) for k in report)
    for k, v in report.items():
        print(f'{k.ljust(width)} : {v}')


if __name__ == '__main__':
    main()

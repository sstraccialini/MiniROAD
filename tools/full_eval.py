"""
tools/full_eval.py
==================
Single-command evaluation for MiniROAD on TSU / Charades.

Reports the same metric schema as MS-Temba's streaming_inference.py and saves
identical output files, so results from the two models can be placed
side-by-side without any post-processing.

Key metric definitions (matching MS-Temba exactly)
---------------------------------------------------
per_frame_mAP
    Mean AP over all valid (unpadded) frames across the full test set.
    Probabilities: sigmoid(logits) × mask.
    Averaging: mean over classes whose AP > 0  (matches APMeter behavior).

sampled_mAP_25
    Same AP computation but on 25 frames sampled uniformly from each video's
    valid frames.  Only videos with valid_t > 25 contribute.

offline_avg_per_frame_latency_ms
    inference_time / T × 1000, where T is the full padded sequence length.
    Averaged over all test videos.

stream_avg_first_chunk_latency_ms   ("ms to first prediction")
    Wall-clock time for the very first GRU chunk (chunk_size frames).
    This is the minimum latency before the model can emit any prediction.

stream_avg_per_frame_latency_ms     ("ms per frame")
    Total streaming time / T × 1000.  Averaged over the demo videos.

stream_avg_mean_chunk_time_ms
    Mean wall-clock time per chunk over the demo videos.

offline_avg_real_time_factor
stream_avg_real_time_factor
    inference_time / video_duration_s.  < 1.0 = faster than real-time.

Saved outputs (identical structure to MS-Temba)
------------------------------------------------
<output_dir>/evaluation_metrics.csv     headline metrics (one row per key)
<output_dir>/evaluation_metrics.pkl     same dict as pickle
<output_dir>/eval_full_probs.pkl        per-video probability arrays
<output_dir>/offline_latency.csv        per-video offline timing (all videos)
<output_dir>/streaming_latency.csv      per-video streaming timing (demo subset)

Note on blocks
--------------
MiniROAD has a single classification head; block metrics are omitted.
All other keys are present and comparable with MS-Temba output.

Usage
-----
# TSU I3D  (accuracy + streaming demo, 50 videos)
python tools/full_eval.py \\
    --config configs/tsu_i3d.yaml \\
    --checkpoint output/.../ckpts/best_xx.pth \\
    --feature_root ../ASO-Temba/data/tsu_features_i3d \\
    --stream_chunk_size 1 --streaming_demo_n 50 \\
    --output_dir output/eval_tsu_i3d_chunk1

# Accuracy only (skip streaming demo):
python tools/full_eval.py \\
    --config configs/tsu_i3d.yaml \\
    --checkpoint output/.../ckpts/best_xx.pth \\
    --feature_root ../ASO-Temba/data/tsu_features_i3d \\
    --stream_chunk_size 0
"""

import argparse
import csv
import os
import pickle
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

from datasets import build_data_loader   # noqa: E402
from model import build_model            # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='MiniROAD evaluation — matches MS-Temba streaming_inference.py schema'
    )
    p.add_argument('--config',         required=True,  help='YAML config file')
    p.add_argument('--checkpoint',     required=True,  help='Path to .pth checkpoint')
    p.add_argument('--feature_root',   default=None,   help='Override feature_root in config')
    p.add_argument('--stream_chunk_size', type=int, default=1,
                   help='Streaming chunk size in frames (0 = skip streaming demo)')
    p.add_argument('--streaming_demo_n', type=int, default=10,
                   help='Number of videos for streaming latency demo (-1 = all)')
    p.add_argument('--output_dir',     default=None,
                   help='Directory to save CSVs and PKLs (default: checkpoint dir)')
    p.add_argument('--device',         default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_dir'] = os.path.abspath(os.path.dirname(args.config))
    if args.feature_root is not None:
        cfg['feature_root'] = args.feature_root
    cfg.setdefault('no_rgb', False)
    cfg.setdefault('no_flow', True)
    return cfg


# ---------------------------------------------------------------------------
# Forward passes (raw logits — monotonic transforms don't affect AP)
# ---------------------------------------------------------------------------

def offline_forward(model, features):
    """Full-sequence causal forward — numerically identical to streaming."""
    x = model.layer1(features)
    h0 = torch.zeros(model.num_layers, features.size(0), model.hidden_dim,
                     device=features.device)
    ht, _ = model.gru(x, h0)
    ht = model.relu(ht)
    return model.f_classification(ht)           # (B, T, K) raw logits


def streaming_forward(model, features, chunk_size):
    """
    Chunk-by-chunk causal forward.  GRU state is carried externally across
    chunks; numerically identical to offline_forward for any chunk_size.

    Returns
    -------
    logits      : (B, T, K) — same as offline_forward
    chunk_times : list of per-chunk wall-clock seconds
    first_chunk_s : float — latency for the very first chunk
    """
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


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def sampled_25(probs, labels, mask):
    """Uniformly sample 25 frames from valid region.  Mirrors MS-Temba utils.py."""
    valid_t = int(mask.sum())
    if valid_t <= 25:
        return None
    sc = valid_t / 25.0
    p1 = probs[:valid_t][1::int(sc)][:25]
    l1 = labels[:valid_t][1::int(sc)][:25]
    return p1, l1


def mean_ap_percent(preds, labels):
    """
    Mean AP over classes with AP > 0.
    Mirrors MS-Temba's  sum(100*apm.value()) / nonzero_count  formula.

    preds, labels : (N, K) numpy arrays
    Returns       : float (percentage)
    """
    K = preds.shape[1]
    aps = []
    for k in range(K):
        if labels[:, k].sum() <= 0:
            continue
        ap = average_precision_score(labels[:, k], preds[:, k])
        if ap > 0:
            aps.append(ap)
    return float(np.mean(aps)) * 100 if aps else 0.0


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r[key], float) and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float('nan')


# ---------------------------------------------------------------------------
# Duration → seconds
# ---------------------------------------------------------------------------

def _to_seconds(raw_duration, cfg):
    unit = cfg.get('duration_unit', 'seconds')
    fps  = float(cfg.get('video_fps', 25.0))
    if unit == 'frames':
        return raw_duration / fps if raw_duration > 0 else float('nan')
    return raw_duration if raw_duration > 0 else float('nan')


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, cfg, args, device):
    model.eval()

    full_preds_list,    full_labels_list    = [], []
    sampled_preds_list, sampled_labels_list = [], []
    full_probs = {}          # per-video sigmoid probs (for PKL)

    tot_loss = 0.0
    num_iter = 0

    offline_rows = []
    stream_rows  = []

    do_streaming = args.stream_chunk_size > 0
    demo_limit   = args.streaming_demo_n    # -1 = unlimited

    with torch.no_grad():
        for batch in dataloader:
            features, labels, mask, video_ids, durations = batch
            vid_id       = video_ids[0]
            duration_s   = _to_seconds(float(durations[0]), cfg)

            features  = features.to(device)
            labels_gpu = labels.to(device)
            mask_gpu   = mask.to(device)
            B, T, _   = features.shape

            video_fps = T / duration_s if np.isfinite(duration_s) and duration_s > 0 \
                        else float('nan')

            # ── Offline causal forward ────────────────────────────────────────
            t0 = time.time()
            logits = offline_forward(model, features)
            offline_elapsed = time.time() - t0

            # ── Probs (sigmoid + mask) ────────────────────────────────────────
            probs = torch.sigmoid(logits) * mask_gpu.unsqueeze(2)

            # ── BCE loss (masked, matching training criterion) ────────────────
            bce = F.binary_cross_entropy_with_logits(logits, labels_gpu, reduction='none')
            bce = bce * mask_gpu.unsqueeze(2)
            mask_sum = mask_gpu.sum().clamp_min(1.0)
            tot_loss += (bce.sum() / mask_sum).item()
            num_iter += 1

            # ── Accuracy accumulators ─────────────────────────────────────────
            p_np = probs[0].cpu().numpy()           # (T, K)
            l_np = labels[0].numpy()                # (T, K)
            m_np = mask[0].numpy()                  # (T,)

            full_preds_list.append(p_np)
            full_labels_list.append(l_np)

            samp = sampled_25(p_np, l_np, m_np)
            if samp is not None:
                sampled_preds_list.append(samp[0])
                sampled_labels_list.append(samp[1])

            # Store masked probs for PKL (only valid frames, transposed like MS-Temba)
            valid_t = int(m_np.sum())
            full_probs[vid_id] = p_np[:valid_t].T   # (K, valid_t)

            # ── Offline latency row ───────────────────────────────────────────
            offline_rows.append({
                'vid_id':              vid_id,
                'num_frames':          T,
                'video_duration_s':    round(duration_s, 4) if np.isfinite(duration_s) else 'nan',
                'video_fps':           round(video_fps, 2)  if np.isfinite(video_fps)  else 'nan',
                'inference_time_s':    offline_elapsed,
                'per_frame_latency_ms':offline_elapsed / T * 1000 if T > 0 else float('nan'),
                'real_time_factor':    offline_elapsed / duration_s
                                       if np.isfinite(duration_s) and duration_s > 0
                                       else float('nan'),
            })

            # ── Streaming demo ────────────────────────────────────────────────
            if do_streaming:
                want_demo = (demo_limit == -1) or (len(stream_rows) < demo_limit)
                if want_demo:
                    _, chunk_times, first_chunk_s = streaming_forward(
                        model, features, args.stream_chunk_size
                    )
                    stream_elapsed = sum(chunk_times)
                    stream_rows.append({
                        'vid_id':                  vid_id,
                        'num_frames':              T,
                        'video_duration_s':        round(duration_s, 4) if np.isfinite(duration_s) else 'nan',
                        'video_fps':               round(video_fps, 2)  if np.isfinite(video_fps)  else 'nan',
                        'chunk_size':              args.stream_chunk_size,
                        'num_chunks':              len(chunk_times),
                        'total_stream_time_s':     stream_elapsed,
                        'per_frame_latency_ms':    stream_elapsed / T * 1000 if T > 0 else float('nan'),
                        'mean_chunk_time_ms':      float(np.mean(chunk_times)) * 1000,
                        'max_chunk_time_ms':       float(np.max(chunk_times))  * 1000,
                        'first_chunk_latency_ms':  first_chunk_s * 1000,
                        'real_time_factor':        stream_elapsed / duration_s
                                                   if np.isfinite(duration_s) and duration_s > 0
                                                   else float('nan'),
                    })

    # ── Aggregate accuracy ────────────────────────────────────────────────────
    full_preds  = np.concatenate(full_preds_list,  axis=0)
    full_labels = np.concatenate(full_labels_list, axis=0)
    per_frame_mAP = mean_ap_percent(full_preds, full_labels)

    if sampled_preds_list:
        sampled_mAP_25 = mean_ap_percent(
            np.concatenate(sampled_preds_list,  axis=0),
            np.concatenate(sampled_labels_list, axis=0),
        )
    else:
        sampled_mAP_25 = float('nan')

    val_loss = tot_loss / max(num_iter, 1)

    # ── Aggregate timing ──────────────────────────────────────────────────────
    offline_summary = {
        'avg_inference_time_s':     _mean(offline_rows, 'inference_time_s'),
        'avg_per_frame_latency_ms': _mean(offline_rows, 'per_frame_latency_ms'),
        'avg_real_time_factor':     _mean(offline_rows, 'real_time_factor'),
    }
    stream_summary = {
        'avg_per_frame_latency_ms':    _mean(stream_rows, 'per_frame_latency_ms'),
        'avg_mean_chunk_time_ms':      _mean(stream_rows, 'mean_chunk_time_ms'),
        'avg_max_chunk_time_ms':       _mean(stream_rows, 'max_chunk_time_ms'),
        'avg_first_chunk_latency_ms':  _mean(stream_rows, 'first_chunk_latency_ms'),
        'avg_real_time_factor':        _mean(stream_rows, 'real_time_factor'),
    }

    return (full_probs, val_loss, per_frame_mAP, sampled_mAP_25,
            offline_rows, stream_rows, offline_summary, stream_summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    cfg    = load_config(args)
    device = torch.device(args.device)

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent.parent / 'eval_results')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data + model ──────────────────────────────────────────────────────────
    testloader = build_data_loader(cfg, mode='test')
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    # Support both plain state-dicts and MS-Temba-style checkpoint dicts
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    demo_desc = (
        'accuracy only (no streaming demo)' if args.stream_chunk_size == 0
        else f'streaming demo on {args.streaming_demo_n} videos '
             f'(chunk_size={args.stream_chunk_size})'
    )
    print(f'Model: {n_params:,} parameters')
    print(f'Mode : {demo_desc}')

    # ── Run evaluation ────────────────────────────────────────────────────────
    eval_start = time.time()
    (full_probs, val_loss, per_frame_mAP, sampled_mAP_25,
     offline_rows, stream_rows,
     offline_summary, stream_summary) = evaluate(
        model, testloader, cfg, args, device
    )
    eval_time = time.time() - eval_start

    if cfg.get('no_flow', True) and not cfg.get('no_rgb', False):
        mode = 'rgb'
    elif cfg.get('no_rgb', False) and not cfg.get('no_flow', True):
        mode = 'flow'
    else:
        mode = 'rgb+flow'

    # ── Build metrics dict (identical keys to MS-Temba streaming_inference.py) ─
    metrics = {
        'model_path':                         args.checkpoint,
        'dataset':                            cfg['data_name'].lower(),
        'mode':                               mode,
        'num_classes':                        cfg['num_classes'],
        'num_parameters':                     n_params,
        'eval_time_seconds':                  round(eval_time, 4),
        'val_loss':                           round(val_loss, 4),

        # Accuracy
        'per_frame_mAP':                      round(per_frame_mAP,  4),
        'sampled_mAP_25':                     round(sampled_mAP_25, 4),
        # (block metrics omitted — MiniROAD has a single head)

        # Offline causal timing  (all test videos)
        'offline_avg_inference_time_s':       round(offline_summary['avg_inference_time_s'],     4),
        'offline_avg_per_frame_latency_ms':   round(offline_summary['avg_per_frame_latency_ms'], 4),
        'offline_avg_real_time_factor':       round(offline_summary['avg_real_time_factor'],     4),

        # Streaming timing  (demo subset)
        'stream_chunk_size':                  args.stream_chunk_size,
        'stream_num_demo_videos':             len(stream_rows),
        'stream_avg_per_frame_latency_ms':    round(stream_summary['avg_per_frame_latency_ms'],   4),
        'stream_avg_mean_chunk_time_ms':      round(stream_summary['avg_mean_chunk_time_ms'],     4),
        'stream_avg_max_chunk_time_ms':       round(stream_summary['avg_max_chunk_time_ms'],      4),
        'stream_avg_first_chunk_latency_ms':  round(stream_summary['avg_first_chunk_latency_ms'], 4),
        'stream_avg_real_time_factor':        round(stream_summary['avg_real_time_factor'],       4),
    }

    # ── Print (same format as MS-Temba) ───────────────────────────────────────
    width = max(len(k) for k in metrics)
    print('\n' + '=' * (width + 14))
    print('EVALUATION SUMMARY')
    print('=' * (width + 14))
    for k, v in metrics.items():
        fmt = f'{v:.4f}' if isinstance(v, float) and np.isfinite(v) else str(v)
        print(f'  {k.ljust(width)} : {fmt}')
    print('=' * (width + 14))

    # ── Save files (same names as MS-Temba) ───────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f'\nMetrics CSV  : {csv_path}')

    pkl_path = os.path.join(args.output_dir, 'evaluation_metrics.pkl')
    pickle.dump(metrics, open(pkl_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f'Metrics PKL  : {pkl_path}')

    probs_path = os.path.join(args.output_dir, 'eval_full_probs.pkl')
    pickle.dump(full_probs, open(probs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f'Probs PKL    : {probs_path}')

    if offline_rows:
        p = os.path.join(args.output_dir, 'offline_latency.csv')
        _save_csv(p, offline_rows)
        print(f'Offline lat. : {p}')

    if stream_rows:
        p = os.path.join(args.output_dir, 'streaming_latency.csv')
        _save_csv(p, stream_rows)
        print(f'Stream lat.  : {p}')


if __name__ == '__main__':
    main()

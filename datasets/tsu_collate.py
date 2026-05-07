import torch


def tsu_collate_fn(batch):
    features, labels, masks, video_ids, durations = zip(*batch)
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)
    # float32 preserves both integer frame counts (TSU) and fractional seconds (Charades).
    durations = torch.as_tensor(list(durations), dtype=torch.float32)
    return features, labels, masks, list(video_ids), durations
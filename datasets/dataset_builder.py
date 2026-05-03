__all__ = [
    'build_dataset',
    'build_data_loader',
]

import torch.utils.data as data
from utils import Registry
from .tsu_collate import tsu_collate_fn

DATA_LAYERS = Registry()

def build_dataset(cfg):
    data_layer = DATA_LAYERS[f'{cfg["data_name"]}']
    return data_layer

def build_data_loader(cfg, mode):
    data_layer = build_dataset(cfg)
    collate_fn = tsu_collate_fn if cfg['data_name'] == 'TSU' else None
    data_loader = data.DataLoader(
        dataset=data_layer(cfg, mode),
        batch_size=cfg["batch_size"] if mode == 'train' else cfg["test_batch_size"],
        shuffle=True if mode == 'train' else False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return data_loader
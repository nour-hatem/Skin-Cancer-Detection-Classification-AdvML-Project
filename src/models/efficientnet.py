import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.config import CLASS_NAMES, DEVICE, LR


def build_model() -> nn.Module:
    m = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return m.to(DEVICE)


def build_criterion(counts: np.ndarray) -> nn.Module:
    weights = torch.tensor(1.0 / np.sqrt(counts + 1), dtype=torch.float32)
    weights = (weights / weights.sum() * len(CLASS_NAMES)).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for p in model.parameters() if not any(p is q for q in head_params)]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": head_params,     "lr": LR},
    ], weight_decay=1e-4)


def build_scheduler(optimizer: torch.optim.Optimizer, train_dl, epochs: int):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR * 0.1, LR],
        steps_per_epoch=len(train_dl),
        epochs=epochs,
        pct_start=0.2,
    )
import argparse
import numpy as np
import torch
import torch.nn as nn

from src.config import DEVICE, EPOCHS, MEL_IDX, SAVE_PATH
from src.data.dataset import load_dataframe, make_dataloaders
from src.models.efficientnet import build_model, build_criterion, build_optimizer, build_scheduler


def evaluate(model: nn.Module, loader, criterion: nn.Module) -> tuple:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    mel_tp, mel_fn = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out   = model(imgs)
            loss  = criterion(out, labels)
            preds = out.argmax(1)
            total_loss += loss.item() * len(imgs)
            correct    += (preds == labels).sum().item()
            n          += len(imgs)
            is_mel      = labels == MEL_IDX
            mel_tp     += (preds[is_mel] == MEL_IDX).sum().item()
            mel_fn     += (preds[is_mel] != MEL_IDX).sum().item()
    mel_recall = mel_tp / (mel_tp + mel_fn + 1e-9)
    return total_loss / n, correct / n, mel_recall


def train(data_dir: str, epochs: int = EPOCHS, save_path: str = SAVE_PATH) -> list:
    df = load_dataframe(data_dir)
    train_dl, val_dl, _, counts = make_dataloaders(df)

    model     = build_model()
    criterion = build_criterion(counts)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, train_dl, epochs)

    best_mel_recall = 0.0
    history = []

    print(f'{"Ep":>4} | {"Tr Loss":>8} {"Tr Acc":>7} | {"Vl Loss":>8} {"Vl Acc":>7} | {"Mel Rec":>7}')
    print("-" * 62)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss    += loss.item() * len(imgs)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_n       += len(imgs)

        tr_loss /= tr_n
        tr_acc   = tr_correct / tr_n
        vl_loss, vl_acc, mel_rec = evaluate(model, val_dl, criterion)
        history.append((tr_loss, tr_acc, vl_loss, vl_acc, mel_rec))

        tag = ""
        if mel_rec > best_mel_recall:
            best_mel_recall = mel_rec
            torch.save(model.state_dict(), save_path)
            tag = "  ✓"

        print(f"{ep:>4} | {tr_loss:>8.4f} {tr_acc:>7.3f} | {vl_loss:>8.4f} {vl_acc:>7.3f} | {mel_rec:>7.3f}{tag}")

    print(f"\nBest val melanoma recall: {best_mel_recall:.3f}")
    print(f"Model saved to: {save_path}")

    h        = np.array(history)
    best_ep  = int(h[:, 4].argmax()) + 1
    print(f"\nBest Epoch : {best_ep}/{epochs}")
    print(f"Train Loss : {h[best_ep-1, 0]:.4f}  |  Train Acc : {h[best_ep-1, 1]:.3f}")
    print(f"Val Loss   : {h[best_ep-1, 2]:.4f}  |  Val Acc   : {h[best_ep-1, 3]:.3f}")
    print(f"Mel Recall : {h[best_ep-1, 4]:.3f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S on HAM10000")
    parser.add_argument("--data",   required=True,        help="Path to HAM10000 root directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--save",   default=SAVE_PATH,    help="Output path for model weights")
    args = parser.parse_args()

    train(args.data, args.epochs, args.save)
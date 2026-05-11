import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import CLASS_NAMES, DEVICE, MEL_IDX, MEL_THRESHOLD, SAVE_PATH, MEAN, STD
from src.data.dataset import load_dataframe, make_dataloaders
from src.models.efficientnet import build_model


def collect_predictions(model, test_dl) -> tuple:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            probs = F.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def threshold_scan(all_probs: np.ndarray, all_labels: np.ndarray) -> None:
    mel_probs = all_probs[:, MEL_IDX]
    true_mel  = all_labels == MEL_IDX
    print(f'{"Thresh":>7} | {"Mel Recall":>10} | {"Precision":>9} | {"F1":>6}')
    print("-" * 42)
    for t in np.arange(0.10, 0.55, 0.05):
        pred_mel = mel_probs >= t
        tp   = (pred_mel &  true_mel).sum()
        fn   = (~pred_mel &  true_mel).sum()
        fp   = (pred_mel & ~true_mel).sum()
        rec  = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1   = 2 * rec * prec / (rec + prec + 1e-9)
        mark = " ← target" if rec >= 0.80 else ""
        print(f"{t:>7.2f} | {rec:>10.3f} | {prec:>9.3f} | {f1:>6.3f}{mark}")


def plot_confusion_matrix(all_probs: np.ndarray, all_labels: np.ndarray, threshold: float = MEL_THRESHOLD) -> np.ndarray:
    preds = np.argmax(all_probs, axis=1)
    preds[all_probs[:, MEL_IDX] >= threshold] = MEL_IDX

    cm      = confusion_matrix(all_labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — Melanoma Threshold = {threshold}", fontsize=13, fontweight="bold", y=1.02)

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw counts",     "d"),
        (axes[1], cm_norm, "Row-normalised", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=11)
        thresh = data.max() * 0.5
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                color = "white" if data[i, j] > thresh else "black"
                if i == MEL_IDX and j != MEL_IDX:
                    color = "#cc3333"
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=9, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.add_patch(plt.Rectangle((-0.5, MEL_IDX - 0.5), len(CLASS_NAMES), 1,
                                   fill=False, edgecolor="#cc3333", lw=1.5, linestyle="--"))

    plt.tight_layout()
    plt.show()
    return preds


def plot_roc_curves(all_probs: np.ndarray, all_labels: np.ndarray) -> None:
    y_bin  = label_binarize(all_labels, classes=list(range(len(CLASS_NAMES))))
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#1abc9c", "#9b59b6", "#3498db", "#f39c12"]

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5 if cls == "mel" else 1.2,
                label=f"{cls}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — One vs Rest", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.show()


def plot_misclassified(model, test_dl, n: int = 20) -> None:
    model.eval()
    wrong_imgs, wrong_true, wrong_pred = [], [], []
    mean_t = torch.tensor(MEAN).view(3, 1, 1)
    std_t  = torch.tensor(STD).view(3, 1, 1)

    with torch.no_grad():
        for imgs, labels in test_dl:
            probs = F.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            preds[probs[:, MEL_IDX] >= MEL_THRESHOLD] = MEL_IDX
            mask  = preds != labels.numpy()
            for idx in np.where(mask)[0]:
                if len(wrong_imgs) >= n:
                    break
                wrong_imgs.append(imgs[idx])
                wrong_true.append(labels[idx].item())
                wrong_pred.append(preds[idx])
            if len(wrong_imgs) >= n:
                break

    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    fig.suptitle("Misclassified Samples", fontsize=14, fontweight="bold")
    for ax, img_t, true_l, pred_l in zip(axes.flat, wrong_imgs, wrong_true, wrong_pred):
        img = (img_t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"True: {CLASS_NAMES[true_l]}\nPred: {CLASS_NAMES[pred_l]}", fontsize=8,
                     color="#e74c3c" if true_l == MEL_IDX or pred_l == MEL_IDX else "black")
        ax.axis("off")
    for ax in axes.flat[len(wrong_imgs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def evaluate(data_dir: str, model_path: str = SAVE_PATH) -> None:
    df = load_dataframe(data_dir)
    _, _, test_dl, counts = make_dataloaders(df)

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    all_probs, all_labels = collect_predictions(model, test_dl)

    print("\n Threshold Scan ")
    threshold_scan(all_probs, all_labels)

    print("\n Confusion Matrix ")
    preds = plot_confusion_matrix(all_probs, all_labels)

    print("\n ROC Curves ")
    plot_roc_curves(all_probs, all_labels)

    print("\n Misclassified Samples ")
    plot_misclassified(model, test_dl)

    print("\n Classification Report ")
    print(classification_report(all_labels, preds, target_names=CLASS_NAMES, digits=3))

    tp  = ((preds == MEL_IDX) & (all_labels == MEL_IDX)).sum()
    fn  = ((preds != MEL_IDX) & (all_labels == MEL_IDX)).sum()
    rec = tp / (tp + fn + 1e-9)
    acc = (preds == all_labels).sum() / len(all_labels)
    print(f"Overall accuracy : {acc:.3f}")
    print(f"Melanoma recall  : {rec:.3f}  (target >0.80)")
    print("Target met!" if rec >= 0.80 else "Below target.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on HAM10000 test set")
    parser.add_argument("--data",  required=True,      help="Path to HAM10000 root directory")
    parser.add_argument("--model", default=SAVE_PATH,  help="Path to model weights (.pth)")
    args = parser.parse_args()

    evaluate(args.data, args.model)
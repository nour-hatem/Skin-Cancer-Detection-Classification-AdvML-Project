import numpy as np
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.config import CLASS_NAMES, LABEL_MAP, IMG_SIZE, BATCH, SEED, MEAN, STD


train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

eval_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfm: transforms.Compose):
        self.paths  = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.tfm    = tfm

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.tfm(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def load_dataframe(data_dir: str) -> pd.DataFrame:
    base     = Path(data_dir)
    csv_path = next(base.rglob("HAM10000_metadata.csv"))
    df       = pd.read_csv(csv_path)

    img_lookup  = {p.stem: p for p in base.rglob("*.jpg")}
    df["path"]  = df["image_id"].map(img_lookup)
    df = df.dropna(subset=["path"]).copy()
    df["label"] = df["dx"].map(LABEL_MAP)
    df = df.drop_duplicates(subset="lesion_id", keep="first").reset_index(drop=True)
    return df


def make_dataloaders(df: pd.DataFrame) -> tuple:
    train_df, tmp_df = train_test_split(df,     test_size=0.2,  stratify=df["label"],     random_state=SEED)
    val_df,  test_df = train_test_split(tmp_df, test_size=0.5,  stratify=tmp_df["label"], random_state=SEED)

    for frame in (train_df, val_df, test_df):
        frame.reset_index(drop=True, inplace=True)

    counts   = np.bincount(train_df["label"].values, minlength=len(CLASS_NAMES)).astype(float)
    w_class  = 1.0 / np.sqrt(counts + 1)
    w_sample = w_class[train_df["label"].values]
    sampler  = WeightedRandomSampler(w_sample, num_samples=len(w_sample), replacement=True)

    train_dl = DataLoader(SkinDataset(train_df, train_tfm), batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(SkinDataset(val_df,   eval_tfm),  batch_size=BATCH, shuffle=False,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(SkinDataset(test_df,  eval_tfm),  batch_size=BATCH, shuffle=False,  num_workers=2, pin_memory=True)

    return train_dl, val_dl, test_dl, counts
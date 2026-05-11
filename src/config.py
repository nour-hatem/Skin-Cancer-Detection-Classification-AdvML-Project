import torch
import numpy as np

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 224
BATCH     = 32
EPOCHS    = 25
LR        = 3e-4
SEED      = 42

CLASS_NAMES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_MAP     = {c: i for i, c in enumerate(CLASS_NAMES)}
MEL_IDX       = LABEL_MAP["mel"]
MEL_THRESHOLD = 0.35

SAVE_PATH = "weights/skin_cancer_model.pth"

MEAN = [0.7630, 0.5456, 0.5700]
STD  = [0.1409, 0.1526, 0.1695]

torch.manual_seed(SEED)
np.random.seed(SEED)
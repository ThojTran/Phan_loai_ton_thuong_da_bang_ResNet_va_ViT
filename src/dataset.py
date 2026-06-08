"""
dataset.py — Dataset dùng chung cho ResNet18 và ViT.
Nhận transform từ ngoài, không phụ thuộc loại mô hình.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# 7 lớp HAM10000 → số nguyên
HAM_LABEL_MAP = {"nv":0,"mel":1,"bkl":2,"bcc":3,"akiec":4,"vasc":5,"df":6}
# 5 lớp DERM7PT → số nguyên
DERM_LABEL_MAP = {"bcc":0, "mel":1, "misc":2, "nevus":3, "sk":4}

class SkinDataset(Dataset):
    """
    Đọc ảnh từ CSV có 2 cột bắt buộc:
        img_path : đường dẫn tới file ảnh (tuyệt đối hoặc tương đối)
        dx       : nhãn dạng chuỗi, ví dụ 'nv', 'mel', ...
    """
    def __init__(self, csv_path: str, transform=None,label_map=None, label_col="dx"):
        csv_path            = Path(csv_path).resolve()
        self.project_root   = csv_path.parent.parent.parent  # splits/ → data/ → gốc project
        self.df             = pd.read_csv(csv_path)
        self.transform      = transform
        self.label_map = label_map or HAM_LABEL_MAP
        self.label_col = label_col
        for col in ("img_path", "dx"):
            if col not in self.df.columns:
                raise ValueError(f"CSV thiếu cột bắt buộc: '{col}'")

        # Resolve img_path tương đối tính từ thư mục gốc project
        self.df["img_path"] = self.df["img_path"].apply(
            lambda p: str((self.project_root / p).resolve())
            if not Path(p).is_absolute() else p
        )

        self.df["label"] = self.df[self.label_col].map(self.label_map)
        if self.df["label"].isna().any():
            unknown = self.df.loc[self.df["label"].isna(), "dx"].unique()
            raise ValueError(f"Nhãn không có trong LABEL_MAP: {unknown}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # convert("RGB") đảm bảo luôn 3 channel dù ảnh gốc là grayscale
        image = Image.open(row["img_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(row["label"])
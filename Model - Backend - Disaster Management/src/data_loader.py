import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DisasterDataset(Dataset):
    def __init__(self, df, class_map, sev_map, size, split):
        self.df = df.reset_index(drop=True)
        self.cmap, self.smap = class_map, sev_map
        self.tf = self.get_tf(size, split)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.filepath).convert("RGB")
        return {
            "image": self.tf(img),
            "class_label": torch.tensor(self.cmap[r.label]),
            "severity_label": torch.tensor(self.smap[r.severity]),
            "path": r.filepath,
        }

    def get_tf(self, s, split):
        norm = transforms.Normalize([0.485]*3, [0.229]*3)
        if split == "train":
            return transforms.Compose([
                transforms.Resize((s+16, s+16)),
                transforms.RandomResizedCrop(s),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), norm
            ])
        return transforms.Compose([
            transforms.Resize((s, s)),
            transforms.ToTensor(), norm
        ])

def build_mappings(df):
    return (
        {c:i for i,c in enumerate(sorted(df.label.unique()))},
        {s:i for i,s in enumerate(sorted(df.severity.unique()))}
    )

def load_csv(p): return pd.read_csv(p)
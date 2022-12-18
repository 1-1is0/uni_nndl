# %%
import pandas as pd
import albumentations as A
import torch
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import os
DIR = "./data/COVID-19 Dataset/CT/"


class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = self.df.iloc[index]["path"]
        x = cv2.imread(fr"{x}")
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if self.transform:
            x = self.transform(image=x)["image"]
        y = self.df.iloc[index]["covid"]
        return x, y


def read_data(path):
    """
    read a dir recursively
    """
    dirs = os.listdir(path)
    results = []
    for dir in dirs:
        dest = os.path.join(path, dir)
        if os.path.isdir(dest):
            results += read_data(dest)
        elif os.path.isfile(dest):
            results.append(dest)
    return results


def get_data():
    images_list = read_data(DIR)
    df = pd.DataFrame(images_list, columns=["path"])
    df["covid"] = df.path.str.split("/", expand=True)[4]
    le = LabelEncoder()
    df["covid"] = le.fit_transform(df["covid"])
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.125) # 0.125 * 0.8 = 0.1
    transform = A.Compose([
        A.Resize(64, 64),
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ])

    batch_size = 32
    shuffle = True
    num_workers = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = True if "cuda" in str(device) else False

    train_set = ImageDataset(train_df, transform)
    val_set = ImageDataset(val_df, transform)
    test_set = ImageDataset(test_df, transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    data_loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    dataset_sizes = {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }
    return data_loader, dataset_sizes

# %%
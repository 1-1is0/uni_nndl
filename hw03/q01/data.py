# %%
import pandas as pd
import albumentations as A
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
    transform = A.Compose([
        A.Resize(64, 64),
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2(),
    ])

    train_set = ImageDataset(train_df, transform)
    test_set = ImageDataset(test_df, transform)
    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=12)

    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=12)

    data_loader = {
        "train": train_loader,
        "val": test_loader,
    }

    dataset_sizes = {
        "train": len(train_set),
        "val": len(test_set)
    }
    return data_loader, dataset_sizes 

# %%

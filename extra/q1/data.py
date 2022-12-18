# %%
import pandas as pd
import numpy as np
import torch
import albumentations as A
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import os
DIR = "./data/COVID-19 Dataset/CT/"


class CardDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index].astype(np.float32)
        y = self.y[index]
        return x, y


def read_data(path):
    print(os.getcwd())
    data = pd.read_csv('./data/creditcard.csv')
    # data = pd.read_csv('extra/q1/data/creditcard.csv')
    return data

def get_percent_of_data(x, y, percent=1):
    fraud_class = 1
    normal_class = 0
    fraud_train_label = y[np.where(y==fraud_class)]
    fraud_train_data = x[np.where(y==fraud_class)]
    normal_train_label = y[np.where(y==normal_class)]
    normal_train_data = x[np.where(y==normal_class)]

    percent_of_data = int(len(fraud_train_label) * percent)
    idx = np.random.choice(len(fraud_train_label), percent_of_data, replace=False)
    xx = fraud_train_data[idx]
    yy = fraud_train_label[idx]
    label = np.concatenate((normal_train_label, yy), axis=0)
    data = np.concatenate((normal_train_data, xx), axis=0)
    return data, label

def get_data(over_sample=True, percent=1):
    data = read_data(DIR)

    data['normAmount'] = StandardScaler().fit_transform(
        data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    X = np.array(data.iloc[:, data.columns != 'Class'])
    y = np.array(data.iloc[:, data.columns == 'Class'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    if over_sample:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train.ravel())
    X_train, y_train = get_percent_of_data(X_train, y_train, percent)
    y_test = y_test.ravel()

    train_set = CardDataset(X_train, y_train)
    test_set = CardDataset(X_test, y_test)

    batch_size = 256
    shuffle = True
    num_workers = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = True if "cuda" in str(device) else False

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    data_loader = {
        "train": train_loader,
        "val": test_loader,
    }

    dataset_sizes = {
        "train": len(train_set),
        "val": len(test_set)
    }

    dataset = {
        "train": train_set,
        "val": test_set
    }
    return data_loader, dataset, dataset_sizes

# # %%
# a, b = get_data()
# # %%
# aa = a['train']
# a_size = b['train']
# for xx, yy in aa:
#     print(xx, yy)
#     break
# # %%
# print(a_size)
# ba = a_size // aa.batch_size

# # %%
# for i in range(ba):
#     if i % (ba // 4) == (ba//4)-1:
#         print(i)

# # %%
# len(aa)
# # %%
# a, b = (1, 1)
# for i in range(10):
#     c = a+b
#     b = a-b
#     a = c
# print(a, b)

# %%

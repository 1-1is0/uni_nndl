# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from data import get_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
first = True


def draw_curve(current_epoch, optimizer_name, loss_name, res):
    global first
    x_epoch = list(range(1, current_epoch+1))
    loss_train = res["loss_train"]
    loss_val = res["loss_val"]
    plt.plot(x_epoch, loss_train, 'bo-', label='train')
    plt.plot(x_epoch, loss_val, 'ro-', label='val')
    if first:
        plt.legend()
        first = False
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs',
                f'train_{optimizer_name}_{loss_name}.jpg'))


# %%


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(3, 3, 3)
        self.dense_net = models.densenet121(
            weights="DenseNet121_Weights.DEFAULT")
        # self.avg_pool_1 = nn.AvgPool2d((1, 1))
        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.batch_norm_1 = nn.BatchNorm2d(1024)
        self.dropout_1 = nn.Dropout()
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(1024, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.dropout_2 = nn.Dropout()
        self.root = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = self.dense_net.features(x)
        x = self.avg_pool_1(x)
        x = self.batch_norm_1(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = F.relu(self.dense_1(x))
        x = self.batch_norm_2(x)
        x = self.dropout_2(x)
        x = F.softmax(self.root(x), dim=1)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


# %%
data_loader, dataset_sizes = get_data()
net = Net().to(device)
net.initialize_weights()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%


def train(epochs=20):
    path = "model"
    state_file_name = f"{path}/state-{net._get_name()}-optimizer-{optimizer.__class__.__name__}-loss-{criterion.__class__.__name__}.pth"
    state_res = {}

    print(state_file_name, end=" ")
    if os.path.exists(state_file_name):
        print("exist")
        state = torch.load(state_file_name)
        net.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        state_res = state["res"]

    else:
        print("Not exist")
    res = {
        "loss_train": state_res.get("loss_train", []),
        "loss_val": state_res.get("loss_val", []),
        "epoch": state_res.get("epoch", 0),
    }

    # loop over the dataset multiple times
    for epoch in range(res["epoch"]+1, epochs):
        running_loss = 0.0
        phase_loss = 0
        for phase in ["train", "val"]:
            if phase == "train":
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode
            for i, data in enumerate(data_loader[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                now_batch_size = inputs.size(0)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # print statistics
                phase_loss += loss.item() * now_batch_size
                running_loss += loss.item()
                if i % 200 == 199 and phase == "train":
                    print(
                        f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.4f}')
                    running_loss = 0.0
            phase_loss = phase_loss / dataset_sizes[phase]
            # y_loss[phase].append(phase_loss)
            res[f"loss_{phase}"].append(phase_loss)
            res["epoch"] = epoch
        print(
            f"Epoch {epoch} loss: {res['loss_train'][-1]:.8f} val: {res['loss_val'][-1]:.8f}")
        draw_curve(epoch, optimizer_name=optimizer.__class__.__name__,
                   loss_name=criterion.__class__.__name__, res=res)

        state = {
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "res": res
        }
        torch.save(state, state_file_name)


train(2)

# %%

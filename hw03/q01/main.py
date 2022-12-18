# %%
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import matplotlib as mpl
import pathlib
import seaborn as sns
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import matplotlib.pyplot as plt
from data import get_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
first_loss = True
first_acc = True


def draw_loss_curve(current_epoch, net_name, optimizer_name, loss_name, res):
    global first_loss
    plt.clf()
    x_epoch = list(range(1, current_epoch+1))
    loss_train = res["loss_train"]
    loss_val = res["loss_val"]
    plt.plot(x_epoch, loss_train, 'bo-', label='train')
    plt.plot(x_epoch, loss_val, 'ro-', label='val')
    if first_loss:
        plt.legend()
        first_loss = False
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs',
                f'train_loss_{net_name}-{optimizer_name}_{loss_name}.jpg'))

def draw_acc_curve(current_epoch, net_name, optimizer_name, loss_name, res):
    global first_acc
    plt.clf()
    x_epoch = list(range(1, current_epoch+1))
    acc_train = res["acc_train"]
    acc_val = res["acc_val"]
    plt.plot(x_epoch, acc_train, 'bo-', label='train')
    plt.plot(x_epoch, acc_val, 'ro-', label='val')
    if first_acc:
        plt.legend()
        first_acc = False
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs',
                f'train_acc_{net_name}-{optimizer_name}_{loss_name}.jpg'))



def total_accuracy():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader['val']:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the {dataset_sizes["val"]} test images: {100 * correct // total} %')

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
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# %%


def get_state_path():
    path = "model"
    net_name = net._get_name()
    criterion_name = criterion.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    os.makedirs(path, exist_ok=True)
    state_file_name = f"{path}/state-{net_name}-optimizer-{optimizer_name}-loss-{criterion_name}.pth"
    return state_file_name


def get_model_wights():
    state_file_name = get_state_path()
    state = torch.load(state_file_name)
    return state["state_dict"]


def get_model_best_wights():
    state_file_name = get_state_path()
    state = torch.load(state_file_name)
    return state["best_net_state"]
# %%


def train(epochs=20):
    path = "model"
    net_name = net._get_name()
    criterion_name = criterion.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    os.makedirs(path, exist_ok=True)
    state_file_name = f"{path}/state-{net_name}-optimizer-{optimizer_name}-loss-{criterion_name}.pth"
    state_res = {}
    best_net_state = None

    print(state_file_name, end=" ")
    if os.path.exists(state_file_name):
        print("exist")
        state = torch.load(state_file_name)
        net.load_state_dict(state["state_dict"])
        best_net_state = state.get("best_net_state", None)
        optimizer.load_state_dict(state["optimizer"])
        state_res = state["res"]

    else:
        print("Not exist")
    res = {
        "loss_train": state_res.get("loss_train", []),
        "loss_val": state_res.get("loss_val", []),
        "acc_val": state_res.get("acc_val", []),
        "acc_train": state_res.get("acc_train", []),
        "epoch": state_res.get("epoch", 0),
    }
    best_val_loss = min(res["loss_val"] + [9999])

    # loop over the dataset multiple times
    try:
        for epoch in range(res["epoch"]+1, epochs+1):
            start_time = time.time()
            running_loss = 0.0
            phase_loss = 0
            for phase in ["train", "val"]:
                total, correct = (0, 0)
                if phase == "train":
                    net.train(True)  # Set model to training mode
                else:
                    net.train(False)  # Set model to evaluate mode

                loop_iter = tqdm(enumerate(data_loader[phase], 0), total=len(
                    data_loader[phase]), leave=False)
                for i, data in loop_iter:
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
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # print statistics
                    phase_loss += loss.item() * now_batch_size
                    running_loss += loss.item()
                    loop_iter.set_postfix({"loss": f"{running_loss / (i+1):04.4f}"})
                phase_loss = phase_loss / dataset_sizes[phase]
                if phase == "val" and phase_loss <= best_val_loss:
                    best_val_loss = phase_loss
                    best_net_state = net.state_dict()
                    print("### BETTER NET STATE ###")
                # y_loss[phase].append(phase_loss)

                res[f"acc_{phase}"].append(round(100*correct/total, 2))
                res[f"loss_{phase}"].append(phase_loss)
                res["epoch"] = epoch
            end_time = time.time() - start_time
            print(
                f"[{end_time:.0f}s] Epoch {epoch} loss : {res['loss_train'][-1]:.8f} acc: {res['acc_train'][-1]} val: {res['loss_val'][-1]:.8f} acc: {res['acc_val'][-1]}%")
            # if epoch % 5 == 0 or epoch in (1, epochs-1):
            #     total_accuracy()
            draw_loss_curve(epoch, net_name=net_name, optimizer_name=optimizer_name,
                       loss_name=criterion_name, res=res)

            draw_acc_curve(epoch, net_name=net_name, optimizer_name=optimizer_name,
                       loss_name=criterion_name, res=res)
            state = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "best_net_state": best_net_state,
                "optimizer": optimizer.state_dict(),
                "res": res
            }
    except KeyboardInterrupt:
        print("Stopping")
        torch.save(state, state_file_name)

    torch.save(state, state_file_name)


# %%
train(300)

# %%
net = Net().to(device)
net.load_state_dict(get_model_best_wights())

y_true = []
y_pred = []
with torch.no_grad():
    for data in data_loader["test"]:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for a, b in zip(labels, predicted):
            # print(a.item(), b.item())
            y_true.append(a.item())
            y_pred.append(b.item())

# %%
# clf_report= classification_report(y_true, y_pred)
# clf_plot = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
# clf_plot.savefig("clf_report.png")


# %%
cm = confusion_matrix(y_true, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot()
plt.savefig("cm.png")
# plt.show()

# %%
def plot_classification_report(y_test, y_pred, title='Classification Report', figsize=(8, 6), dpi=70, save_fig_path=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
    clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    df.sort_values(by=['support'], inplace=True) 
    
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
 
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", fmt='.3g',
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    
    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                    ) 
            
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 360)
         
    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)
    
    return fig, ax

fig, ax = plot_classification_report(y_true, y_pred, 
                    title='Classification Report',
                    figsize=(8, 6), dpi=70,
                    target_names=["Covid19","Non-Covid19"], 
                    save_fig_path = "classificationreport_plot.png")
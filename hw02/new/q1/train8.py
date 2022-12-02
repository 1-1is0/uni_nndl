# %%
import torch.optim as optim
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# %%
transform32 = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.RandomHorizontalFlip(p=0.9),
    #  transforms.RandomVerticalFlip(p=0.9),
    #  transforms.RandomRotation(degrees=180),
     #  transforms.GaussianBlur(kernel_size=501),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])

transform32_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])

transform16 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((16, 16)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])

transform16_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((16, 16)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


transform8 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((8, 8)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform8_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((8, 8)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
num_workers = 12
shuffle = True

trainset32 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform32)
trainset16 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform16)

trainset8 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform8)

testset32 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform32)

testset16 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform16_test)

testset8 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform8)

trainloader32 = torch.utils.data.DataLoader(trainset32, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)

trainloader16 = torch.utils.data.DataLoader(trainset16, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)

trainloader8 = torch.utils.data.DataLoader(trainset8, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers)

testloader32 = torch.utils.data.DataLoader(testset32, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers)

testloader16 = torch.utils.data.DataLoader(testset16, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers)

testloader8 = torch.utils.data.DataLoader(testset8, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_loader = {
    "train": trainloader16,
    "val": testloader16,
}

dataset_sizes = {
    "train": len(trainset16),
    "val": len(testset16)
}
# %%
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
    plt.savefig(os.path.join('./loss_graphs',
                f'train_{optimizer_name}_{loss_name}.jpg'))


def total_accuracy():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader8:
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
        f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def class_accuracy():
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        for data in testloader8:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
# %%


class Net8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = self.pool(x)
        x = self.dropout1(x)
        # print("shape x", x.shape)p
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def initialize_weights(self):
        # https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


net = Net8().to(device)
net.initialize_weights()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
    res["epoch"]

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
                inputs = fn.resize(inputs, (32, 32))
                labels = labels.to(device)
                now_batch_size = labels.size()[0]
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


# %%
train(100)
print('Finished Training')

# %%
PATH = './cifar_net_train_8.pth'
torch.save(net.state_dict(), PATH)

# %%
class_accuracy()
total_accuracy()
# %%
correct = 0
total = 0
PATH = './cifar_net_train_8.pth'
net = Net8().to(device)
net.load_state_dict(torch.load(PATH))
# since we're not training, we don't need to calculate the gradients for our outputs
y_true = []
y_pred = []
with torch.no_grad():
    for data in testloader8:
        images, labels = data
        images = fn.resize(images, [32, 32])
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for a, b in zip(labels, predicted):
            # print(a.item(), b.item())
            y_true.append(a.item())
            y_pred.append(b.item())
        # break
        # the class with the highest energy is what we choose as prediction
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()


# %%
print(classification_report(y_true, y_pred))

# %%
f1_score(y_true, y_pred, average="macro")
# %%
precision_score(y_true, y_pred, average="macro")
# %%
accuracy_score(y_true, y_pred)
# %%

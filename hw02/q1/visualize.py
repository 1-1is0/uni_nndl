# %%
import torch.optim as optim
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt
import numpy as np

# %%
transform32 = transforms.Compose(
     #  transforms.CenterCrop(10),
    [transforms.ToTensor(),
    #  transforms.RandomHorizontalFlip(p=0.9),
    #  transforms.RandomRotation(degrees=180),
     #  transforms.Grayscale(num_output_channels=1),
     #  transforms.GaussianBlur(kernel_size=501),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])
#  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform16 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((16, 16)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform8 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((8, 8)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
num_workers = 12
shuffle = False

trainset32 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform32)
trainset16 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform16)

trainset8 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform8)

testset32 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform32)

testset16 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform16)

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

# %%


def iter_sample_fast(iterable1, iterable2, iterable3, samplesize):
    results = [[0, 0, 0, 0] for i in range(samplesize)]
    iterator = iter(zip(iterable1, iterable2, iterable3))
    for j, ((d32, l32), (d16, l16), (d8, l8)) in enumerate(iterator, samplesize):
        r = random.randint(0, j)
        if r < samplesize:
            i = 0
            d0, d1, d2 = d32[i], d16[0], d8[0]
            # at a decreasing rate, replace random items
            results[r] = (d0, d1, d2, l32[0])

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")
    return results

# %%


def show(data):
    for plot_num, sample in enumerate(data, 1):
        fig, axes = plt.subplots(1, 3)
        images, label = sample[:-1], sample[-1]
        fig.tight_layout()
        fig.suptitle(f"label: {classes[label]}")
        for col, img in enumerate(images):
            img = img * 0.5 + 0.5
            img = img.numpy()
            img = np.einsum("ijk->jki", img)
            img_size = img.shape[1]
            axes[col].set_xlabel(f"{img_size}x{img_size}")
            axes[col].set_xticks([])
            axes[col].set_yticks([])
            axes[col].imshow(img)

            plt.savefig(f"figs/fig{plot_num}.png")
        #     plt.show()

# %%
vis_data = iter_sample_fast(trainloader32, trainloader16, trainloader8, 10)
show(vis_data)


# %%


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.batch_norm = nn.BatchNorm2d(100)

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.flatten = nn.Flatten()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.dense1 = nn.Linear(in_features=64 * 3 * 3, out_features=512)
        self.dropout3 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        # print(x.size())
        x = F.relu(self.dense1(x))
        x = self.dropout3(x)
        x = F.softmax(self.dense2(x))
        return x


# %%
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# %%

# def accuracy(outputs, labels):
#     n_correct, n_wrong = 0, 0

#     abs_delta = np.abs(outputs_scaled - labels_scaled)
#     n_correct = (abs_delta < 175000).sum().item()
#     n_wrong = outputs.size()[0] - n_correct
#     return n_correct, n_wrong


def draw_curve(current_epoch, optimizer_name, loss_name, x_epoch, y_loss):
    x_epoch.append(current_epoch)
    plt.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    plt.plot(x_epoch, y_loss['val'], 'ro-', label='val')

    if current_epoch == 0:
        plt.legend()
    plt.savefig(os.path.join('./loss_graphs',
                f'train_{optimizer_name}_{loss_name}.jpg'))


# %%
data_loader = {
    "train": trainloader32,
    "val": testloader32,
}

dataset_sizes = {
    "train": len(trainset32),
    "val": len(testset32)
}


# %%
def train_model(model, optimizer, criterion, epochs=200):
    x_epoch = []
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []

    for epoch in range(epochs+1):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            for i, (data, labels) in enumerate(data_loader[phase], 0):
                data = data.to(device)
                labels = labels.to(device)
                now_batch_size = data.size()[0]
                # data = data.to(device)
                # labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                # print("output shape", outputs.shape)
                # print("lable shape", labels.shape)
                loss = criterion(outputs, labels)
                # return
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * now_batch_size
                # del loss
                # n_correct, n_wrong = accuracy(outputs, labels)
                # running_corrects += n_correct

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_acc = 0

            print(
                f'epoch: {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            y_loss[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val':
                draw_curve(epoch, optimizer_name=optimizer.__class__.__name__,
                           loss_name=criterion.__class__.__name__, x_epoch=x_epoch, y_loss=y_loss)


# %%
net = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
# %%

train_model(net, optimizer, criterion, epochs=50)

# %%

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%
dataiter = iter(testloader32)
images, labels = next(dataiter)
images = images.to(device)
labels = labels.to(device)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("show.png")
# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))


# %%
PATH = './cifar_net.pth'
net = Model().to(device)
net.load_state_dict(torch.load(PATH))
# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                               for j in range(len(labels))))


# %%
def total_accuracy():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader32:
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


# %%
def class_accuracy():
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        for data in testloader32:
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
total_accuracy()
class_accuracy()
# %%

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(576, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        # print("shape x", x.shape)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader32, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
# %%
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader32:
        images, labels = data
        
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# %%

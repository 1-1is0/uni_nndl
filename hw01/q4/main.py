# %%
import os
import pandas as pd
import data_describe as dd
import matplotlib.pyplot as plt
import numpy as np


# %%
houses = pd.read_csv("./data/houses.csv")


# %%
houses.info()


# %% [markdown]
# # B. Count Nan values
# 

# %%
houses.isna().sum()


# %% [markdown]
# # C. Correltaion Matrix and Price Correlation
# 

# %%
correlation_matrix = houses.corr(numeric_only=True)


# %%
dd.correlation_matrix(houses, cluster=True, viz_backend="plotly")

# %% [markdown]
# ### `sqft_living` has the highest correlation with price
# 

# %%
correlation_matrix.price.nlargest(n=2)


# %%
correlation_matrix.price


# %% [markdown]
# # D. Price dist
# 

# %%
import plotly.express as px
fig = px.histogram(houses.price, x="price", title="Price Histogram")
fig.show()


# %%
fig = px.scatter(houses, x="price", y="sqft_living",
                 trendline="ols", opacity=0.5, trendline_color_override="green")
fig.show()


# %% [markdown]
# # E. Date to year and month
# 

# %%
houses["year"] = houses.date.str[0:4].astype(int)
houses["month"] = houses.date.str[4:6].astype(int)
houses = houses.drop(columns=["date"])


# %% [markdown]
# # F. Test Train split
# 

# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(houses, test_size=0.2)


# %% [markdown]
# # G. MinMaxScaler
# 

# %%
from sklearn import preprocessing

scalers = {}

# dont_scale = ["id", "lat", "long", "year", "month", "zipcode",
#               "floors", "waterfront", "view", "condition", "grade"]
dont_scale = ["id"]


def normalize(train, test):
    # dont' scale the id
    for feature in train.columns.drop(dont_scale):
        min_max_scaler = preprocessing.MinMaxScaler()
        train[feature] = min_max_scaler.fit_transform(train[[feature]])
        test[feature] = min_max_scaler.transform(test[[feature]])
        scalers[feature] = min_max_scaler
    return train, test


# %%
train, test = normalize(train, test)


# %%
train


# %% [markdown]
# # H. MLP 2 layers
# 

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


# %%
class HousesDataset(Dataset):
    def __init__(self, df, cols=[]):
        self.df = df
        self.x = self.df[cols].values
        self.y = self.df[["price"]].values
        self.x = torch.tensor(self.x, dtype=torch.float).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float).to(device)
        self.ids = self.df[["id"]].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_id(self, idx):
        return self.ids[idx]


# %%
houses.corr().price


# %%
houses.columns.values


# %%
cols = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    # "floors", "waterfront"
    'view', 'grade', 'sqft_above',
    'sqft_basement',  'lat',
    'sqft_living15'
]
feature_length = len(cols)
print(len(cols))


# %%
train_dataset = HousesDataset(train, cols=cols)
test_dataset = HousesDataset(test, cols=cols)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=True, num_workers=12)


# %%
for i, (data, labels) in enumerate(train_loader):
    print(data.shape, labels.shape)
    print(data, labels)
    break


# %%
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        a, b, c = (20, 18, 15)
        self.fc1 = nn.Linear(10, a)
        self.fc2 = nn.Linear(a, b)
        self.fc3 = nn.Linear(b, c)
        # self.fc4 = nn.Linear(c, d)
        # self.fc5 = nn.Linear(d, e)

        self.out = nn.Linear(c, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        # x = self.fc4(x).clamp(min=0)
        # x = F.relu(x)

        # x = self.fc5(x).clamp(min=0)
        # x = F.relu(x)

        x = self.out(x)
        x = torch.sigmoid(x)
        return x

    def initialize_weights(self):
        # TODO random initiate
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


# %%
data_loader = {
    "train": train_loader,
    "val": test_loader,
}

dataset_sizes = {
    "train": len(train_dataset),
    "val": len(test_dataset)
}


# %%
def accuracy(outputs, labels):
    n_correct, n_wrong = 0, 0

    price_scaler = scalers["price"]
    outputs_scaled = price_scaler.inverse_transform(outputs.detach().numpy())
    labels_scaled = price_scaler.inverse_transform(labels.detach().numpy())

    abs_delta = np.abs(outputs_scaled - labels_scaled)
    n_correct = (abs_delta < 175000).sum().item()
    n_wrong = outputs.size()[0] - n_correct
    return n_correct, n_wrong


# %%
def draw_curve(current_epoch, optimizer_name, loss_name, x_epoch, y_loss):
    x_epoch.append(current_epoch)
    plt.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    plt.plot(x_epoch, y_loss['val'], 'ro-', label='val')

    if current_epoch == 0:
        plt.legend()
    plt.savefig(os.path.join('./lossGraphs',
                f'train_{optimizer_name}_{loss_name}.jpg'))


# %%
def train_model(model, optimizer, criterion, epochs=200):
    fig = plt.figure()
    plt.clf()
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
                now_batch_size = data.size()[0]
                # data = data.to(device)
                # labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * now_batch_size
                # del loss
                n_correct, n_wrong = accuracy(outputs, labels)
                running_corrects += n_correct

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if epoch % 10 == 0:
                print(
                    f'epoch: {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            y_loss[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val':
                draw_curve(epoch, optimizer_name=optimizer.__class__.__name__,
                           loss_name=criterion.__class__.__name__, x_epoch=x_epoch, y_loss=y_loss)


# %%
def random_test(model):
    sample_generator = torch.utils.data.RandomSampler(
        data_source=test_dataset, num_samples=5)
    price_scaler = scalers["price"]
    for index, num in enumerate(sample_generator, 1):
        data, label = test_dataset[num]
        # i = test_dataset.get_id(num)
        # print("ID", i)
        model.eval()
        output = model(data)
        output_scaled = price_scaler.inverse_transform(
            np.array(output.item()).reshape(-1, 1))[0][0]
        label_scaled = price_scaler.inverse_transform(
            np.array(label.item()).reshape(-1, 1))[0][0]
        # print(output_scaled, label_scaled)
        print(f"{index}: {output_scaled:,.0f}, {label_scaled:,.0f} ", end="")
        print(f"error: {abs(label_scaled-output_scaled):,.0f}")


# %% [markdown]
# ## 1. Train SGD, MSELoss

# %%
model1 = MLP().to(device)
optimizer = torch.optim.SGD(model1.parameters(), lr=0.05)
criterion = nn.MSELoss()
train_model(model1, optimizer, criterion)


# %%
random_test(model1)


# %% [markdown]
# ## 2. Train SGD, L1Loss

# %%
model2 = MLP().to(device)
optimizer = torch.optim.SGD(model2.parameters(), lr=0.05)
criterion = nn.L1Loss()
train_model(model2, optimizer, criterion)


# %%
random_test(model2)

# %% [markdown]
# ## 3. Train Adadelta, MSELoss

# %%
model3 = MLP().to(device)
optimizer = torch.optim.Adadelta(model3.parameters(), lr=0.05)
criterion = nn.MSELoss()
train_model(model3, optimizer, criterion)

# %%
random_test(model3)

# %% [markdown]
# ## 4. Train Adadelta L1Loss

# %%
model4 = MLP().to(device)
optimizer = torch.optim.Adadelta(model4.parameters(), lr=0.05)
criterion = nn.L1Loss()
train_model(model4, optimizer, criterion)

# %%
random_test(model4)

# %%
total = 0
b = 0
max_diff = 0
min_diff = houses.price.max()
for i, (data, label) in enumerate(test_loader):
    model.eval()
    output = model(data)
    diff = abs(output - label)
    total += diff.sum().item()
    m = diff.max().item()
    max_diff = m if m > max_diff else max_diff

    m = diff.min().item()
    min_diff = m if m < min_diff else min_diff
    b += data.size()[0]

print("mean", total/b)
print('max diff', max_diff, "min diff", min_diff)




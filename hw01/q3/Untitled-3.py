# %%
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data

import torch
from sklearn import preprocessing
import pandas as pd
import numpy as np

# %%
movies = pd.read_csv("./data/movies.csv")
ratings = pd.read_csv("./data/ratings.csv")

# %% [markdown]
# # A. showing some data

# %% [markdown]
# ## 5 first movies

# %%
movies.head(5)

# %% [markdown]
# # 5 last movies

# %%
movies.tail(5)

# %% [markdown]
# ## 5 first ratings

# %%
ratings.head(5)

# %% [markdown]
# ## 5 last ratings

# %%
ratings.tail(5)

# %% [markdown]
# ## movies dataset shape

# %%
movies.shape

# %% [markdown]
# ## ratings dataset shape

# %%
ratings.shape

# %% [markdown]
# ### create a column in movie dataset

# %%

movies["list_index"] = movies.index

# %% [markdown]
# # B. merge the two datasets

# %%
df = pd.merge(movies, ratings, on="movieId")

# %%
df.head()

# %%
df.shape


# %% [markdown]
# # C. Delete the extra rows

# %% [markdown]
# title, genres and timestamp seems to have no use looking forward

# %%
df = df.drop(columns=["title", "genres", "timestamp", "list_index"])


# %%
df.head()

# %% [markdown]
# # D. Group by `userId`

# %%
groups = df.groupby("userId")
groups.agg({
    "movieId": "count",
    "rating": "mean",

})


# %% [markdown]
# # E. Normalize the ratings


# %%
df[["rating"]]

# %%
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(df[["rating"]])
train_x

# %%
# df["rating"] = train_x

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device= "cpu"
print(device)

# %%
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# %%


def convert_to_output(row, user_rated_movie, user_id):
    movie_id = user_rated_movie.movieId.values
    if row in movie_id:
        return df.loc[(df.movieId == row) & (df.userId == user_id)].rating.values[0]
        # return
    else:
        return 0


def get_train_data():
    all_data = []
    for name, group in groups:
        print(name)
        rated_zero = pd.DataFrame(df.movieId.unique(), columns=["movieId"])
        # print(group)
        rated_zero["rating"] = rated_zero.movieId.apply(
            convert_to_output, user_rated_movie=group, user_id=name)
        all_data.append(rated_zero)
    return all_data


# %%
all_data = get_train_data()

# %% [markdown]
# # F. Build tne network

# %%


class RBM():
    def __init__(self, n_visible_layer, n_hidden_layer=20, lr=0.1):
        self.lr = lr
        self.W = nn.Parameter(torch.randn(
            (n_hidden_layer, n_visible_layer))*1e-2, requires_grad=False)
        self.a_bias = nn.Parameter(torch.randn(
            1, n_hidden_layer), requires_grad=False)
        self.r_bias = nn.Parameter(torch.randn(
            1, n_visible_layer), requires_grad=False)

    def forward(self, x0):
        print(x0.shape, self.W.shape, self.a_bias)
        _a0 = F.linear(x0, self.W, self.a_bias)
        _a0 = torch.sigmoid(_a0)
        a0 = torch.relu(torch.sign(_a0 - torch.randn(_a0.shape)))
        return x0, _a0, a0

    def back(self, a0):
        _x1 = torch.sigmoid(F.linear(a0, self.W.t(), self.r_bias))
        x1 = torch.relu(torch.sign(_x1=torch.randn(_x1.shape)))
        a1 = torch.sigmoid(F.linear(x1, self.W, self.a_bias))
        # x = torch.sigmoid(a)
        return _x1, x1, a1

    def train(self, x0, x1, a0, a1):
        pos = torch.mm(x0.t(), a0)
        neg = torch.mm(x1.t(), a1)
        cd = (pos - neg) / x0.shape[0]
        self.W += self.lr * cd
        self.r_bias += self.lr * torch.mean(x0 - x1, 0)
        # self.r_bias += self.lr * (x - xp)
        self.a_bias += self.lr * torch.mean(a0 - a1, 0)

    def loss(self, train, xp):
        indexes = data[data.rating != 0].index.values
        user_ratings = data[data.rating != 0].rating.values
        user_ratings = torch.tensor(user_ratings, dtype=torch.float)
        pred_ratings = rating_p[0, indexes]

        l = user_ratings - pred_ratings
        l = torch.pow(l, 2)
        l = torch.sum(l)
        return l


# %% [markdown]
# # G. Train for 20 epochs

# %%
n_visible_layer = df.movieId.unique().shape[0]
n_hidden_layer = 20
# predict_movie = RBM()
rbm = RBM(n_visible_layer=n_visible_layer, n_hidden_layer=n_hidden_layer, lr=1)
# train_op = optim.SGD(rbm.parameters(), 0.1)

total_loss = []
# TODO sort by movieId
for epoch in range(20):
    loss = []
    for data in all_data:
        # print(data)
        rating = data.rating.values
        rating = torch.tensor(rating, dtype=torch.float)
        # print(len(rating))
        x0 = Variable(rating.view(-1, len(rating)))

        x0, _a0, a0 = rbm.forward(x0)
        _x1, x1, a1  = rbm.back(a0)
        rbm.train(x0, x1, a0, a1)
        # loss.append(rbm.loss(data, rating_p))
    # total_loss.append(sum(loss))
# 
    # print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))

# %%
plt.plot(total_loss)


# %%
sample_user = all_data[17]
rating = sample_user.rating.values
rating = torch.tensor(rating, dtype=torch.float)
rating = Variable(rating.view(-1, len(rating)))
a = rbm.forward(rating)
xp = rbm.back(a)
# %%

xp_numpy = xp.detach().numpy()
print(np.transpose(xp_numpy).shape)
sample_user["recom"] = np.transpose(xp_numpy).tolist()
sample_user["recom"] = sample_user['recom'].apply(lambda x: x[0])
# %%

sample_user.shape

# %%

non_rated = sample_user[sample_user.rating == 0]  # type: pd.DataFrame
non_rated.nlargest(15, 'recom')

# %%

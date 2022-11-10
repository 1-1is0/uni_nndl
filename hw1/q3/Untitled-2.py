# %%
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
df["rating"] = train_x

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
    def __init__(self, n_visible_layer, n_hidden_layer=20, lr=0.05):
        self.lr = lr
        self.W = nn.Parameter(torch.randn(
            (n_hidden_layer, n_visible_layer))*1e-2, requires_grad=False)
        self.a_bias = nn.Parameter(torch.randn(
            1, n_hidden_layer), requires_grad=False)
        self.r_bias = nn.Parameter(torch.randn(
            1, n_visible_layer), requires_grad=False)

    def forward(self, x):
        x = F.linear(x, self.W, self.a_bias)
        a = torch.sigmoid(x)
        return a

    def back(self, a):
        a = F.linear(a, self.W.t(), self.r_bias)
        x = torch.sigmoid(a)
        return x

    def sample_visible(a):
        return torch.bernoulli(a)

    def sample_hidden(x):
        return torch.bernoulli(x)

    def train(self, x, xp, a, ap):
        self.W += self.lr * (torch.mm(x.t(), a) - torch.mm(xp.t(), ap)).t()
        self.r_bias += self.lr * (x - xp)
        self.a_bias += self.lr * (a - ap)
        # self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        # add zero to keep b as a tensor of 2 dimension
        # self.b += torch.sum((v0 - vk), 0)
        # self.a += torch.sum((ph0 - phk), 0)


# %%
# j = 1/2 * torch.sum(torch.pow(torch.abs(x), 2))
# %% [markdown]
# # G. Train for 20 epochs

# %%
n_visible_layer = df.movieId.unique().shape[0]
n_hidden_layer = 20
# predict_movie = RBM()
rbm = RBM(n_visible_layer=n_visible_layer, n_hidden_layer=n_hidden_layer)
# train_op = optim.SGD(rbm.parameters(), 0.1)

# TODO sort by movieId
for epoch in range(20):
    loss_ = []
    for data in all_data:
        # print(data)
        rating = data.rating.values
        rating = torch.tensor(rating, dtype=torch.float)
        # print(len(rating))
        rating = Variable(rating.view(-1, len(rating)))
        a = rbm.forward(rating)
        rating_p = rbm.back(a)
        ap = rbm.forward(rating_p)
        rbm.train(rating, rating_p, a, ap)

    # print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))


# %%
sample_user = all[75]
rating = sample_user.rating.values
rating = torch.tensor(rating, dtype=torch.float)
rating = Variable(rating.view(-1, len(rating)))
a = rbm.forward(rating)
xp = rbm.back(a)
# %%

xp_numpy = xp.detach().numpy()
print(np.transpose(xp_numpy).shape)
sample_user['recom'] = np.transpose(xp_numpy).tolist()
# %%
sample_user

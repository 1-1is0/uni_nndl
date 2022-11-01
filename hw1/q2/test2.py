import pandas as pd
import numpy as np


data = pd.read_csv("Attachments\Q2_Dataset\MadaLine.csv")

X1 = data.iloc[:, 0]
X2 = data.iloc[:, 1]
target = data.iloc[:, 2]
target=np.where(target==0,-1,1).reshape(-1,1)
class1 = data.loc[target == -1, :]
class2 = data.loc[target == 1, :]
X1.shape

data.iloc[:, 2]


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.autolayout"] = True


plt.plot(class1.iloc[:, 0], class1.iloc[:, 1], "y*")
plt.plot(class2.iloc[:, 0], class2.iloc[:, 1], 'm*')
plt.legend(["class1", "class2"], loc="upper left")
plt.show()


r = 0.05

w = np.random.uniform(-r, r, [2, 2])
print(w)
print(w[0].shape)
b = np.random.uniform(-r, r, [2, 1])

vy = np.random.uniform(-r, r, [1, 2])
by = np.random.uniform(-r, r, [1, 1])


lr = 0.05


w1 = np.hstack((w[0], b[0]))

w2 = np.hstack((w[1], b[1]))

weight= np.vstack((w1,w2)).T
bias=np.ones((199,1))
xx=np.vstack((X1,X2)).T

print(xx.shape)
weight=weight
print(weight.shape)


class mad():
    def __init__(self, learning_rate, iter, neuron):
        self.lr = learning_rate
        self.iteration = iter
        self.number_of_neuron = neuron

    def update_weight(self, i, k, x):
        self.weight[2, i] = self.weight[2, i] + self.lr*(1-self.z_in[k, i])
        print(weight.shape)
        self.weight[:2, i] = self.weight[0:2, i] + self.lr * \
            (1-self.z_in[k, i])*x[k, 0:2]
        return weight

    def fit(self, x, target):
        x = self.add_bias(x)

        for epoch in range(self.iteration):

            self.make_z(x)
            self.predict()
            for k in range(xx.shape[0]):

                if target[k] == 1 and self.y[k] != 1:
                        print(k)
                        weight = self.update_weight(self.indexes[k], k, x)
                if target[k] == -1 and self.y[k] != -1:
                    print("&************zin", self.z_in.shape)
                    for i in range(self.z_in.shape[1]):
                        if (self.z_in[k, i] > 0):    
                            weight = self.update_weight(i, k, x)
                    
    def make_z(self, x):
        self.z_in = np.zeros((x.shape[0], self.number_of_neuron))
        self._initial_weights(x)
        print("weights", self.weight)
        print("x", x.shape)
        self.z_in = x.dot(self.weight)
        print("self.zin", self.z_in.shape)
        self.z = np.where(self.z_in >= 0.0, 1, -1)

        self.indexes = np.argmin(np.abs(self.z_in), axis=1)

    def add_bias(self, x):
        bias_x = np.ones((x.shape[0], 1))
        x = np.hstack((x, bias_x))
        return x

    def cost(self, y, target):
        e = y-target
        cost = (e**2).sum()/2
        return cost

    def _initial_weights(self, x):
        """ Initialise weights - normal distribution sample with standard dev 0.01 """

        random_gen = np.random.RandomState(1)
        vy = random_gen.uniform(-0.05, 0.05, [1, self.number_of_neuron])
        by = random_gen.uniform(-0.05, 0.05, [1, 1])
        self.wy = np.hstack((vy, by)).reshape(-1, 1)
        self.weight = random_gen.normal(
            loc=0.0, scale=0.01, size=(x.shape[1], self.number_of_neuron))
        print("weight", self.weight.shape)
        return self

    def predict(self):

        self.z = np.hstack((self.z, np.ones((self.z.shape[0], 1))))
        y_in = self.z.dot(self.wy)
        self.y = np.where(y_in > 0.0, 1, -1)


a=mad(0.05,200,6)
a.fit(xx,target)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.autolayout"] = True


plt.plot(class1.iloc[:, 0], class1.iloc[:, 1], "y*")
plt.plot(class2.iloc[:, 0], class2.iloc[:, 1], 'm*')
plt.legend(["data1", "data2"], loc="upper left")
x = np.linspace(-5, 5, 100)
plt.xlim(-3, 4)
plt.ylim(-2, 2)
for i in range(a.weight.shape[1]):
    
    y = - (a.weight[0, i]/a.weight[1, i]) * x - a.weight[2, i]/a.weight[1, i]
    plt.plot(x, y, "--r")
# plt.show()
plt.savefig("plot.png")

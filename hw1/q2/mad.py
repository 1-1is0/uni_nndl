# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %%
data = pd.read_csv("./MadaLine.csv")


# %%
X1 = data.iloc[:, 0]
X2 = data.iloc[:, 1]
target = data.iloc[:, 2]
target = np.where(target == 0, -1, 1).reshape(-1, 1)
class1 = data.loc[target == -1, :]
class2 = data.loc[target == 1, :]
X1.shape

# %%
data.iloc[:, 2]


# %%

plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.autolayout"] = True


plt.plot(class1.iloc[:, 0], class1.iloc[:, 1], "y*")
plt.plot(class2.iloc[:, 0], class2.iloc[:, 1], 'm*')
plt.legend(["class1", "class2"], loc="upper left")
plt.show()


# %%
xx = np.vstack((X1, X2)).T
xx.shape

# %%
np.random.seed(42)

# %%


class mad():
    def __init__(self, learning_rate, n_iter, neuron):
        self.lr = learning_rate
        self.iteration = n_iter
        self.number_of_neuron = neuron

    def update_weight(self, i, k):
        self.weight[2, i] = self.weight[2, i] + self.lr*(1-self.z_in[k, i])
        self.weight[:2, i] = self.weight[0:2, i] + self.lr * \
            (1-self.z_in[k, i])*self.x[k, 0:2]
        return self.weight

    def fit(self, x, target):
        self.x = self.add_bias(x)
        self._initial_weights()

        self.cost = []
        for epoch in range(self.iteration):

            self.make_z()
            self.predict()
            for k in range(self.x.shape[0]):

                if target[k] == 1 and self.y[k] != 1:
                    self.weight = self.update_weight(self.indexes[k], k)
                if target[k] == -1 and self.y[k] != -1:
                    for i in range(self.z_in.shape[1]):
                        if (self.z_in[k, i] > 0):
                            self.weight = self.update_weight(i, k)

    def make_z(self):
        self.z_in = np.zeros((self.x.shape[0], self.number_of_neuron))
        self.z_in = self.x.dot(self.weight)
        # ues rule for test
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

    def accuracy(self, target, y):
        n_zeros = np.count_nonzero((y-target) == 0)
        acc = n_zeros/len(y)
        return acc

    def _initial_weights(self):
        """ Initialise weights - normal distribution sample with standard dev 0.01 """
        loc, scale = 0, 0.01
        random_gen = np.random.RandomState(10)
        # vy = random_gen.normal(loc=loc, scale=scale,
        #                        size=(1, self.number_of_neuron))

        # by = random_gen.normal(loc=loc, scale=scale, size=(1, 1))

        # self.wy = np.hstack((vy, by)).reshape(-1, 1)

        random_gen = np.random.RandomState(10)
        # number of neuron plus the bias
        self.wy = random_gen.normal(loc=loc, scale=scale,
                                    size=(self.number_of_neuron + 1, 1))

        random_gen = np.random.RandomState(20)
        self.weight = random_gen.normal(
            loc=loc, scale=scale, size=(self.x.shape[1], self.number_of_neuron))
        print(self.weight.shape)

        return self

    def predict(self):

        self.z = np.hstack((self.z, np.ones((self.z.shape[0], 1))))
        y_in = self.z.dot(self.wy)
        self.y = np.where(y_in > 0.0, 1, -1)


# %%
def draw(w, b):
    lines = []
    plt.clf()
    for (w2, w1), b in zip(w, b):
        x = np.linspace(-5, 5, 100)
        y = - (w1/w2) * x - b/w2
        # print(w1, w2, b)
        lines.append((x, y))
        # Line2.append(-domain[i]*(trained_w1[1][0]/trained_w1[1]
        #              [1])-trained_b1[1]/trained_w1[1][1])
        # Line3.append(-domain[i]*(trained_w1[2][0]/trained_w1[2]
        #              [1])-trained_b1[2]/trained_w1[2][1])

    plt.plot(class1.iloc[:, 0], class1.iloc[:, 1], "y*")
    plt.plot(class2.iloc[:, 0], class2.iloc[:, 1], 'm*')
    plt.legend(["data1", "data2"], loc="upper left")
    for l in lines:
        plt.plot(x, y, "--r")
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
# %%


class mad2():
    def __init__(self, number_of_neuron, x):
        self.number_of_neuron = number_of_neuron
        self.x = x

    def init_weights(self):
        random_gen = np.random.RandomState(10)
        L1_weights = np.random.normal(0, 0.01, (self.number_of_neuron, 2))
        random_gen2 = np.random.RandomState(20)
        L1_bias = np.random.normal(0, 0.01, self.number_of_neuron)
        # return L1_weights,L1_bias

        loc, scale = 0, 0.01
        self.weight = random_gen.normal(
            loc=loc, scale=scale, size=(self.x.shape[1]+1, self.number_of_neuron))
        # print("single weight shape", self.weight.shape)
        # print("shape weights", self.weight[:-1,
        #       :].shape, self.weight[-1, :].shape)
        return self.weight[:-1, :], self.weight[-1, :]

    def accuracy(self, y_pred, y_test):
        n_zeros = np.count_nonzero((y_pred-y_test) == 0)
        acc = n_zeros/len(y_pred)
        return acc

    def activation_function(self, z):
        res = np.where(z >= 0, 1, -1)
        return res
        res = []
        for i in range(len(z)):
            if z[i] >= 0:
                res.append(1)
            else:
                res.append(-1)
        return res

    def net_function(self, X, w, bias):
        print("X.shpae", X.shape, "wshape", w.shape)
        res = X.dot(w)+bias
        # res = np.dot(w, X)+bias
        print("shape res", res.shape)
        return res

    def training(self, X_train, w1, w2, bias1, bias2, y_train, learning_rate, iteration):
        costs = []
        for epoch in range(iteration):
            cost = 0
            z = self.net_function(X_train, w1, bias1)
            z_in = self.activation_function(z)
            print("z_in shape", z_in.shape)
            y_in = z_in.dot(w2) + bias2
            print(y_in.shape)
            y = self.activation_function(y_in)
            print(y)
            for i in range(len(y)):
                errors = y_train[i] - y[i]
                if errors != 0 and y_train[i] == 1:
                    print("z.shape", z.shape)
                    idx = np.argmin(abs(z), axis=1)
                    p1 = learning_rate * X_train[i][0]*(1-z[idx])
                    p2 = learning_rate * X_train[i][1]*(1-z[idx])
                    p = np.zeros((len(w1), self.number_of_neuron))
                    p[0, idx] = p1
                    p[1, idx] = p2
                    # print("w111")
                    w1 = w1 + p
                    bias1[idx] = bias1[idx] + learning_rate*(1-z[idx])
                if errors != 0 and y_train[i] == -1:
                    for j in range(len(z)):
                        # for k in range(4):
                        #     print("zi  shape", z.shape, z[j].shape)
                        p1 = learning_rate * X_train[i][0]*(-1-z[j])
                        p2 = learning_rate * X_train[i][0]*(-1-z[j])
                        p = np.array((p1, p2))

                        w1 = w1 + p
                        print("biase shpae", bias1.shape)
                        bias1 = bias1 + learning_rate*(-1-z[j])
                        # print("p1", p1.shape, "p2", p2.shape)
                        # print(p1, p2)

                    # return
                    # for j in range(len(z[i])):
                    # if z[i, j] > 0:

                    #     print("X_train", X_train[i][0].shape)
                    #     print("zi shape", z[i].shape)

                    #     p1 = learning_rate * X_train[i][0]*(-1-z[i, j])

                    #     p2 = learning_rate * X_train[i][1]*(-1-z[i, j])
                    #     p = np.zeros((len(w1), self.number_of_neuron))
                    #     p[0, j] = p1
                    #     p[1, j] = p2
                    #     # print(p)
                    #     # print("P shape", p.shape, "w1shape", w1.shape)
                    #     w1 = w1 + p
                    #     bias1[i] = bias1[i] + learning_rate*(-1-z[i, j])
                cost = cost+(errors**2) / 2.0
            costs.append(cost)

        return w1, bias1, costs

        #     for i in range(len(X_train)):
        #         # print("Xshpae before", X_train.shape)
        #         z = self.net_function(X_train[i], w1, bias1)
        #         y_in = np.dot(w2, self.activation_function(z))+bias2
        #         if y_in >= 0:
        #             out_pred = 1
        #         else:
        #             out_pred = -1
        #         errors = y_train[i] - out_pred
        #         if errors != 0 and y_train[i] == 1:
        #             idx = np.argmin(abs(z))
        #             p1 = learning_rate * X_train[i][0]*(1-z[idx])
        #             p2 = learning_rate * X_train[i][1]*(1-z[idx])
        #             p = np.zeros((len(w1), self.number_of_neuron))
        #             p[0, idx] = p1
        #             p[1, idx] = p2
        #             # print("w111")
        #             w1 = w1 + p
        #             bias1[idx] = bias1[idx] + learning_rate*(1-z[idx])
        #         if errors != 0 and y_train[i] == -1:
        #             for i in range(len(z)):
        #                 if z[i] > 0:

        #                     # print(i, "z shape", z.shape,
        #                     #       "X_train shape", X_train.shape)
        #                     p1 = learning_rate * X_train[i][0]*(-1-z[i])
        #                     p2 = learning_rate * X_train[i][1]*(-1-z[i])
        #                     p = np.zeros((len(w1), self.number_of_neuron))
        #                     p[0, i] = p1
        #                     p[1, i] = p2
        #                     # print(p)
        #                     # print("P shape", p.shape, "w1shape", w1.shape)
        #                     w1 = w1 + p
        #                     bias1[i] = bias1[i] + learning_rate*(-1-z[i])
        #         cost = cost+(errors**2) / 2.0
        #     costs.append(cost)
        # return w1, bias1, costs


data2 = data.to_numpy()
# set one
x1 = data2[0:99, 0]
y1 = data2[0:99, 1]

# set two
x2 = data2[99:199, 0]
y2 = data2[99:199, 1]

X_train = data2[0:round(1*len(data)), 0:2]
y_train = data2[0:round(1*len(data)), 2]
for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = -1


neuron_Num = 4
mm = mad2(neuron_Num, x=X_train)
np.random.seed(19)
w1, bias1 = mm.init_weights()
w2 = np.zeros((neuron_Num, 1))
w2[0:neuron_Num] = 1
bias2 = neuron_Num-1
learning_rate = 0.01
iteration = 50

trained_w1, trained_b1, costs = mm.training(
    X_train, w1, w2, bias1, bias2, y_train, learning_rate, iteration)
draw(trained_w1, trained_b1)

# %%
test(4)


# %%
test(5)

# %%
test(6)

# %%
test(8)

# %%


def test(i):
    neuron_Num = i
    mm = mad2(neuron_Num, x=X_train)
    np.random.seed(19)
    w1, bias1 = mm.init_weights()
    w2 = np.zeros((neuron_Num))
    w2[0:neuron_Num] = 1
    bias2 = neuron_Num-1
    learning_rate = 0.01
    iteration = 200

    trained_w1, trained_b1, costs = mm.training(
        X_train, w1, w2, bias1, bias2, y_train, learning_rate, iteration)
    draw(trained_w1, trained_b1)


# %%

plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.autolayout"] = True


plt.plot(class1.iloc[:, 0], class1.iloc[:, 1], "y*")
plt.plot(class2.iloc[:, 0], class2.iloc[:, 1], 'm*')
plt.legend(["data1", "data2"], loc="upper left")
x = np.linspace(-5, 5, 100)
# plt.xlim(-3, 4)
# plt.ylim(-2, 2)

for i in range(m.weight.shape[1]):
    w1_i, w2_i = 0, 1
    y = - (m.weight[w1_i, i]/m.weight[w2_i, i]) * \
        x - m.weight[2, i]/m.weight[w2_i, i]
    plt.plot(x, y, "--r")
plt.show()


# %% 
def Accuracy(y_pred,y_test):
    n_zeros = np.count_nonzero((y_pred-y_test)==0)
    acc=n_zeros/len(y_pred)
    return acc
def activation_function(z):
    res=[]
    for i in range(len(z)):
        if z[i] >= 0 :
            res.append(1)
        else:
            res.append(-1)
    return res

def net_function(X,w,bias):
    res=np.dot(w,X)+bias;
    return res

def init_weights(neuron_Num):
    random_gen1 = np.random.RandomState(10)
    L1_weights = np.random.normal(0,0.01,(neuron_Num,2))
    random_gen2 = np.random.RandomState(20)
    L1_bias    = np.random.normal(0,0.01,neuron_Num)
    return L1_weights,L1_bias




def Adaline_training(X_train,w1,w2,bias1,bias2,y_train,learning_rate,iteration):
    costs=[]
    for epoch in range(iteration):
        cost=0
        for i in range(len(X_train)):
            z=net_function(X_train[i],w1,bias1)
            y_in=np.dot(w2,activation_function(z))+bias2;
            if y_in >= 0 :
                out_pred=1
            else :
                out_pred=-1   
            errors = y_train[i] - out_pred   
            if errors != 0 and y_train[i]==1 : 
                idx=np.argmin(abs(z))
                p1=learning_rate * X_train[i][0]*(1-z[idx])
                p2=learning_rate * X_train[i][1]*(1-z[idx])
                p=np.zeros((len(w1),2))
                p[idx,0]=p1
                p[idx,1]=p2
                w1 = w1 + p
                bias1[idx] = bias1[idx] + learning_rate*(1-z[idx]) 
            if errors != 0 and y_train[i]==-1 : 
                for i in range(len(z)):
                    if z[i]>0 :
                        p1=learning_rate * X_train[i][0]*(-1-z[i])
                        p2=learning_rate * X_train[i][1]*(-1-z[i])
                        p=np.zeros((len(w1),2))
                        p[i,0]=p1
                        p[i,1]=p2
                        w1 = w1 + p
                        bias1[i] = bias1[i] + learning_rate*(-1-z[i])
            cost = cost+(errors**2)/ 2.0  
        costs.append(cost)
    return w1,bias1,costs


# set one
data1=data.to_numpy()
x1 = data1[0:99,0]
y1 = data1[0:99,1]

# set two
x2 = data1[99:199,0]
y2 = data1[99:199,1]

X_train=data1[0:round(1*len(data1)),0:2]
y_train=data1[0:round(1*len(data1)),2]
for i in range (len(y_train)):
    if y_train[i]== 0:
        y_train[i]=-1

def draw(w, b):
    lines = []
    plt.clf()
    for (w2, w1), b1 in zip(w, b):
        x = np.linspace(-5, 5, 100)
        y = - (w1/w2) * x - b1/w2
        # print(w1, w2, b)
        lines.append((x, y))
        # Line2.append(-domain[i]*(trained_w1[1][0]/trained_w1[1]
        #              [1])-trained_b1[1]/trained_w1[1][1])
        # Line3.append(-domain[i]*(trained_w1[2][0]/trained_w1[2]
        #              [1])-trained_b1[2]/trained_w1[2][1])

    plt.plot(class1.iloc[:, 0], class1.iloc[:, 1], "y*")
    plt.plot(class2.iloc[:, 0], class2.iloc[:, 1], 'm*')
    plt.legend(["data1", "data2"], loc="upper left")
    for l in lines:
        plt.plot(l[0], l[1], "--r")
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    plt.savefig(f"figs/madaline{b.shape[0]}.png")
    plt.show()

# %% 
def train(n):
    np.random.seed(19)
    neuron_Num=n
    w1,bias1=init_weights(neuron_Num)
    w2=np.zeros((neuron_Num))
    w2[0:neuron_Num]=1
    bias2=neuron_Num-1
    learning_rate=0.01
    iteration=200
    trained_w1,trained_b1,costs=Adaline_training(X_train,w1,w2,bias1,bias2,y_train,learning_rate,iteration)
    print(trained_w1,trained_b1)
    draw(trained_w1, trained_b1)


# %%
train(4)
# %%
train(6)
# %%
train(8)
# %%

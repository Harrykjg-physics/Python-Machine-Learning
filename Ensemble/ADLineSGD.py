import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Decision_Region import plot_decision_regions
from numpy.random import seed


class AdlineSGD:

    def __init__(self, eta=0.01, n_iter=20, shuffle=True,
                 random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

# partial_fit：不重新初始化，也就是如果已经初始化过，则不执行初始化了
# 下面的 if 语句是什么意思啊？

    def partial_fit(self, X, y):

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _initialize_weights(self, m):
        self.w_ = np.zeros(m+1)
        self.w_initialized = True

# 下面 _shuffle 函数保持了原来样品与数据的对应关系

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        errors = target - output
        self.w_[1:] += self.eta * xi.dot(errors)
        self.w_[0] += self.eta * errors
        cost = 0.5 * (errors**2)
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-'
                 'learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

ada = AdlineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adline- Stochastic Gradient Decent')
plt.xlabel('sepal length(standardized)')
plt.ylabel('petal length(standardized)')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epoches')
plt.ylabel('Average cost')
plt.show()














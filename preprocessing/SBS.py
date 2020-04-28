from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data')

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols',
                   'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)


class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        # 提取训练集的特征数
        dim = X_train.shape[1]
        # 生成元组，比如说特征数为 8，则生成 (0,1,2,3,4,5,6,7)
        self.indices_ = tuple(range(dim))
        # 生成一个列表，列表只有一个元素，为上面的元组
        self.subsets_ = [self.indices_]
        # 计算包含 所有特征 的模型的表现
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        # 生成一个列表，为上面的表现值，因而要将所有
        # 子序列的表现值与之作比较 —— 维度有可能不变噢！
        self.scores_ = [score]
        # while 循环，开始按规则减少特征数
        while dim > self.k_features:
            scores = []
            subsets = []
            # combination 函数，给定一个可迭代对象（列表，元组等）
            # 给出该迭代对象长度为 r 的所有子集
            for p in combinations(self.indices_, r=dim-1):
                # 计算每个子集的表现
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            # argmax 返回 array 中最大数的下标
            # array 是 numpy 中使用的数据类型
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train,
                    X_test, y_test, indices):
        # 由 combination 生成的元组来指定列
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim(0.5, 1.0)
plt.xlabel('numbers of features')
plt.ylabel('scoring')
plt.grid()
plt.show()

k6 = list(sbs.subsets_[7])
print('这是显示的最佳特征组合')
print(df_wine.columns[1:][k6])

print('这是特征选择前：')
knn.fit(X_train_std, y_train)
print('Training Accuracy: ', knn.score(X_train_std, y_train))
print('Test Accuracy: ', knn.score(X_test_std, y_test))
print('这是特征选择后：')
knn.fit(X_train_std[:, k6], y_train)
print('Training Accuracy: ', knn.score(X_train_std[:, k6], y_train))
print('Test Accuracy: ', knn.score(X_test_std[:, k6], y_test))

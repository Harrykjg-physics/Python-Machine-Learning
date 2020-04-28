import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# 转换 ordinal 标签
# 通过列表初始化一个 DataFrame 数据类型的类
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
    ])
df.columns = ['color', 'size', 'prize', 'class']
print(df)

# Dataframe 这个数据类支持用 [] 索引某列数据，并显示行数

print(df['size'])

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1,
}

df['size'] = df['size'].map(size_mapping)
print(df)

print(size_mapping.items())
inv_size_mapping = {k: v for v, k in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print(df)

# 转换类标签为整数值

class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['class']))}

print(class_mapping)

df['class'] = df['class'].map(class_mapping)
print(df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['class'] = df['class'].map(inv_class_mapping)
print(df)

class_le = LabelEncoder()
y = class_le.fit_transform(df['class'].values)
print(df['class'].values)
print(y)

x = class_le.inverse_transform(y)
print(x)

# 转化 nominal 特征为整数值

X = df[['color', 'size', 'prize']].values
print(X)
# X[:, 0] = class_le.fit_transform(X[:, 0])
# X[:, 1] = class_le.fit_transform(X[:, 1])
# print(X)

ohe = OneHotEncoder(categories=X[:, 0])
print(ohe.fit_transform(X).toarray())



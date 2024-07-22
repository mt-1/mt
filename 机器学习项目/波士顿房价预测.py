from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import seaborn as sns

house = load_boston()

# 描述信息
print(house.DESCR)

x = house.data
y = house.target

df = pd.DataFrame(x, columns=house.feature_names)

df['Target'] = pd.DataFrame(y, columns=['Target'])

print(df.head())

plt.figure(figsize=(15,15))
# 画热力图，数值为两个变量之间的相关系数
p = sns.heatmap(df.corr(), annot=True, square=True)
plt.show()

# 数据标准化
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x = ss.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

# 创建模型
model = LassoCV()
model.fit(x_train, y_train)

# lasso系数
print(model.alpha_)
# 相关系数
print(model.coef_)

print(model.score(x_test, y_test))





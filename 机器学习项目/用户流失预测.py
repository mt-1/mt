import numpy as np

train_data = np.genfromtxt('Churn-Modelling.csv',delimiter=',',dtype=np.str)
test_data = np.genfromtxt('Churn-Modelling-Test-Data.csv',delimiter=',',dtype=np.str)

x_train = train_data[1:,:-1]
y_train = train_data[1:,-1]
x_test = test_data[1:,:-1]
y_test = test_data[1:,-1]

x_train = np.delete(x_train,[0,1,2],axis=1)
x_test = np.delete(x_test,[0,1,2],axis=1)

print(x_train[:5])
print(y_train[:5])

# 字符串类型太多时较为麻烦
# x_train[x_train=='Female'] = 0
# x_train[x_train=='Male'] = 1

from sklearn.preprocessing import LabelEncoder
labelencoder1 = LabelEncoder()
x_train[:,1] = labelencoder1.fit_transform(x_train[:,1])
x_test[:,1] = labelencoder1.fit_transform(x_test[:,1])
labelencoder2 = LabelEncoder()
x_train[:,2] = labelencoder2.fit_transform(x_train[:,2])
x_test[:,2] = labelencoder2.fit_transform(x_test[:,2])

print(x_train[:5])

# 转类型
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

from sklearn.preprocessing import StandardScaler
# 特征 减去 平均值 再 除 方差
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

LR = LogisticRegression()
LR.fit(x_train,y_train)

predictions = LR.predict(x_test)
print(classification_report(y_test,predictions))



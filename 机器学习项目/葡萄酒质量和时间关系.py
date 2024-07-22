import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 载入数据
data = np.genfromtxt('linear.csv',delimiter=',')
# 画图
plt.scatter(data[1:,0],data[1:,1])
plt.title('Age Vs Quality (Test set)')
plt.xlabel('Age')
plt.xlabel('Quality')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data[1:,0],data[1:,1],test_size=0.3)
# print(x_train)
# 1D->2D,给数据增加一个维度，主要是训练模型时，函数要求传入2维数据
x_train = x_train[:, np.newaxis]  # 在列上加一个维度
x_test = x_test[:, np.newaxis]
# print(x_train)
# 训练模型
model = LinearRegression()
model.fit(x_train, y_train)

# 训练集的散点图
plt.scatter(x_train, y_train, c='b')
# 模型对训练集的预测结果
plt.plot(x_train,model.predict(x_train), c='r',linewidth=5)
# 画表头和xy坐标描述
plt.title('Age Vs Quality (Test set)')
plt.xlabel('Age')
plt.xlabel('Quality')
plt.show()


# 测试集的散点图
plt.scatter(x_test, y_test, c='b')
# 模型对训练集的预测结果
plt.plot(x_test,model.predict(x_test), c='r',linewidth=5)
# 画表头和xy坐标描述
plt.title('Age Vs Quality (Test set)')
plt.xlabel('Age')
plt.xlabel('Quality')
plt.show()



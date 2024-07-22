import tensorflow as tf
from d2l import tensorflow as d2l

n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))


def f(x):
    return 2 * tf.sin(x) + x ** 0.8


y_train = f(x_train) + tf.random.normal((n_train,), 0.0, 0.5)  # 测试样本的输出
x_test = tf.range(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
# print(y_hat)
# plot_kernel_reg(y_hat)


X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train,axis=0)
# print("x_train: ",x_train)
# print("tf.expand_dims(x_train,axis=1): ",tf.expand_dims(x_train,axis=1))

# print("X_repeat: ",X_repeat)
# print("\nX_repeat - tf.expand_dims(x_train,axis=1): ",X_repeat - tf.expand_dims(x_train,axis=1))
# print("\nX_repeat - tf.expand_dims(x_train,axis=1): ",(X_repeat - tf.expand_dims(x_train,axis=1)) ** 2/2)
# print("\ntf.nn.softmax(X_repeat - tf.expand_dims(x_train,axis=1)**2/2: ",tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train,axis=1))**2/2,axis=1)  )

attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train,axis=1)) ** 2/2,axis=1)
# print("attention_weights: ",attention_weights)
y_hat = tf.matmul(attention_weights,tf.expand_dims(y_train,axis=1))
# print(y_hat)
# plot_kernel_reg(y_hat)


# print(X_repeat - tf.expand_dims(x_train,axis=1)[0])
# print(tf.nn.softmax(X_repeat - tf.expand_dims(x_train,axis=1)[0]))


# weights = tf.ones((2,10)) * 0.1
# print(weights)
# values = tf.reshape(tf.range(20.0),shape=(2,10))
# print(values)
# res = tf.matmul(tf.expand_dims(weights,axis=1),tf.expand_dims(values,axis=-1)).numpy()
# print(tf.expand_dims(weights,axis=0))
# print(tf.expand_dims(weights,axis=1))
# print(tf.expand_dims(weights,axis=-1))
# print(tf.expand_dims(values,axis=0))
# print(tf.expand_dims(values,axis=1))
# print(tf.expand_dims(values,axis=-1))
# print(res)


a,b = tf.random.normal(shape=(2, 1, 20)), tf.ones((2, 10, 2))
print(a)
print(b)






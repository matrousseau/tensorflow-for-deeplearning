import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)
y_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)

m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0


for x,y in zip(x_data, y_data):
    y_hat = m*x + b
    error += (y-y_hat)**2



optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_step = 15

    for i in range(training_step):
        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])


x_test = np.linspace(-1,11,10)

y_pred = np.add(np.multiply(final_slope,x_test),final_intercept)


plt.plot(x_test, y_pred, 'r')
plt.plot(x_data, y_data, '*')
plt.show()
print("error : ", error)

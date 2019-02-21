import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------- Generate data -------------------

x_data = np.linspace(0.0,10,1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(x_data, columns=['X Data'])
y_df = pd.DataFrame(y_true, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)


# print(my_data.head(5))
#
# my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
# plt.show()

# ---------------------- Variables ----------------------

batch_size = 16

m = tf.Variable(0.44)
b = tf.Variable(0.87)

X_placeholder = tf.placeholder(tf.float32, [batch_size])
Y_placeholder = tf.placeholder(tf.float32, [batch_size])

y_model = m * X_placeholder + b

init = tf.global_variables_initializer()

# -------------------- Loss Function ---------------------

error = tf.reduce_sum(tf.square(Y_placeholder-y_model))

# --------------------- Optimizer -------------------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# ---------------------- Train ----------------------------

with tf.Session() as sess:
    sess.run(init)
    batches = 10000
    for i in range (batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {X_placeholder:x_data[rand_ind],Y_placeholder:y_true[rand_ind]}
        sess.run(train, feed_dict=feed)
    final_slope, final_intercept = sess.run([m,b])

print("final_slope : ", final_slope)
print("final_intercept : ",final_intercept )

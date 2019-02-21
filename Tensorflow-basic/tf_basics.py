import tensorflow as tf
print(tf.__version__)


# ---------------- Addition ---------------

a = tf.constant(10)
b = tf.constant(15)

with tf.Session() as sess:
    result = sess.run(a + b)

print(result)

# ---------------- Generate tensor --------

fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
my_randn = tf.random_normal((4,4), mean=0, stddev=1.0)
myrandu = tf.random_uniform((4,4), minval=0, maxval=1)

# --------------- Interactive Sess ---------

my_ops = [a, fill_mat, myzeros, my_randn]

sess = tf.InteractiveSession()

for op in my_ops:
    print(sess.run(op), '\n')

# ----------------- Matrix ----------------

a = tf.constant([[1,2],[3,4]])
b = tf.constant([[10],[4]])

print(sess.run(tf.matmul(a,b)))


# --------- Placeholders/Variables-----------
# We need to initialize variables before running the Session

sess = tf.InteractiveSession()

my_tensor = tf.random_uniform((4,4),0,1)
my_var = tf.Variable(initial_value=my_tensor)
my_ph = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(my_var))

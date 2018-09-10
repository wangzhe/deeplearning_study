import numpy as np
import tensorflow as tf

coefficient = np.array([[1.], [-20.], [100.]])

w = tf.Variable(0, dtype=tf.float32)  # Variable is what we want to train
x = tf.placeholder(tf.float32, [3, 1])
# cost = tf.add(tf.add(w ** 2, tf.multiply(-10., w)), 25)
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]  # forward function
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:    # session = tf.Session()
    session.run(init)            # session.run(init)
    session.run(w)
    for i in range(1000):
        session.run(train, feed_dict={x: coefficient})
    print(session.run(w))

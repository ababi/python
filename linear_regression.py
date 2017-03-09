import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Create input data using Numpy

x_data = np.random.rand(100).astype(np.float32)

y_data = x_data*0.6 + 0.5


# Create model
W = tf.Variable(tf.random_uniform([1],0.0,1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')

y = W * x_data + b

# Building the training function

# define loss
loss = tf.reduce_mean(tf.square(y-y_data))

# define the optimizer with its learning rate

optimizer = tf.train.AdamOptimizer(0.01)

# Minimize loss

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# create the Session and send it  to runtime of tensorflow to execute

sess = tf.Session()

sess.run(init)

y_initial = sess.run(y)   # Save te initial values for plotting

# Perform training

for step in range(10000):
    sess.run(train)
    if step % 100 == 0:
        print step, sess.run([W,b])

print sess.run([W,b])


plt.plot(x_data,y_data, '.', label ="target_values")
plt.plot(x_data,y_initial,'_', label="initial_values")
plt.plot(x_data,sess.run(y),'.', label="trainied_values" )
plt.plot(sess.run(W),'.', label="weight")
plt.legend()
plt.ylim(0,1.0)
plt.show()

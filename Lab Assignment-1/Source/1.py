# MNIST dataset is used for this project
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


import tensorflow as tf

# parameters have been taken in consideration for logistic regression
learning_rate = 0.01
training_iteration_loop = 50
batch_number = 50
display_step = 3

# TF graph input
x1 = tf.placeholder("float", [None, 784]) # 784 is of data image size 28*28
y1 = tf.placeholder("float", [None, 10]) # 10 classes have been employed

# Create a model

# weights of the model has been set
Wa = tf.Variable(tf.zeros([784, 10]))
ba = tf.Variable(tf.zeros([10]))

# Construct a linear model
model = tf.nn.softmax(tf.matmul(x1, Wa) + ba) # Softmax

# Minimize error using cross entropy
# Cross entropy
# calculating the error rates between 0 and 1
cost_function = -tf.reduce_sum(y1*tf.log(model))
# optimiser can be used for reducing the cost_function
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initialise command has been used for the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)
    # Training cycle
    for iteration in range(training_iteration_loop):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_number)
        # Loop condition
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_number)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x1: batch_xs, y1: batch_ys})
            # Compute loss of average
            avg_cost += sess.run(cost_function, feed_dict={x1: batch_xs, y1: batch_ys})/total_batch
        # iteration loop
        if iteration % display_step == 0:
            print ("epoch:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    writer.close()
    print ("Tuning completed!")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y1, 1))
    # accuracy is calculated
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print ("Accuracy:", accuracy.eval({x1: mnist.test.images, y1: mnist.test.labels}))
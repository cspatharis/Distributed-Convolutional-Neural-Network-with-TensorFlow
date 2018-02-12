import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
#import cifar10

# Convolutional Layer 1
filter_size1 = 5          # 5x5
num_filters1 = 16

# Convolutional Layer 2
filter_size2 = 5
num_filters2 = 36        

# Fully-connected layer
fc_size = 128             # Number of neurons

#import data
#mnist handwritten dataset can be imported directly from tensorflow library
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

img_size = 28

# Size of image as 1-d vector
img_vector_size = img_size * img_size

# Height and Width
img_shape = (img_size, img_size)

# Number of colour channels: 1 = gray-scale , 3 = RGB
num_channels = 1

# Number of classes
num_classes = 10

def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights-filters and biases
    weights = create_weights(shape=shape)
    biases = create_biases(length=num_filters)

    # Convolution operation
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    # Use pooling to down-sample the image resolution
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # ReLU - calculates max(x, 0) for each input pixel x.
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()

    # Number of features = img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def fully_connected_layer(input,num_inputs, num_outputs, use_relu=True):
    
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # w * X + b
    layer = tf.matmul(input, weights) + biases

    if use_relu:
    	layer = tf.nn.relu(layer)

    return layer


#Tensorflow placeholders for the variables
#The variables needed in order TensorFlow to create the computational graph
x = tf.placeholder(tf.float32, shape=[None, img_vector_size], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

#Conv Layer 1
layer_conv1, weights_conv1 =  new_conv_layer(input=x_image,
											num_input_channels=num_channels,
											filter_size=filter_size1,
											num_filters=num_filters1,
											use_pooling=True)

#Conv Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
						                   num_input_channels=num_filters1,
						                   filter_size=filter_size2,
						                   num_filters=num_filters2,
						                   use_pooling=True)

#Flatten Layer - Reshape to 2d-tensors
layer_flat, num_features = flatten_layer(layer_conv2)

#Fully Connected Layer
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

#Second fully connected Layer - Output the 10 classes
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

#Softmax
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)


#Cost Function Optimization
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

#Optimization Method - Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#Run TensorFlow
session = tf.Session()

# Initialize Variables
session.run(tf.global_variables_initializer())

train_batch_size = 64


total_iterations = 0
def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Create a dict with batch images and corresponding classes
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print accuracy status every 10 iterations
        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))


    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


if __name__ == "__main__":
    num_iterations = 10000
    optimize(num_iterations=num_iterations)
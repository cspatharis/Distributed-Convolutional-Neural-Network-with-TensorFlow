import socket
import tensorflow as tf

#import data
from tensorflow.examples.tutorials.mnist import input_data


# Define the master and the worker nodes
# At least one master and at least one worker node
# Include the port for the communication
tf.app.flags.DEFINE_string("ps_hosts", "master:42000",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "worker1:42001,worker2:42002",
                           "Comma-separated list of hostname:port pairs")
#For more nodes
#tf.app.flags.DEFINE_string("worker_hosts", "worker1:42001,worker2:42002",worker3:42003",worker4:42004",
#                                                           "Comma-separated list of hostname:port pairs")



# Two possible types of job: 'ps' - parameter server and 'worker'
# The parameter server has task_index 0 within the job 'ps' and there is only one parameter server
# For the 'workers' we assigned task_index 0, 1, ...
# IMPORTANT: We have to name every vm instance we the same name as the following, based on its job
name = socket.gethostname()
# Flags for defining the tf.train.Server
if name == 'master':
    tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
elif name == 'worker1':
    tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
elif name == 'worker2':
    tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 1, "Index of task within the job")
#elif name == 'worker3':
#    tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
#    tf.app.flags.DEFINE_integer("task_index", 2, "Index of task within the job")
#elif name == 'worker4':
#    tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
#    tf.app.flags.DEFINE_integer("task_index", 3, "Index of task within the job")

tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS
#Define the size of the training images
img_size = 28


def create_weights(shape):
    #tf.truncated_normal returns random values from a truncated normal distribution.
    return tf.truncated_normal(shape, stddev=0.1)

def create_biases(shape):
    return tf.constant(0.1, shape=shape)

#Convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Max Pooling
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def main(_):
      # Create a list of parameter server and workers
      ps_hosts = FLAGS.ps_hosts.split(",")
      worker_hosts = FLAGS.worker_hosts.split(",")

      #tf.train.ClusterSpec defines what the cluster looks like
      #Each machine is aware of itelf and all the other machines
      cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

      #Create  a server in order to communicate
      #It is important to provide the server with the ClusterSpec object
      #along with the name of the current job and the task_index of the job
      #so the server is aware of each machine's role within the system
      server = tf.train.Server(cluster,
                               job_name=FLAGS.job_name,
                               task_index=FLAGS.task_index)

      #If the machine is 'ps', we call server.join() - this blocks until the server is manually shut down
      #If the machine is a worker, we need to define all of the variables and operations that define our model.
      if FLAGS.job_name == "ps":
        server.join()
      elif FLAGS.job_name == "worker":

        #We must specify which device will take care of which operations
        #Option 1: manually specify which device will handle specific operations
        #Option 2: use a tf.train.replica_device_setter, which automatically handles assigning tasks to devices
        #All tf.Variable() objects will be stored on 'ps', while computational tasks will be placed on 'worker'
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/replica:0/task:%d" % FLAGS.task_index,
            cluster=cluster)):

          global_step = tf.Variable(0, trainable=False)
          is_chief = FLAGS.task_index == 0

          # 28x28 size of image
          img_vector_size = img_size * img_size
          x = tf.placeholder(tf.float32, [None, img_vector_size])
          y_ = tf.placeholder(tf.float32, [None, 10])

          #Read the input data
          mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


          # Initialize the weights and biases
          W = tf.Variable(tf.zeros([img_vector_size, 10]))
          b = tf.Variable(tf.zeros([10]))

          #Softmax
          y = tf.nn.softmax(tf.matmul(x, W) + b)

          #Conv Layer 1
          W_conv1 = create_weights([5, 5, 1, 32])
          b_conv1 = create_biases([32])

          # reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height
          # and the final dimension corresponding to the number of color channels - 1 for greyscale and 3 for RGB
          x_image = tf.reshape(x, [-1, 28, 28, 1])

          #Convolution and max pooling
          h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
          h_pool1 = max_pool(h_conv1)

          #Conv Layer 2
          W_conv2 = create_weights([5, 5, 32, 64])
          b_conv2 = create_biases([64])
          h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
          h_pool2 = max_pool(h_conv2)

          #Fully Connected Layer 1
          W_fc1 = create_weights([7 * 7 * 64, 1024])
          b_fc1 = create_biases([1024])
          h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
          h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

          #Dropout
          keep_prob = tf.placeholder(tf.float32)
          h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

          # Second fully connected Layer - Output the 10 classes
          W_fc2 = weight_variable([1024, 10])
          b_fc2 = bias_variable([10])
          y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


          # Cross_entropy function, training step, correct prediction, and accuracy functions.
          cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
          train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy, global_step=global_step)
          correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

          #saver allows saving/restoring variables to/from checkpoints during training
          saver = tf.train.Saver()
          #summary tracks all summaries of the graph
          summary = tf.summary.merge_all()
          #init defines the operation to initialize all tf.Variable()s
          init = tf.global_variables_initializer()

          # Configuration
          # allow_soft_placement=True: allows computations to be placed on devices that are not explicitly defined
          # log_device_placement=False: don't log device placement
          # device_filters: only use devices whose names match the names provided here
          sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/replica:0/task:%d" % FLAGS.task_index])

          if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
          else:
            print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

        #Create a "supervisor", which oversees the training process
        #There can only be one chief node
        #The chief is responsible for initializing the model
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir="train_logs",
                                 summary_op=summary,
                                 init_op=init,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)
        server_grpc_url = "grpc://" + worker_hosts[FLAGS.task_index]

        #sv.prepare_or_wait_for_session: if a worker is the chief, it will initialize all that needs to be initialized
        # or if a worker is not the chief will wait until the chief has initialized everything
        with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:
          step = 0
          #If anything goes wrong, sv.should_stop() will halt execution on a worker
          while (not sv.should_stop()) and (step < 10000):
            # Run a training step asynchronously.
            batch = mnist.train.next_batch(FLAGS.batch_size)
            if step % 10 == 0:
              train_accuracy = accuracy.eval(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
              print('step %d, training accuracy %g' % (step, train_accuracy))
            _, step = sess.run([train_step, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

          print('test accuracy %g' % accuracy.eval(feed_dict={
              x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        print("ALL FINISHED")
        sv.stop()



if __name__ == "__main__":
  tf.app.run()
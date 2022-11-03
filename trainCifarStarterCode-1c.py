from scipy import misc
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import tensorflow as tf
if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import random
import matplotlib.pyplot as plt
import matplotlib as mp
import imageio.v2 as imageio

# --------------------------------------------------
# setup

def weight_variable(shape, name):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    initial = tf.truncated_normal(shape, stddev=0.01)
    W = tf.Variable(initial)

    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)

    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_max

result_dir = './results-1c/' # directory where the results from the training are saved

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 100
nsamples = nclass * ntrain

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = imageio.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = imageio.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

plt.imshow(Train[-1,:,:,0],cmap = 'gray')
plt.show()
sess = tf.InteractiveSession()

start_time = time.time() # start timing

tf_data = tf.placeholder("float", shape=[None,imsize,imsize,nchannels]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder("float", shape=[None,nclass]) #tf variable for labels

# --------------------------------------------------
# model
# first convolutional layer and max pooling
W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer and max pooling
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# first fully connected layer
W_fc1 = weight_variable([7*7*64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully connected layer and softmax
W_fc2 = weight_variable([1024, nclass], 'W_fc2')
b_fc2 = bias_variable([nclass])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')


# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
learningRate = [1e-2]
Loss_list = {1e-2:[]}
Accuracy_list = {1e-2:[]}

for lr in learningRate:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))
    # optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    # optimizer = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)
    optimizer = tf.train.MomentumOptimizer(lr, momentum = 0.9).minimize(cross_entropy)
    # try AdamOptimizer, AdagradOptimizer, MomentumOptimizer(0.9), and Learning rate: 1e-4, 1e-3, 1e-2
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # --------------------------------------------------
    # optimization

    sess.run(tf.global_variables_initializer())
    batch_xs = np.zeros((batchsize,imsize,imsize,nchannels)) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_ys = np.zeros((batchsize,nclass)) #setup as [batchsize, the how many classes]

    for i in range(40000): # try a small iteration size once it works then continue
        perm = np.arange(nsamples)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
            batch_ys[j,:] = LTrain[perm[j],:]

        if i % 200 == 0:
            # Record train accuracy and loss in dictionary
            train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
            train_loss = cross_entropy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
            Accuracy_list[lr].append(train_accuracy)
            Loss_list[lr].append(train_loss)
            W_firstLayer = W_conv1.eval()

            print("step: {} with lr: {}, training accuracy: {}, loss: {}".format(i, lr, train_accuracy, train_loss))

        # save the checkpoints every 1100 iterations
        # if i % 1000 == 0:
        #         checkpoint_file = os.path.join(result_dir, 'checkpoint')
        #         saver.save(sess, checkpoint_file, global_step=i)

        if i%10 == 0:
            #calculate train accuracy and print it
           optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

    # --------------------------------------------------
    # test
    Activation_firstL= np.array([h_conv1.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})])
    Activation_secL= np.array(h_conv2.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

    print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

    stop_time = time.time()
    print('The training takes %f second to finish' % (stop_time - start_time))

    print('FirstLayer activation stat: Min: {},Max :{},Mean:{},std:{},var:{}'.format(Activation_firstL.min(),
                                                                                     Activation_firstL.max(),
                                                                                     Activation_firstL.mean(),
                                                                                     Activation_firstL.std(),
                                                                                     Activation_firstL.var()))
    print('SecondLayer activation stat: Min: {},Max :{},Mean:{},std:{},var:{}'.format(Activation_secL.min(),
                                                                                     Activation_secL.max(),
                                                                                     Activation_secL.mean(),
                                                                                     Activation_secL.std(),
                                                                                     Activation_secL.var()))

sess.close()

fig = plt.figure()
fig.suptitle('Visualize First Conv Layer',fontsize = 18)
for i in range(32):
    ax = fig.add_subplot(4, 8, 1 + i)
    ax.imshow(W_firstLayer[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.show()
fig.savefig('FistConv.png')
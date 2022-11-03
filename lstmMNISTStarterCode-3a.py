import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #call mnist function

# import tensorflow_datasets
# mnist = tensorflow_datasets.load('mnist')

learningRate = 1e-3
trainingIters = 60000
batchSize = 128
displayStep = 10

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 128 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

# x = tf.placeholder('float', [None, nSteps, nInput])
# y = tf.placeholder('float', [None, nClasses])
#
# weights = {
# 	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
# }
#
# biases = {
# 	'out': tf.Variable(tf.random_normal([nClasses]))
# }

def RNN(x, weights, biases, Method, nHidden):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels

	if Method == 'lstm':
		lstmCell =  tf.contrib.rnn.LSTMCell(nHidden)#find which lstm to use in the documentation
		outputs, states = tf.contrib.rnn.static_rnn(cell = lstmCell, inputs = x,dtype = tf.float32)  # for the rnn where to get the output and hidden state
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	elif Method == 'gru':
		gruCell = tf.contrib.rnn.GRUCell(nHidden)
		outputs, states = tf.contrib.rnn.static_rnn(cell = gruCell, inputs = x,dtype = tf.float32)  # for the rnn where to get the output and hidden state
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	elif Method == 'rnn':
		rnnCell = tf.contrib.rnn.BasicRNNCell(nHidden)
		outputs, states = tf.contrib.rnn.static_rnn(cell = rnnCell, inputs = x,dtype = tf.float32)  # for the rnn where to get the output and hidden state
		return tf.matmul(outputs[-1], weights['out']) + biases['out']

def train(Method , nHidden):
	x = tf.placeholder('float', [None, nSteps, nInput])
	y = tf.placeholder('float', [None, nClasses])

	weights = {
		'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
	}

	biases = {
		'out': tf.Variable(tf.random_normal([nClasses]))
	}

	pred = RNN(x, weights, biases, Method, nHidden)

	#optimization
	#create the cost, optimization, evaluation, and accuracy
	#for the cost softmax_cross_entropy_with_logits seems really good
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

	correctPred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
	accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32, name = 'Correct_Prediection'), name = 'accuracy')

	init = tf.initialize_all_variables()

	Loss_list = []
	Accuracy_list = []

	with tf.Session() as sess:
		sess.run(init)
		step = 1

		while step* batchSize < trainingIters:
			batchX, batchY = mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
			batchX = batchX.reshape((batchSize, nSteps, nInput))

			sess.run(optimizer, feed_dict={x:batchX, y: batchY})

			if step % displayStep == 0:
				acc = accuracy.eval(feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
				loss = cost.eval(feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
				Accuracy_list.append(acc)
				Loss_list.append(loss)
				print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc))
			step +=1
		print('Optimization finished')

		testData = mnist.test.images.reshape((-1, nSteps, nInput))
		testLabel = mnist.test.labels
		Test_Accuracy = sess.run(accuracy, feed_dict={x: testData, y: testLabel})
		print("Testing Accuracy:", \
			Test_Accuracy)

	sess.close()
	return Accuracy_list, Loss_list, Test_Accuracy

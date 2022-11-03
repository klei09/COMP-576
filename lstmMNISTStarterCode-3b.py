tf.reset_default_graph()

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #call mnist function

# import tensorflow_datasets
# mnist = tensorflow_datasets.load('mnist')

learningRate = 8e-3
trainingIters = 60000
batchSize = 128
displayStep = 10

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
# nHidden = 128 #number of neurons for the RNN
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

def RNN(x, weights, biases,nHidden,Method):
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

def train(method , nHidden):
	x = tf.placeholder('float', [None, nSteps, nInput])
	y = tf.placeholder('float', [None, nClasses])

	weights = {
		'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
	}

	biases = {
		'out': tf.Variable(tf.random_normal([nClasses]))
	}

	pred = RNN(x, weights, biases,nHidden,method)

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

			sess.run(optimizer, feed_dict={x: batchX, y: batchY})

			if step % displayStep == 0:
				acc = accuracy.eval(feed_dict={x: batchX, y: batchY})
				loss = cost.eval(feed_dict={x: batchX, y: batchY})
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


# do training for three methods
# rnn_acc, rnn_loss, rnn_test = train('rnn',128)
# lstm_acc, lstm_loss,lstm_test = train('lstm',128)
# gru_acc, gru_loss,gru_test = train('gru',128)
rnn_acc1, rnn_loss1, rnn_test1 = train('rnn',64)
tf.reset_default_graph()
rnn_acc2, rnn_loss2, rnn_test2 = train('rnn',128)
tf.reset_default_graph()
rnn_acc3, rnn_loss3, rnn_test3 = train('rnn',256)


# #plot Accuracy for three methods
# fig, ax = plt.subplots()
# fig.suptitle('Same 128 Hiddenlayers,Different Method', fontsize = 18)
# ax.plot(range(len(rnn_acc)),rnn_acc,'k',label = 'Train Accuracy with RNN ')
# ax.plot(range(len(lstm_acc)), lstm_acc, 'g', label = 'Train Accuracy with LSTM ')
# ax.plot(range(len(gru_acc)),gru_acc,'r',label = 'Train Accuracy with GRU ')
# ax.set(xlabel = 'Iteration', ylabel = 'Accuracy')
# ax.legend(loc = 'lower right', shadow = True)
# plt.show()
# fig.savefig('DifferentMethodACC.png')
#
# #plot loss for three methods
# fig, bx = plt.subplots()
# fig.suptitle('Same 128 Hiddenlayers,Different Method', fontsize = 18)
# bx.plot(range(len(rnn_loss)),rnn_loss,'k',label = 'Train LOSS with RNN ')
# bx.plot(range(len(lstm_loss)), lstm_loss, 'g', label = 'Train LOSS with LSTM ')
# bx.plot(range(len(gru_loss)),gru_loss,'r',label = 'Train LOSS with GRU ')
# bx.set(xlabel = 'Iteration', ylabel = 'Loss')
# bx.legend(loc = 'upper right', shadow = True)
# plt.show()
# fig.savefig('DifferentMethodLOSS.png')
#
# print('Test Accuracy: RNN:{}, LSTM:{},GRU :{}'.format(rnn_test,lstm_test,gru_test))

#plot Accuracy for diff layers
fig, ax = plt.subplots()
fig.suptitle('Same RNN method, Different nHidden', fontsize = 18)
ax.plot(range(len(rnn_acc1)),rnn_acc1,'k',label = 'Train Accuracy with 64' )
ax.plot(range(len(rnn_acc2)),rnn_acc2,'g',label = 'Train Accuracy with 128')
ax.plot(range(len(rnn_acc3)),rnn_acc3,'r',label = 'Train Accuracy with 256')
ax.set(xlabel = 'Iteration', ylabel = 'Accuracy')
ax.legend(loc = 'lower right', shadow = True)
plt.show()
fig.savefig('DifferentLayerACC.png')

#plot loss for diff layers
fig, bx = plt.subplots()
fig.suptitle('Same RNN method, Different nHidden', fontsize = 18)
bx.plot(range(len(rnn_loss1)),rnn_loss1,'k',label = 'Train LOSS with 64')
bx.plot(range(len(rnn_loss2)),rnn_loss2,'g',label = 'Train LOSS with 128')
bx.plot(range(len(rnn_loss3)),rnn_loss3,'r',label = 'Train LOSS with 256')
bx.set(xlabel = 'Iteration', ylabel = 'Loss')
bx.legend(loc = 'upper right', shadow = True)
plt.show()
fig.savefig('DifferentLayerLOSS.png')

print('Test Accuracy: n64:{}, n128:{}, n256 :{}'.format(rnn_test1,rnn_test2,rnn_test3))
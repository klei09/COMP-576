import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

########################################################################################################################
########################################################################################################################
# BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################


class Layer(object):
    def __init__(self, nn_input_dim, nn_output_dim, last_layer = 0, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.last_layer = last_layer

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_input_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_output_dim))


    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        self.z = np.dot(X, self.W) + self.b

        # Intermediate Layer:
        if self.last_layer == 0:
            self.a = actFun(self.z)
        # Last Layer:
        else:
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return None



class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, num_layers, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.num_layers = num_layers
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda


        #Create "n" Layers:
        self.LayerList = []
        for count in range(num_layers - 1):
            if count == 0:
                x = Layer(nn_input_dim = self.nn_input_dim, nn_output_dim = self.nn_hidden_dim, last_layer = 0, actFun_type = self.actFun_type, seed = count)
            elif (count < num_layers - 2):
                x = Layer(nn_input_dim=self.nn_hidden_dim, nn_output_dim=self.nn_hidden_dim, last_layer=0, actFun_type=self.actFun_type, seed = count)
            else:
                x = Layer(nn_input_dim=self.nn_hidden_dim, nn_output_dim=self.nn_output_dim, last_layer = 1, actFun_type=self.actFun_type, seed = count)
            self.LayerList.append(x)
            # print(self.LayerList[count].nn_input_dim)
            # print(self.LayerList[count].nn_output_dim)
            # print(self.LayerList[count].last_layer)



    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        if type == 'tanh':
            value = np.tanh(z)
        elif type == 'sigmoid':
            value = 1 / (1 + np.exp(-z))
            #value = np.divide( np.ones(len(z)), (1 + np.exp(-z)))
        elif type == 'relu':
            #value = np.max(z, 0)
            value = z * (z > 0)
        else:
            raise Exception ('Invalid activation function type!')

        return value

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        if  type == 'tanh':
            value = np.tanh(z)
            diffvalue =  1 - (value * value)

            #diffvalue = (1 - np.power(z, 2))


        elif type == 'sigmoid':
            value = 1 / (1 + np.exp(-z))
            #value = np.divide(np.ones(len(z)), (1 + np.exp(-z)))
            diffvalue = value * (1 - value)
        elif type == 'relu':
            #diffvalue = np.array(z)
            #diffvalue[diffvalue < 0] = 0
            #diffvalue[diffvalue > 0] = 1

            diffvalue = (z >= 0) * 1

            #if z > 0:
                #diffvalue = 1
            #else:
                #diffvalue = 0
        else:
            raise Exception ('Invalid activation function type!')

        return diffvalue

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        for count in range(self.num_layers - 1):
            if count == 0:
              self.LayerList[count].feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            elif (count < self.num_layers - 2):
              self.LayerList[count].feedforward(self.LayerList[count-1].a, lambda x: self.actFun(x, type=self.actFun_type))
            else:
              self.LayerList[count].feedforward(self.LayerList[count-1].a, lambda x: self.actFun(x, type=self.actFun_type))
              self.probs = self.LayerList[count].probs

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # CALCULATION OF THE LOSS
        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))
        # data_loss =

        # Add regulatization term to loss
        tempsum = 0
        for count in range(self.num_layers - 1):
            tempsum += np.sum(np.square(self.LayerList[count].W))
        data_loss += self.reg_lambda / 2 * tempsum

        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        dW = []
        db = []
        delta = []
        for count in range(self.num_layers-1):
            dW.append([])
            db.append([])
            delta.append([])


        #print("Back Prop")
        # print (self.num_layers)

        num_examples = len(X)
        # delta3 = self.probs
        # delta3[range(num_examples), y] -= 1
        delta[self.num_layers-2] = self.probs
        delta[self.num_layers-2][range(num_examples), y] -= 1


        for count in range(self.num_layers-2, -1, -1):
            # print (count)
            if count == self.num_layers-2:
                dW[count] = (self.LayerList[count-1].a.T).dot(delta[count])
                db[count] = np.sum(delta[count], axis=0, keepdims=True)

            elif count == 0:
                delta[count] = delta[count + 1].dot(self.LayerList[count + 1].W.T) * self.diff_actFun(self.LayerList[count].z, self.actFun_type)
                dW[count] = (X.T).dot(delta[count])
                db[count] = np.sum(delta[count], axis=0, keepdims=True)
            else:
                delta[count] = delta[count+1].dot(self.LayerList[count+1].W.T) * self.diff_actFun(self.LayerList[count].z, self.actFun_type)
                dW[count] = (self.LayerList[count - 1].a.T).dot(delta[count])
                db[count] = np.sum(delta[count], axis=0, keepdims=True)


        # dW[1] = (self.LayerList[0].a.T).dot(delta3)
        # db[1] = np.sum(delta3, axis=0, keepdims=True)
        # delta2 = delta3.dot(self.LayerList[1].W.T) * self.diff_actFun(self.LayerList[0].z, self.actFun_type)
        # dW[0] = np.dot(X.T, delta2)
        # db[0] = np.sum(delta2, axis=0)

        return dW, db


    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for count in range(self.num_layers -1):
                dW[count] += self.reg_lambda * self.LayerList[count].W
                self.LayerList[count].W += -epsilon * dW[count]
                self.LayerList[count].b += -epsilon * db[count]


            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)

def main():

    # # Sample Run:
    # X, y = generate_data()

    X, y = generate_circle_data()
    # print (X.shape)
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    # num_layers tells you the total number of layers in the network.
    # num_layers = 5, means 1 input layer, 1 output layer, 3 hidden layers:
    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim = 4, nn_output_dim=2, num_layers = 8, actFun_type='tanh')
    # model.fit_model(X,y)
    model.fit_model(X, y, epsilon = 0.001)
    model.visualize_decision_boundary(X,y)


if __name__ == "__main__":
    main()
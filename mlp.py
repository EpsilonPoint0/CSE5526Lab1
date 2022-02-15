import numpy as np
import matplotlib as plt
import matplotlib.pyplot as mp

# The MLP class is the implementation of the multilayer perceptron algorithm with backpropagation
# To use the class, instansiate an object with an array that has a length of the number of layers in the perceptron and each value corresponds to the number of nodes in that layer
# Optional parameters are:
# alpha: The momentum term in the weight update. Default is none
# nu: The learning rate in the weight update. The default is 0.0002
# auto_stop: the minimum number of epochs to run before potentially stopping early
# Then use the train function on the instantiated object to train the network
# The parameters are:
# X: the data
# y: the true classifications
# epochs: The number of times to run the algorithm. Default is 10000
# Finally, use the step_function function to return the predictions and the classification_error function to return the classification error
# Use plot_data to plot the data
class MLP:

    # Initialize data
    def __init__(self, layers, alpha=0, nu=0.0002, auto_stop=10000):
        self.W1 = []
        self.W2 = []
        self.layers = layers
        self.nu = nu
        self.alpha = alpha
        self.auto_stop = auto_stop

        # Uniformly generate two sets of data and targets
        self.test1 = np.random.uniform(low=-1, high=1, size=(200, 2))
        self.target1colors = []
        self.target1nums = []
        for pt in self.test1:
            if np.abs(np.sin(np.pi*pt[0])) > np.abs(pt[1]):
                self.target1colors.append('blue')
                self.target1nums.append(1)
            else:
                self.target1colors.append('red')
                self.target1nums.append(0)
        

        self.test2 = np.random.uniform(low=-1, high=1, size=(200, 2))
        self.target2colors = []
        self.target2nums = []
        for pt in self.test2:
            if np.abs(np.sin(np.pi*pt[0])) > np.abs(pt[1]):
                self.target2colors.append('blue')
                self.target2nums.append(1)
            else:
                self.target2colors.append('red')
                self.target2nums.append(0)

        # Initialize weight vectors
        for i in np.arange(0, len(layers) -2):
            W1 = np.random.uniform(low=-0.1, high = 0.1, size=(layers[i] +1, layers[i + 1] + 1))
            self.W1.append(W1 / np.sqrt(layers[i]))
            W2 = np.random.uniform(layers[i] +1, layers[i + 1] + 1)
            self.W2.append(W2 / np.sqrt(layers[i]))

        w = np.random.uniform(low=-0.1, high = 0.1, size=(layers[-2] +1, layers[-1]))
        self.W1.append(w / np.sqrt(layers[-2]))


    def sigmoid(self, Z):
        return 1.0/(1+np.exp(-Z))

    def sigmoid_derivative(self, Z):
        return Z * (1 - Z)

    # Split data into x and y coordinates to plot
    def plot_data(self):
        x_vals = []
        y_vals = []
        for pt in self.test1:
            x_vals.append(pt[0])
            y_vals.append(pt[1])

        self.x_vals = x_vals
        self.y_vals = y_vals

        for i in range(len(x_vals)):
            mp.scatter(x_vals[i], y_vals[i], c = self.target1colors[i])



    def train(self, X, Y, epochs=10000):
        update=100

        # Create biases with unifrom distrobution and add to data
        b1 = np.random.uniform(low=-0.1, high=0.1, size=200)
        self.b1 = b1
        X = np.c_[X, b1]

        loss_per_epoch = {}

        # Train the network
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, Y):
                self.weight_update(x, target)
            
            if epoch == 0 or (epoch + 1) % update == 0:
                loss = self.loss_function(X, Y)
                loss_per_epoch[epoch + 1] = loss
                print("Current Epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))
                # Stop early based on difference of loss per 100 epochs
                if epoch > self.auto_stop and (loss_per_epoch[epoch + 1] - loss_per_epoch[epoch -99]) < 0.00001:
                    print("Stopping...")
                    return 
                
                
    
    def weight_update(self, x, y):

        output_activations = [np.atleast_2d(x)]
        #print(self.W1)

        # Forward propagation phase
        for layer in np.arange(0, len(self.W1)):

            net_input = output_activations[layer].dot(self.W1[layer])

            net_output = self.sigmoid(net_input)

            output_activations.append(net_output)

        # Backpropagate and use chain rule for all layers of network   
        error = output_activations[-1] - y

        partial_deriv = [error * self.sigmoid_derivative(output_activations[-1])]

        for layer in np.arange(len(output_activations) - 2, 0, -1):

            delta = partial_deriv[-1].dot(self.W1[layer].T)
            delta = delta * self.sigmoid_derivative(output_activations[layer])
            partial_deriv.append(delta)

        partial_deriv = partial_deriv[::-1]

        # Update the weights with an optional momentum term alpha
        for layer in np.arange(0, len(self.W1)):
            if self.alpha == 0:
                self.W1[layer] += -self.nu * output_activations[layer].T.dot(partial_deriv[layer])
            else:
                self.W1[layer] += -self.nu * output_activations[layer].T.dot(partial_deriv[layer]*self.alpha)

    def output(self, X, last_layer=True):

        prediction = np.atleast_2d(X)

        if last_layer:
            prediction = np.c_[prediction, self.b1[0]]
            #print (prediction)

        # Pass through activation function
        for layer in np.arange(0, len(self.W1)):

            prediction = self.sigmoid(np.dot(prediction, self.W1[layer]))

        return prediction

    def loss_function(self, X, targets):
        predictions = self.output(X, last_layer=False)
        loss = self.nu * np.sum((predictions - targets) ** 2)
        return loss

    def step_function(self, data):
        predictions = []
        sum = 0
        length = []
        for x in data:
            pred = self.output(x)[0][0]
            length.append(pred)
            sum += pred
        
        avg = sum / len(length)
        for i in length:
            predictions.append(1) if i > avg else predictions.append(0)

        return predictions

    def print_data(self):
        print(self.test2)
        print(self.target2nums)

    def classification_error(self, results):
        correct = 0
        incorrect = 0
        for i in range(len(results)):
            if(MLP1.target1nums[i] == results[i]):
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        return incorrect / len(results)


MLP1 = MLP([2, 20, 1], alpha=0.8,auto_stop=1000)

MLP1.plot_data()

MLP1.train(MLP1.test1, MLP1.target1nums, epochs=10000)

results = MLP1.step_function(MLP1.test2)


classification_error = MLP1.classification_error(results)
print(classification_error)

x = np.linspace(-1, 1, 100)
y = np.sin(4.7*x)
mp.plot(x, y)


#mp.show()




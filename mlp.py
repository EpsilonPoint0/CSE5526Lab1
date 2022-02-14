import numpy as np
import matplotlib as plt
import matplotlib.pyplot as mp
class MLP:

    # Initialize data
    def __init__(self, layers, alpha=0, nu=0.0002):
        self.W1 = []
        self.W2 = []
        self.layers = layers
        self.nu = nu
        self.alpha = alpha

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
            #W1 = np.random.uniform(layers[i] +1, layers[i + 1] + 1)
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
        #print(xvals)
        #print(yvals)


    def train(self, X, Y, epochs=1000):
        update=100
        # Create biases and combine with data
        b1 = np.random.uniform(low=-0.1, high=0.1, size=200)
        #b2 = np.random.uniform(low=-0.1, high=0.1, size=200)
        #print(X)
        #print(b1)
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
                if epoch > 1000 and (loss_per_epoch[epoch + 1] - loss_per_epoch[epoch -99]) < 0.01:
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
                self.W1[layer] += -self.nu * output_activations[layer].T.dot(partial_deriv[layer])*self.alpha

    def predict(self, X, last_layer=True):

        p = np.atleast_2d(X)

        if last_layer:

            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W1)):

            p = self.sigmoid(np.dot(p, self.W1[layer]))

        return p

    def loss_function(self, X, targets):
        predictions = self.predict(X, last_layer=False)
        loss = self.nu * np.sum((predictions - targets) ** 2)
        # return the loss
        return loss

    def step_function(self, data):
        predictions = []
        for x in data:
            pred = self.predict(x)[0][0]
            predictions.append(1) if pred > 0.5 else predictions.append(0)
        return predictions

    def print_data(self):
        print(self.test2)
        print(self.target2nums)


MLP1 = MLP([2, 6, 1])
# MLP1.print_data
#print(MLP1.test1)
MLP1.plot_data()

min1, max1 = MLP1.test2[:, 0].min()-1, MLP1.test2[:, 0].max()+1
min2, max2 = MLP1.test2[:, 1].min()-1, MLP1.test2[:, 1].max()+1

x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

xx, yy = np.meshgrid(x1grid, x2grid)

r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

grid = np.hstack((r1,r2))
MLP1.train(MLP1.test1, MLP1.target1nums, epochs=10000)
#predictions = MLP1.step_function(MLP1.test2)
print(grid)
yhat = MLP1.step_function(MLP1.test2)

#zz = np.reshape(yhat, xx.shape)

#print(xx, yy, zz)
#mp.contourf(xx, yy, zz, cmap=plt.cm.Paired)

mp.show()

def MLP_4hu():
    MLP2 = MLP([2, 4, 1])
    #print(MLP2.test1)
    #print(MLP2.target1nums)
    #MLP2.plot_data()
    #print(MLP2.test1)
    MLP2.train(MLP2.test1, MLP2.target1nums, epochs=1000)
    MLP2.step_function(MLP2.test1, MLP2.target1nums)


def MLP_8hu():
    MLP3 = MLP([2, 8, 1])
    #print(MLP2.test1)
    #print(MLP2.target1nums)
    #MLP2.plot_data()
    #print(MLP2.test1)
    MLP3.train(MLP3.test1, MLP3.target1nums, epochs=1000)
    MLP3.step_function(MLP3.test1, MLP3.target1nums)


def MLP_12hu():
    MLP4 = MLP([2, 12, 1])
    #print(MLP2.test1)
    #print(MLP2.target1nums)
    #MLP2.plot_data()
    #print(MLP2.test1)
    MLP4.train(MLP4.test1, MLP4.target1nums, epochs=1000)
    MLP4.step_function(MLP4.test1, MLP4.target1nums)

def MLP_20hu():
    MLP5 = MLP([2, 20, 1])
    #print(MLP2.test1)
    #print(MLP2.target1nums)
    #MLP2.plot_data()
    #print(MLP2.test1)
    MLP5.train(MLP5.test1, MLP5.target1nums, epochs=1000)
    MLP5.step_function(MLP5.test1, MLP5.target1nums)


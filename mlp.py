import numpy as np
import matplotlib as plt
import matplotlib.pyplot as mp



class MLP:
    test1 = None
    test2 = None
    target1 = None
    target2 = None
    num_inputs = 2
    num_neurons = 6
    num_outputs = 1
    W1 = np.random.uniform(size=(num_inputs,num_neurons))
    W2 = np.random.uniform(size=(num_neurons,num_outputs))
    b1 = np.random.uniform(low=-0.1, high=0.1, size=(1,num_neurons))
    b2 = np.random.uniform(low=-0.1, high=0.1, size=(1,num_neurons))
    parameters = {"W1": W1, "W2": W2, "b1":b1, "b2": b2}

    mlp_types= [
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
    ]

    # Initialize data
    def __init__(self):
        self.test1 = np.random.uniform(low=-1, high=1, size=(200, 2))
        self.target1 = []
        for pt in self.test1:
            if np.abs(np.sin(np.pi*pt[0])) > np.abs(pt[1]):
                self.target1.append('blue')
            else:
                self.target1.append('red')
        test1 = self.test1
        target1 = self.target1

        self.test2 = np.random.uniform(low=-1, high=1, size=(200, 2))
        self.target2 = []
        for pt in self.test2:
            if np.abs(np.sin(np.pi*pt[0])) > np.abs(pt[1]):
                self.target2.append('blue')
            else:
                self.target2.append('red')
        test2 = self.test2
        target2 = self.target2

    def linear_dot_product(W, X, b):
        return (X@W) + b

    def sigmoid(Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_derivative(dA, Z):
        sig_result = MLP.sigmoid(Z)
        return dA * sig_result * (1 - sig_result)

    def relu(Z):
        return np.maximum(0,Z)

    def relu_derivative(dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0
        return dZ


    def cost_function(neurons_activated, expected_values):
        return (np.mean(np.power(neurons_activated - expected_values, 2)))/2

    def single_layer_fp(A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        
        if activation == "relu":
            activation_func = MLP.relu
        elif activation == "sigmoid":
            activation_func = MLP.sigmoid
        
        
        return activation_func(Z_curr), Z_curr

    def full_fp(X, params_values, mlp_types):
        memory = {}
        A_curr = X
        
        for idx, layer in enumerate(mlp_types):
            layer_idx = idx + 1
            A_prev = A_curr
            
            activ_function_curr = layer["activation"]
            W_curr = params_values["W" + str(layer_idx)]
            b_curr = params_values["b" + str(layer_idx)]
            A_curr, Z_curr = MLP.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
            
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        
        return A_curr, memory


    def plot_data(self):
        x_vals = []
        y_vals = []
        for pt in self.test1:
            x_vals.append(pt[0])
            y_vals.append(pt[1])

        self.x_vals = x_vals
        self.y_vals = y_vals

        for i in range(len(x_vals)):

            mp.scatter(x_vals[i], y_vals[i], c = self.target1[i])
        mp.show()
        #print(xvals)
        #print(yvals)

    def get_cost_value(Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    def get_accuracy_value(Y_hat, Y):
        Y_hat_ = MLP.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def convert_prob_into_class(Y_hat):
        #TODO
        x =0

    def single_layer_bp(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]
    
        if activation == "relu":
            ba_function = MLP.relu_derivative
        elif activation == "sigmoid":
            ba_function = MLP.sigmoid_derivative

    
        dZ_curr = ba_function(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_bp(Y_hat, Y, memory, params_values, mlp_types):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
    
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
        
        for layer_idx_prev, layer in reversed(list(enumerate(mlp_types))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            
            dA_curr = dA_prev
            
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]
            
            dA_prev, dW_curr, db_curr = MLP.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
            
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr
        
        return grads_values

    def update(params_values, grads_values, mlp_types, learning_rate):
        for layer_idx, layer in enumerate(mlp_types):
            params_values["W1" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
            params_values["b1" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        return params_values

    def train_network(X, Y, mlp_types, epochs, learning_rate):
        
        cost_history = []
        accuracy_history = []
        
        for i in range(epochs):
            Y_hat, cashe = MLP.full_fp(X, MLP.parameters, mlp_types)
            cost = MLP.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = MLP.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
            
            grads_values = MLP.full_bp(Y_hat, Y, cashe, params_values, mlp_types)
            params_values = MLP.update(params_values, grads_values, mlp_types, learning_rate)
            
        return params_values, cost_history, accuracy_history

    

MLP1 = MLP()
MLP1.plot_data()     
MLP1.train_network(MLP1.x_vals, MLP1.y_vals, MLP1.mlp_types, 100)
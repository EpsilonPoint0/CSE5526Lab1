import numpy as np
import matplotlib as plt
import matplotlib.pyplot as mp


def sigmoid(output):
	return output * (1.0 - output)
class MP:
    test1 = None
    test2 = None
    target1 = None
    target2 = None
    num_inputs = 2
    num_neurons = 6
    num_output = 1
    b1 = np.random.uniform(low=-0.1, high=0.1, size=(1,num_neurons))
    b2 = np.random.uniform(low=-0.1, high=0.1, size=(1,num_neurons))

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

    def plot_data(self):
        xvals = []
        yvals = []
        for pt in self.test1:
            xvals.append(pt[0])
            yvals.append(pt[1])

        for i in range(len(xvals)):

            mp.scatter(xvals[i], yvals[i], c = self.target1[i])
        mp.show()
        #print(xvals)
        #print(yvals)

MP1 = MP()
MP1.plot_data()      
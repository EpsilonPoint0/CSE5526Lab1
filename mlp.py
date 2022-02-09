import numpy as np
import matplotlib as plt
import matplotlib.pyplot as mp

class MultiLayerPerceptron:
    test1 = np.random.uniform(low=-1, high=1, size=(200, 2))
    target1 = []

    def initialize_data():
        
        for pt in self.test1:
            if np.abs(np.sin(np.pi*pt[0])) > np.abs(pt[1]):
                target1.append('blue')
            else:
                target1.append('red')
        xvals = []
        yvals = []
        for pt in test1:
            xvals.append(pt[0])
            yvals.append(pt[1])
        #print(xvals)
        #print(yvals)
        for i in range(len(xvals)):

            mp.scatter(xvals[i], yvals[i], c = target1[i])
        mp.show()
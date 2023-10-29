import pandas as pd
import numpy as np
import NN
import Particle

#Testing data and forward propagation
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

neural_network = NN.NeuralNetwork(2,1)

output = neural_network.forward_propagation(test_data)
print("output is:")
print(output)
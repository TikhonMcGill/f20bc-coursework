import pandas as pd
import numpy as np
import NN

dataset = pd.read_csv("data_banknote_authentication.txt") #Load the dataset

#Testing data and forward propagation
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

neural_network = NN.NeuralNetwork(2,1)

output = neural_network.forward_propagation2(test_data)
print("output is:")
print(output)
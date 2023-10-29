import numpy as np
import NN
import Particle

#Testing data and forward propagation
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

neural_network = NN.NeuralNetwork(2,1)

output = neural_network.forward_propagation(test_data)
print("output is:")
print(output)

#Make sure that we have weights and biases for each layer
assert len(neural_network.weights) == len(neural_network.biases)

#Make sure that, for each element of the test data, there is the corresponding output
assert len(output) == len(test_data)

#Testing particle vector and velocity size correctness
test_particle = Particle.Particle(10)

assert len(test_particle.vector) == 10
assert len(test_particle.velocity) == 10
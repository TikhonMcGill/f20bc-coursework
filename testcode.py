import numpy as np
import NN
import Particle
import ParticleConversion

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

#Testing Neural Network Particle size
neural_network_size = ParticleConversion.neural_network_particle_vector_size(neural_network.layer_sizes)

#The Neural Network base class uses profile2, which is 2 hidden layers of size 4 each.
#The input size is 2
#Therefore, from input to hidden layer 1, there are 2x4 = 8 connections, +4 biases (for the hidden layer)
#Then, from hidden layer 1 to hidden layer 2, there are 4x4 = 16 connections, +4 biases
#Finally, from hidden layer 2 to output, there are 4x1 = 4 connections, +1 bias
#This gives 8 + 4 + 16 + 4 + 4 + 1 = 37
#So, if this Test Neural Network is converted to a Particle, its vector size should be 37
assert neural_network_size == 37
import numpy as np
import NN
import Particle
import ParticleConversion
import hyperparameter_profile as profile

#Testing data and forward propagation
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

neural_network = NN.NeuralNetwork(*profile.profile2())

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
neural_network_size = ParticleConversion.get_particle_vector_size(neural_network.layer_sizes)

#The Neural Network base class uses profile2, which is 2 hidden layers of size 4 each.
#The input size is 2
#Therefore, from input to hidden layer 1, there are 2x4 = 8 connections, +4 biases (for the hidden layer)
#Then, from hidden layer 1 to hidden layer 2, there are 4x4 = 16 connections, +4 biases
#Finally, from hidden layer 2 to output, there are 4x1 = 4 connections, +1 bias
#This gives 8 + 4 + 16 + 4 + 4 + 1 = 37
#So, if this Test Neural Network is converted to a Particle, its vector size should be 37
assert neural_network_size == 37

#Furthermore, if this Test Neural Network is converted to a Particle, each layer would have the following sums of
#weights and connections: [12,20,5]
neural_network_layer_sizes = ParticleConversion.get_particle_layer_counts(neural_network.layer_sizes)

assert (neural_network_layer_sizes == [12,20,5]).all()

#Create a 37-size particle - the same size as the Test Neural Network we created
test_particle = Particle.Particle(37)

#Check that its elements are correctly-arranged in the "get_rough_layers" method
rough_layers = ParticleConversion.get_rough_layers(neural_network.layer_sizes,test_particle)

assert len(rough_layers[0]) == 12
assert len(rough_layers[1]) == 20
assert len(rough_layers[2]) == 5
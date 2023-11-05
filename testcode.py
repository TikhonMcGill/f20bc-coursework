import numpy as np
import NN
import Particle
import ParticleConversion
import hyperparameter_profiles as profiles

#Choose a profile for testing
test_profile = profiles.profile2

#Testing data and forward propagation
test_data = np.array([[0, 0, 1, 2], [0, 1, 1, 0], [1, 0, 5, 6], [1, 1, 3, 2]])

neural_network = NN.NeuralNetwork(test_profile.layer_sizes,test_profile.activation_functions)

output = neural_network.forward_propagation(test_data)
print("output is:")
print(output)

#Make sure that we have weights and biases for each layer
assert len(neural_network.weights) == len(neural_network.biases)

#Make sure that, for each element of the test data, there is the corresponding output
assert len(output) == len(test_data)

#Testing particle vector and velocity size correctness
test_particle = Particle.Particle(10)

assert len(test_particle.position) == 10
assert len(test_particle.velocity) == 10

#Testing Neural Network Particle size
neural_network_size = ParticleConversion.get_particle_vector_size(neural_network.layer_sizes)

#The Neural Network base class uses profile2, which is 2 hidden layers of size 4 each.
#The input size is 4
#Therefore, from input to hidden layer 1, there are 4x4 = 16 connections, +4 biases (for the hidden layer)
#Then, from hidden layer 1 to hidden layer 2, there are 4x4 = 16 connections, +4 biases
#Finally, from hidden layer 2 to output, there are 4x1 = 4 connections, +1 bias
#This gives 16 + 4 + 16 + 4 + 4 + 1 = 45
#So, if this Test Neural Network is converted to a Particle, its vector size should be 37
assert neural_network_size == 45

#Furthermore, if this Test Neural Network is converted to a Particle, each layer would have the following sums of
#weights and connections: [20,20,5]
neural_network_layer_sizes = ParticleConversion.get_particle_layer_counts(neural_network.layer_sizes)

assert (neural_network_layer_sizes == [20,20,5]).all()

#Create a 45-size particle - the same size as the Test Neural Network we created
test_particle = Particle.Particle(45)

#Check that its elements are correctly-arranged in the "get_layer_vectors" method
rough_layers = ParticleConversion.get_layer_vectors(neural_network.layer_sizes,test_particle)

assert len(rough_layers[0]) == 20
assert len(rough_layers[1]) == 20
assert len(rough_layers[2]) == 5

#Create a Neural Network based on the layer vectors, and see if it works by checking lengths of weights and biases
particle_neural_network = ParticleConversion.particle_to_neural_network(test_profile.layer_sizes,test_profile.activation_functions,test_particle)

assert len(particle_neural_network.biases[0]) == 4 #Make sure 4 biases from Input to Hidden Layer 1 (H.L. 1)
assert len(particle_neural_network.biases[1]) == 4 #Make sure 4 biases from Hidden Layer 1 to Hidden Layer 2 (H.L. 2)
assert len(particle_neural_network.biases[2]) == 1 #Make sure 1 bias from Hidden Layer 2 to Output

assert len(particle_neural_network.weights[0]) == 4 #Make sure first dimension of weight matrix 2 from Input to H.L. 1
assert len(particle_neural_network.weights[1]) == 4 #Make sure first dimension of weight matrix 4 from H.L. 1 to H.L. 2
assert len(particle_neural_network.weights[2]) == 4 #Make sure first dimension of weight matrix 4 from H.L. 2 to Output

assert len(particle_neural_network.weights[0][0]) == 4 #Make sure second dimension of weight matrix 4 from Input to H.L. 1
assert len(particle_neural_network.weights[1][0]) == 4 #Make sure second dimension of weight matrix 4 from H.L. 1 to H.L. 2
assert len(particle_neural_network.weights[2][0]) == 1 #Make sure second dimension of weight matrix 1 from H.L. 2 to Output
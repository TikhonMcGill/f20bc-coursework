import numpy as np
from NN import NeuralNetwork
from Particle import Particle

#Code that returns a list of numbers. Each number in this list represents the sum of the number of weights and biases
#for that layer in the Neural Network - this will be useful in converting Particles to Neural Networks
def get_particle_layer_counts(layer_sizes) -> np.array:
    #Initialize the empty array
    result = []

    for layer in range(len(layer_sizes)):
        #If layer is 0, it is the first layer, so ignore it
        if layer == 0:
            continue
        else:
            #Otherwise, get the size of the previous layer and multiply it by the size of this layer,
            #which gives the size of the Weight Matrix (since every neuron is connected to every other neuron)
            weight_size = layer_sizes[layer] * layer_sizes[layer-1]

            #The number of biases is simply the number of neurons in this layer
            bias_size = layer_sizes[layer]

            #Add the number of weights and biases to the total result
            result.append(weight_size + bias_size)
    
    return np.array(result)

#Code for getting the size of a Particle's Vector if a Neural Network is converted into a Particle
#This is for information reasons ()
def neural_network_particle_vector_size(layer_sizes) -> int:
    return np.sum(get_particle_layer_counts(layer_sizes))

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
def get_particle_vector_size(layer_sizes) -> int:
    return np.sum(get_particle_layer_counts(layer_sizes))

#Code for converting a Particle's Vector, a long list of numbers, into a list of smaller lists of numbers,
#each smaller list representing all the weights and biases for the resulting Neural Network from this particle
def get_rough_layers(layer_sizes, particle : Particle):
    #Get the number of elements in each layer
    layer_counts = get_particle_layer_counts(layer_sizes)

    resulting_layers = []

    #Copy the particle's vector, so that we can pop elements from it easily
    removable_array = particle.vector.copy()

    #Go through every layer in the layer count, adding elements corresponding to its
    #index to the resulting layers
    for layer in range(len(layer_counts)):
        new_rough_layer = []

        for element in range(layer_counts[layer]):
            #Add the first element of the removable array to this layer, then
            #remove it  from the list
            new_rough_layer.append(removable_array[0])
            removable_array = removable_array[1:]
        
        resulting_layers.append(new_rough_layer)
    
    return resulting_layers
import numpy as np
import hyperparameter_profile as hp

#Neural network class with adjustable number of hidden layers and number of neurons
class NeuralNetwork:
    #Only needed parameters to create the neural network are input size and number of neurons in the output layer
    #both of which will be dependent on the dataset and will be fixed based on that
    #The number of neurons in the hidden layers and the number of hidden layers as well as the activation function are all adjusted in the profile file
    def __init__(self, input_vector_size, output_layer):
        self.input_vector_size = input_vector_size
        self.hidden_layers, self.activation_function  = hp.profile1()
        self.output_size = output_layer
        
        # Initialize the weights and biases for the layers
        layer_sizes = [input_vector_size] + self.hidden_layers + [output_layer]
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

    def forward_propogation(self,input):
        # Plan for forward propagation:
        # 1. Organize inputs from previous layer (or just inputs) as a column vector
        # 2. Organize Weights in the previous layer as a matrix, where each row represents all the connections from
        # every previous neuron to one neuron
        # 3. Multiply the Input Vector by the Weight Matrix, getting a Vector as an output
        # 4. Organize the Biases as a column vector
        # 5. Add the Bias Vector to the Output Vector
        # 6. Apply the Activation function to each component of the Output vector
        return print("Not implemented yet")
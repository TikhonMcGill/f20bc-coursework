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

        #1. Store the sizes of each layer in an array
        layer_sizes = [input_vector_size] + self.hidden_layers + [output_layer]
        #2. Initialize random weights for each layer
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        #3. Initialize an array of biases of 0 for each layer (except for the output layer)
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

    def forward_propogation(self,input):
        #Ensure that the input given is of the same size as the input vector size
        if len(input) != self.input_vector_size:
            print ("Input given is not the size of input vector size!")
            return

        # 1. Organize inputs from previous layer (or just inputs) as a column vector
        input_vector = np.array(input)
        
        # 2. TODO Organize Weights between the previous layer's neurons and this layer's neurons as a matrix, each row 
        # representing all connections from every previous neuron to one neuron per row
        
        # 3. TODO Multiply the Input Vector by the Weight Matrix, getting a Vector as an output
        
        # 4. TODO Organize the Biases as a column vector
        
        # 5. TODO Add the Bias Vector to the Output Vector
        
        # 6. TODO Apply the Activation function to each component of the Output vector
        
        return print("Not implemented yet")
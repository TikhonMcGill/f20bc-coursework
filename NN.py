import numpy as np
import hyperparameter_profile as hp

##TODO - Being able to add different activation functions for each layer

#Neural network class with adjustable number of hidden layers and number of neurons
class NeuralNetwork:
    #Only needed parameters to create the neural network are input size and number of neurons in the output layer
    #both of which will be dependent on the dataset and will be fixed based on that
    #The number of neurons in the hidden layers and the number of hidden layers as well as the activation function are all adjusted in the profile file
    def __init__(self, input_vector_size, output_vector_size):
        self.input_vector_size = input_vector_size
        self.hidden_layers, self.activation_functions = hp.profile2()
        self.output_neurons = output_vector_size
        
        
        # Initialize the weights and biases for the layers

        #1. Store the sizes of each layer in an array
        self.layer_sizes = [self.input_vector_size] + self.hidden_layers + [self.output_neurons]
        #2. Initialize random weights for each layer
        self.weights = [np.random.rand(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes) - 1)]
        #3. Initialize an array of biases of 0 for each layer (except for the output layer)
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]

    #Forward Propagation Function
    def forward_propagation(self,input):
        #Initialize the previous output as the input initially, 
        #because that is what is being passed into the first hidden layer
        prev_output = input

        #Do the matrix calculation over every hidden layer
        for layer in range(len(self.hidden_layers)):
            #Multiply the weights and inputs(also outputs from previous layers) together
            matrix_product = np.dot(prev_output,self.weights[layer])
            #Add the Biases
            matrix_total = np.add(matrix_product,self.biases[layer])
            #Apply the Activation Function and set the Output of this hidden layer to replace the previous Output
            prev_output = self.activation_functions[layer](matrix_total)   
        
        #Now let's deal with the Final Output Layer
        #Same as before, except we use the [-1] notation to get the final weights listed in self.weights
        #(which will belong to the Output Layer)
        matrix_output_product = np.dot(prev_output, self.weights[-1])
        #Add the Biases again using the last indexed biases
        matrix_output_total = np.add(matrix_output_product, self.biases[-1])
        #Apply the activation function and set the Output
        output = self.activation_functions[-1](matrix_output_total)
        return output

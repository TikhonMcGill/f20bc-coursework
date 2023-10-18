import numpy as np
import hyperparameter_profile as hp

#Neural network class with adjustable number of hidden layers and number of neurons
class NeuralNetwork:
    #Only needed parameters to create the neural network are input size and number of neurons in the output layer
    #both of which will be dependent on the dataset and will be fixed based on that
    #The number of neurons in the hidden layers and the number of hidden layers as well as the activation function are all adjusted in the profile file
    def __init__(self, input_vector_size, output_vector_size):
        self.input_vector_size = input_vector_size
        self.hidden_layers, self.activation_function = hp.profile1()
        self.output_size = output_vector_size
        
        # Initialize the weights and biases for the layers

        #1. Store the sizes of each layer in an array
        self.layer_sizes = [self.input_vector_size] + self.hidden_layers + [self.output_size]
        #2. Initialize random weights for each layer
        self.weights = [np.random.rand(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes) - 1)]
        #3. Initialize an array of biases of 0 for each layer (except for the output layer)
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]

    def forward_propogation(self,input):
        #Ensure that the input given is of the same size as the input vector size
        if len(input) != self.input_vector_size:
            print ("Input given is not the size of input vector size!")
            return

        # Go through every layer, taking either the previous layer (or just input) as the input, and producing activations for
        # the next layer

        previous_output = [] #Initialize the "previous_layer" as an empty array, initially
        
        output_layer = len(self.layer_sizes) - 1 #Initialize an int for the output layer, for readability

        output = -1 #Initialize a variable for output, which is -1 initially

        for layer in range(len(self.layer_sizes)):
            # 1. Organize inputs from previous layer (or just inputs) as a column vector
            if layer == 0:
                previous_output = np.array(input)
                #If we're at the first layer, no activations or weights, just the input, so continue, but
                #set the "previous_layer" for the next layer to just be the input
                continue
            else:
                input_vector = np.array(previous_output)

            # 2. Organize Weights between the previous layer's neurons and this layer's neurons as a matrix, each row 
            # representing all connections from every previous neuron to one neuron per row
            weight_matrix = self.weights[layer-1] #Get the weight matrix corresponding to the layer before this neuron

            # 3. Multiply the Input Vector by the Weight Matrix, getting a Vector as an output
            # Each row in the matrix represents all connections to one neuron.
            # Therefore, we multiply each column in the neuron's row by the element in the input vector
            # Then, we add all of the products together - that is the total
            
            # Transpose the Weight Matrix, so that we can do matrix multiplication
            # (number of rows needs to equal size of output vector)
            np.transpose(weight_matrix)
            
            if layer == output_layer:
                #If we're at the output layer, the output is just the final product of weights and previous
                #neurons' activation functions, so we set the output and break out of the loop
                output = np.matmul(weight_matrix,input_vector)
                break
            else:
                #Otherwise, we maintain the output vector, for work with biases etc.
                output_vector = np.matmul(weight_matrix,input_vector)


            # 4. Organize the Biases as a column vector
            bias_vector = self.biases[layer]

            # 5. Add the Bias Vector to the Output Vector
            output_vector = np.add(output_vector,bias_vector)

            # 6. Apply the Activation function to each component of the Output vector
            activation_vector_func = np.vectorize(self.activation_function) #Create an array-based function using np.vectorize

            output_vector = activation_vector_func(output_vector) #Apply the activation function to each element of output vector
            
            # 7. Set the next layer's previous output to be this layer's output
            previous_output = output_vector 

        if output == -1:
            print("Warning! No output was calculated!")

        return output
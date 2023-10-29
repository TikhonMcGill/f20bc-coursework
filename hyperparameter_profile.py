import activation_functions as actfunc

# Profile class to easily adjust the number of hidden layers
# with their respective number of neurons and the activation function

# Generic function for creating a profile
def profile(_layer_sizes,_activation_functions):
    layer_sizes = _layer_sizes
    activation_function = _activation_functions
    return layer_sizes, activation_function

#Create profile with 2 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
def profile1():
    return profile([2,4,4,1],[actfunc.activation_relu, actfunc.activation_elu, actfunc.activation_leaky_relu])

#Create profile with 2 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
def profile2():
    return profile([2,4,4,1],[actfunc.activation_relu, actfunc.activation_relu, actfunc.activation_logistic])

#Create profile with 2 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
def profile3():
    return profile([2,4,4,1],[actfunc.activation_tanh, actfunc.activation_relu, actfunc.activation_elu])

#Create profile with 2 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
def profile4():
    return profile([2,4,4,1],[actfunc.activation_leaky_relu,actfunc.activation_logistic,actfunc.activation_tanh])

#Create profile with 2 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
def profile5():
    return profile([2,4,4,1],[actfunc.activation_elu,actfunc.activation_leaky_relu,actfunc.activation_logistic])
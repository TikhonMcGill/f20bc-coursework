import activation_functions as actfunc

# Profile class to easily adjust the number of hidden layers
# with their respective number of neurons and the activation function

# Generic function for creating a profile
def profile(_hidden_layers,_activation_function):
    hidden_layers = _hidden_layers
    activation_function = _activation_function
    return hidden_layers, activation_function

#Create profile with 2 hidden layers of 4 neurons each, utilizing the ReLU activation function
def profile1():
    return profile([4,4],actfunc.activation_relu)

#Create profile with 2 hidden layers of 4 neurons each, utilizing the logistic activation function
def profile2():
    return profile([4,4],actfunc.activation_logistic)

#Create profile with 2 hidden layers of 4 neurons each, utilizing the tanh activation function
def profile3():
    return profile([4,4],actfunc.activation_tanh)

#Create profile with 2 hidden layers of 4 neurons each, utilizing the leaky ReLU activation function
def profile4():
    return profile([4,4],actfunc.activation_leaky_relu)

#Create profile with 2 hidden layers of 4 neurons each, utilizing the eLU activation function
def profile5():
    return profile([4,4],actfunc.activation_elu)
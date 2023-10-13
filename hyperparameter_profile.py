import activation_functions as actfunc

#profile class to easily adjust the number of hidden layers with their respective number of neurons and the actiavtion function
def profile1():
    hidden_layers = [4,4]
    activation_function = actfunc.activation_relu
    return hidden_layers, activation_function


from HyperparameterProfile import HyperparameterProfile

import activation_functions as actfunc

#A Script containing various profiles to check out

#Profile with 4 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
profile1 = HyperparameterProfile([4,4,4,1],[actfunc.activation_relu, actfunc.activation_elu, actfunc.activation_leaky_relu],
                                 0.7,1.2,1.8,1.9,10000,100)

#Profile with 4 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
profile2 = HyperparameterProfile([4,4,4,1],[actfunc.activation_relu, actfunc.activation_relu, actfunc.activation_logistic],
                                 0.7,1.2,1.8,1.9,10000,100)

#Profile with 4 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
profile3 = HyperparameterProfile([4,4,4,1],[actfunc.activation_tanh, actfunc.activation_relu, actfunc.activation_elu],
                                 0.7,1.2,1.8,1.9,10000,100)

#Profile with 4 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
profile4 = HyperparameterProfile([4,4,4,1],[actfunc.activation_leaky_relu,actfunc.activation_logistic,actfunc.activation_tanh],
                                 0.7,1.2,1.8,1.9,10000,100)

#Profile with 4 input layers, 2 hidden layers of 4 neurons each, and 1 output layer, utilizing various activation functions
profile5 = HyperparameterProfile([4,4,4,1],[actfunc.activation_elu,actfunc.activation_leaky_relu,actfunc.activation_logistic],
                                 0.7,1.2,1.8,1.9,10000,100)
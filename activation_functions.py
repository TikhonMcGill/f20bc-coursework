import math
import numpy as n
import pandas as p

def activation_logistic(input):
    return (1 + (math.e ** -input)) ** -1

def activation_relu(input):
    return max(0,input)

def activation_tanh(input):
    return math.tanh(input)

def activation_leaky_relu(input):
    return max(0.03*input,input)

def activation_elu(input):
    if input > 0:
        return input
    else:
        return (n.exp(input)-1)
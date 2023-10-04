import math

def activation_logistic(input):
    return (1 + (math.e ** -input)) ** -1

def activation_relu(input):
    return max(0,input)

def activation_tanh(input):
    return math.tanh(input)
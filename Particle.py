import numpy as np

#Particle Class which holds a vector and a velocity
class Particle:
    #Create a new Particle with a vector of the vector size and a vector velocity of the vector size
    def __init__(self,vector_size : int):
        self.vector = np.zeros(vector_size)
        self.velocity = np.zeros(vector_size)
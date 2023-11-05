import numpy as np

#Particle Class which holds a vector and a velocity
class Particle:
    #Create a Particle with a vector of the vector size, and the velocity being initialized at 0
    def __init__(self,vector_size : int):
        self.vector = np.zeros(vector_size)
        self.velocity = np.zeros(vector_size)
    
    #Update the Particle's Vector based on its velocity
    def update_vector(self):
        self.vector = np.add(self.vector,self.velocity)
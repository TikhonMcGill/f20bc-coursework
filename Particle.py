import numpy as np

#Particle Class which holds a position and a velocity
class Particle:
    #Create a Particle with a position of the vector size, and the velocity being initialized at 0
    def __init__(self,vector_size : int):
        self.position = np.zeros(vector_size)
        self.velocity = np.zeros(vector_size)
    
    #Update the Particle's Vector based on its velocity
    def update_position(self):
        self.position = np.add(self.position,self.velocity)
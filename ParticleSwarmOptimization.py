import numpy as np

from Particle import Particle
import ParticleConversion as pc

from NN import NeuralNetwork

class ParticleSwarmOptimization:
    def pso(self,data,labels,iterations,nn_layer_sizes,nn_activation_functions,no_particles):
        #particle swarm optimization hyperparameters
        a = 0.7
        b = 1.2
        g = 1.8
        t = 1
        #not sure if the jump size is actually necessary so left it for now as it does not really make sense to me
        #jump_size = ?

        #Initialize the particles
        particles = []
        for p in range(no_particles):
            #Fill the particles array with particles, with random velocities and positions
            #encoding the weights and biases of the neural network
            
            #Get the size of the Particle's Vector, based on the layer sizes of the Neural Network given
            vector_size = pc.get_particle_layer_counts(nn_layer_sizes)

            #Create a new Particle with this vector size
            new_particle = Particle(vector_size)

            #Initialize the Particle's Position, with each dimension's value being between 0 and N, where N is the number
            #of particles
            new_particle.position = np.random.rand(vector_size) * no_particles

            #Initialize the Particle's Velocity, each dimension's value being between 0 and N/2, where N = no. particles
            new_particle.velocity = np.random.rand(vector_size) * no_particles/2

            #Add the New Particle to the Particles Array
            particles.append(new_particle)


        #initialize the global best
        best = 0
        #For the number of iterations do:

            #go through each particle and acess fitness
                #update the global best

            
            #gather information for every particle
                #for every particles velocity update each dimension
            
            #update each particles velocity

        #return the best
        return best

    def access_fitness(self,particle):
        pass

    def update_velocity(self,particle):
        pass
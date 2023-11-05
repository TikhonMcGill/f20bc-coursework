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
            
            #Get the size of the Particle's Vector, based on the layer sizes of the Neural Network given.
            #Also, keep this vector size, since it'll be useful in future calculations (e.g. velocity etc.)
            self.vector_size = pc.get_particle_layer_counts(nn_layer_sizes)

            #Create a new Particle with this vector size
            new_particle = Particle(self.vector_size)

            #Initialize the Particle's Position, with each dimension's value being between 0 and N, where N is the number
            #of particles
            new_particle.position = np.random.rand(self.vector_size) * no_particles

            #Initialize the Particle's Velocity, each dimension's value being between 0 and N/2, where N = no. particles
            new_particle.velocity = np.random.rand(self.vector_size) * no_particles/2

            #Add the New Particle to the Particles Array
            particles.append(new_particle)


        #Initialize the value of the global best, and its position
        global_best = 0
        global_best_position = np.zeros(self.vector_size)
        
        #For the number of iterations, go through each particle and:
        #1. Update its position based on current velocity
        #2. Evaluate the Fitness of its Position
        #3. If the Fitness at this position Exceeds the Particle's Personal best, update Particle's 
        #Personal Best to be this position
        #4. If the Fitness at this position Exceeds the Best of ALL particles, update the global best and
        #the global best position to be, respectively, this particle's fitness and this particle's position
        #5. Update the velocity of the Particle, based on inertia, the cognitive factor (its personal best),
        #and the social factor (pick N/10 informants, where N is no. particles, calculate average of their personal bests),
        #and the global best position

        for iteration in range(iterations):
            pass


        #Return the Global Best
        return global_best

    def access_fitness(self,particle):
        pass

    def update_velocity(self,particle):
        pass
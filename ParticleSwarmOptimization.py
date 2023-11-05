import Particle as p
import ParticleConversion as pc

class ParticleSwarmOptimization:
    def pso(self,data,labels,iterations,no_particles):
            #particle swarm optimization hyperparameters
            a = 0.7
            b = 1.2
            g = 1.8
            t = 1
            #not sure if the jump size is actually necessary so left it for now as it does not really make sense to me
            #jump_size = ?

            #initialize the particles
            particles = []
            for p in no_particles:
                #fill the particles array with particles using random velovities and encoding the weights and biases of the neural network
                break

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

    def update_position(self,particle):
        pass
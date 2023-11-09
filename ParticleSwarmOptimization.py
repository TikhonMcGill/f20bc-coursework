import numpy as np
import pandas as pd

from Particle import Particle
import ParticleConversion as pc

from HyperparameterProfile import HyperparameterProfile

from NN import NeuralNetwork

class ParticleSwarmOptimization:
    
    def __init__(self,profile : HyperparameterProfile):
        #Particle Swarm Optimization Hyperparameters
        self.a = profile.a#Inertial Factor
        self.b = profile.b#Cognitive Factor
        self.g = profile.g#Social Factor - based on best out of selected informants
        self.gl = profile.gl#Global Factor - based on the best position out of ALL particles
        
        self.iterations = profile.iterations

        no_particles = profile.no_particles

        #Get the size of each Particle's Vector, based on the layer sizes of the Neural Network given.
        #Also, keep this vector size, since it'll be useful in future calculations (e.g. velocity etc.)
        self.vector_size = pc.get_particle_vector_size(profile.layer_sizes)

        #Initialize the particles
        self.particles = []

        for p in range(no_particles):
            #Fill the particles array with particles, with random velocities and positions
            #encoding the weights and biases of the neural network

            #Create a new Particle with this vector size
            new_particle = Particle(self.vector_size)

            #Initialize the Particle's Position, with each dimension's value being between 0 and N, where N is the number
            #of particles

            new_particle.position = np.random.rand(self.vector_size) * no_particles

            #Initialize the Particle's Velocity, each dimension's value being between 0 and N/2, where N = no. particles
            new_particle.velocity = np.random.rand(self.vector_size) * no_particles/2

            #Add the New Particle to the Particles Array
            self.particles.append(new_particle)


        #Initialize the value of the global best, and its position
        self.global_best = 0
        self.global_best_position = np.zeros(self.vector_size)

    def pso(self):
        for iteration in range(self.iterations):
            for particle in self.particles:
                #Update the Particle's position based on current velocity
                particle.update_position()

                #Evaluate the Fitness of the Particle's Position
                fitness = self.access_fitness(particle)

                #If Fitness at this position Exceeds Particle's Personal best, update Particle's Personal Best to be here
                if fitness > particle.personal_best:
                    particle.personal_best = fitness
                    particle.personal_best_position = particle.position

                #If Fitness at this position Exceeds the Best of ALL particles, update the global best and
                #the global best position to be, respectively, this particle's fitness and this particle's position
                if fitness > self.global_best:
                    self.global_best = fitness
                    self.global_best_position = particle.position
                
                #Update the velocity of the Particle, based on inertia, cognitive factor (its personal best),
                #social factor (average of personal bests of randomly-chosen informants), and global best position
                self.update_velocity(particle)

        #Return the Global Best
        return self.global_best

    def access_fitness(self,data,labels,profile,particle : Particle) -> float:
        fitness = 0
        threshold = 0.5

        #Convert the Particle into a Neural Network
        nn = pc.particle_to_neural_network(profile.layer_sizes,profile.activation_functions,particle)
        #run through the data and get the output
        #print("data is:")
        #print(data.head())
        data = np.array(data)
        #print("data is:")
        #print(data)
        out = []
        for point in data:
            #print("point is:")
            #print(point)
            out.append(nn.forward_propagation(point))
            #print("out is:")
            #print(out)
        print("out is:")
        print(out)
        #turn the output into a dataframe and join the labels, for easier comparison
        results = pd.DataFrame(np.around(out, decimals=3)).join(labels)
        #rename all collums for easier debugging/calculation later
        results.rename(columns={results.columns[1]: "labels"}, inplace=True)
        #print("results are:")
        #print(results)
        #replace the output with 1 if it is above the threshold, 0 otherwise
        results["predicted"] = results[0].apply(lambda x: 1 if x > threshold else 0)
        #print(results)
        #compare the predicted output with the actual output and get the accuracy, which will be used as the fitness
        fitness = results.loc[results['predicted']==results['labels']].shape[0] / results.shape[0] * 100
        return fitness
    
    #Code to pick N/10 informants, where N is the number of particles
    def get_informants(self,particle : Particle):

        no_informants = max(1,len(self.particles)/10)

        #Make sure this particle cannot be picked as an informant
        possible_choices = self.particles.copy()
        possible_choices.remove(particle) 

        return np.random.choice(possible_choices,no_informants,replace=False)

    def update_velocity(self,particle : Particle):
        inertial_weight = self.a
        cognitive_weight = self.b
        social_weight = self.g
        global_weight = self.gl

        #First, update the velocity based on the inertial weight - this is non-random and applies to all dimensions,
        #so is a matter of just adding a multiplied vector to the velocity vector
        particle.velocity = np.add(particle.velocity,particle.velocity * inertial_weight)

        #Get the informants of this particle, and the best position among them
        informants = self.get_informants(particle)
        informant_best = 0
        informant_best_position = np.zeros(self.vector_size)

        for i in informants:
            if i.personal_best > informant_best:
                informant_best = i.personal_best
                informant_best_position = i.position

        #Next, we'll have to update other weights separately for each dimension
        for dimension in range(len(particle.velocity)):
            #Get the cognitive part - difference between current position and personal best, multiplied by random
            #value between 0 and cognitive weight
            cognitive_part = cognitive_weight * np.random.rand() * (particle.position[dimension] - particle.personal_best_position[dimension])

            #Get the social part - difference between current position and best of the informants, multiplied by random value
            #between 0 and social weight
            social_part = social_weight * np.random.rand() * (particle.position[dimension] - informant_best_position[dimension])

            #Get the global part - difference between current position and global best position, multipled by random value between
            #0 and global weight
            global_part = global_weight * np.random.rand() * (particle.position[dimension] - self.global_best_position[dimension])

            #Finally, add the three parts together, and add them to this dimension of the velocity
            total_change = cognitive_part + social_part + global_part

            particle.velocity[dimension] += total_change
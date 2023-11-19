import pandas as pd
import numpy as np
import NN
import ParticleSwarmOptimization as pso
import data_preparation
import hyperparameter_profiles as hp

dataset = pd.read_csv("data_banknote_authentication.txt") #Load the dataset

train_data, test_data, train_labels, test_labels = data_preparation.prepare_data(0.3) #Use a 30% test size

#Choose a profile here
profile = hp.profile2

#Initialize a Particle Swarm Optimizer on the Training Data
pso = pso.ParticleSwarmOptimization(profile,train_data,train_labels)

#Carry out the PSO
pso.pso()
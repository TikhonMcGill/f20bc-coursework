class HyperparameterProfile:
    def __init__(self,network_layer_sizes,network_layer_activations,
                 pso_inertial_weight,pso_cognitive_weight,pso_social_weight,pso_global_weight,pso_iterations):
        
        self.layer_sizes = network_layer_sizes
        self.activation_functions = network_layer_activations
        
        self.a = pso_inertial_weight
        self.b = pso_cognitive_weight
        self.g = pso_social_weight
        self.gl = pso_global_weight
        self.iterations = pso_iterations
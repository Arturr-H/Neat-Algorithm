use crate::neural_network::average::FitnessAverage;

use super::mutation::{GenomeMutationProbablities, WeightChangeProbablities};

#[derive(Clone)]
pub struct NetworkConfig {
    pub mutation_probabilities: GenomeMutationProbablities,
    pub weight_change_probabilities: WeightChangeProbablities,

    /// If we should initialize networks with pre-
    /// established connections between input and
    /// output neurons. (Bias nodes not included)
    pub initialize_with_connections: bool,

    /// How many hidden neurons we start with for
    /// "boosting" the early stages of evolution.
    /// (neuron creation probabilities are often
    /// very low)
    pub initial_hidden_neurons: usize,

    /// What method we use to calculate average
    /// fitness for a network
    pub fitness_averaging_method: FitnessAverage,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            mutation_probabilities: Default::default(),
            weight_change_probabilities: Default::default(),
            initialize_with_connections: true,
            initial_hidden_neurons: 0,
            fitness_averaging_method: FitnessAverage::Regular
        }
    }
}

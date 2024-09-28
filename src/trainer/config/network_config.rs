use super::mutation::{GenomeMutationProbablities, WeightChangeProbablities};

#[derive(Clone)]
pub struct NetworkConfig {
    pub mutation_probabilities: GenomeMutationProbablities,
    pub weight_change_probabilities: WeightChangeProbablities,

    /// If we should initialize networks with pre-
    /// established connections between input and
    /// output neurons. (Bias nodes not included)
    pub initialize_with_connections: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            mutation_probabilities: Default::default(),
            weight_change_probabilities: Default::default(),
            initialize_with_connections: true
        }
    }
}

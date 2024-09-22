use super::mutation::{GenomeMutationProbablities, WeightChangeProbablities};

#[derive(Default, Clone)]
pub struct NetworkConfig {
    pub mutation_probabilities: GenomeMutationProbablities,
    pub weight_change_probabilities: WeightChangeProbablities,
}


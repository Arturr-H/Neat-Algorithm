use super::{mutation::{GenomeMutationProbablities, WeightChangeProbablities}, stop_condition::StopCondition};

#[derive(Default, Clone)]
pub struct NetworkConfig {
    pub mutation_probabilities: GenomeMutationProbablities,
    pub weight_change_probabilities: WeightChangeProbablities,
}


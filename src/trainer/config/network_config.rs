use super::{mutation::MutationProbablities, stop_condition::StopCondition};

#[derive(Default, Clone)]
pub struct NetworkConfig {
    pub mutation_probabilities: MutationProbablities,
}


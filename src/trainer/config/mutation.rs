
#[derive(Clone, Copy)]
pub struct GenomeMutationProbablities {
    pub split_connection: usize,
    pub create_connection: usize,
    pub change_weight: usize,
    pub toggle_weight: usize,
    pub nothing: usize,
}

impl Default for GenomeMutationProbablities {
    fn default() -> Self {
        Self {
            split_connection: 5,
            create_connection: 8,
            change_weight: 100,
            toggle_weight: 2,
            nothing: 20
        }
    }
}

#[derive(Clone, Copy)]
pub struct WeightChangeProbablities {
    pub addition_small: usize,
    pub addition_large: usize,
    pub multiplication_small: usize,
    pub multiplication_large: usize,
    pub change_sign: usize,
}

impl Default for WeightChangeProbablities {
    fn default() -> Self {
        Self {
            addition_small: 20,
            addition_large: 5,
            multiplication_small: 20,
            multiplication_large: 5,
            change_sign: 5,
        }
    }
}

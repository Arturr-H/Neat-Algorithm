
#[derive(Clone, Copy)]
pub struct MutationProbablities {
    pub split_connection: usize,
    pub create_connection: usize,
    pub change_weight: usize,
    pub toggle_weight: usize,
    pub nothing: usize,
}

impl Default for MutationProbablities {
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

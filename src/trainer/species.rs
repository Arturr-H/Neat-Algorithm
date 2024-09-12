use crate::neural_network::network::{self, NeatNetwork};

const SPEICES_NETWORK_SIZE: usize = 10;

/// How many networks we eliminate after 
/// each evaluation (worst performing)
const SPEICES_REMOVAL_COUNT: usize = 5;

pub struct Species {
    /// The representative of a species is a network just like the other
    /// networks in the species, but when we compare distance between
    /// nets, we only compare any net to the representative, not to another
    /// net => too much computation. The first element of this list is 
    /// the representative (index 0)
    /// 
    /// All networks who are initially cloned from the representative
    networks: Vec<NeatNetwork>,

}

impl Species {
    pub fn new(representative: NeatNetwork) -> Self {
        let mut networks: Vec<NeatNetwork> = Vec::with_capacity(SPEICES_NETWORK_SIZE);
        
        networks.push(representative.clone());
        for i in 0..SPEICES_NETWORK_SIZE {
            networks.push(representative.clone());
        }

        Self { networks }
    }

    /// Removes `SPEICES_REMOVAL_COUNT` amount of the worst
    /// performing networks in the current species
    pub fn eliminate(&mut self, fitness_function: fn(&mut NeatNetwork) -> f32) -> () {
        let mut scores = Vec::new();
        for net in self.networks.iter_mut() {
            let points = fitness_function(net);
            scores.push(points);
        }

        let worst_performing_indexes = Self::bottom_n_with_indices(&scores, SPEICES_REMOVAL_COUNT);
        let best_performing_index = Self::top_n_with_indices(&scores, 1)[0];
        
        for index in 0..self.networks.len() {
            if worst_performing_indexes.contains(&index) {
                dbg!("elim", index);
                self.networks[index] = self.networks[best_performing_index].clone();

                // Mutate once to add a bit of variation
                self.networks[index].mutate();
            }
        }
    }

    fn bottom_n_with_indices(numbers: &Vec<f32>, n: usize) -> Vec<usize> {
        let mut indexed_numbers: Vec<(usize, &f32)> = numbers.iter().enumerate().collect();
        indexed_numbers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed_numbers.into_iter().take(n).map(|(i, _)| i).collect()
    }
    fn top_n_with_indices(numbers: &Vec<f32>, n: usize) -> Vec<usize> {
        let mut indexed_numbers: Vec<(usize, &f32)> = numbers.iter().enumerate().collect();
        indexed_numbers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_numbers.into_iter().take(n).map(|(i, _)| i).collect()
    }
}

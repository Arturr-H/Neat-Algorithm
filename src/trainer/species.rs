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

    previous_aveage_score: f32,
}

impl Species {
    /// Creates a new species with a representative cloned to
    /// have `size` amount of identical networks that slightly
    /// mutate away from the representative
    pub fn new(representative: NeatNetwork, size: usize) -> Self {
        assert!(size > 0, "Size must be at least 1 to fit representative");
        let mut networks: Vec<NeatNetwork> = Vec::with_capacity(SPEICES_NETWORK_SIZE);
        
        networks.push(representative.clone());
        for i in 0..size - 1 {
            let mut net = representative.clone();
            net.mutate();
            networks.push(net);
        }

        Self { networks, previous_aveage_score: 0. }
    }

    /// Makes every net go trough a fitness function and determines the top 
    /// 30% of all nets. These nets automatically go to the next generation
    /// without changes. The 70% of the rest networks are randomly mutated
    /// and THEN placed in the next generation
    pub fn compute_generation(&mut self, fitness_function: fn(&mut NeatNetwork) -> f32) -> () {
        let mut scores = Vec::new();
        let mut total_score = 0.0;
        for net in self.networks.iter_mut() {
            let points = fitness_function(net);
            total_score += points;
            scores.push(points);
        }
        self.previous_aveage_score = total_score / self.networks.len() as f32;

        // We won't modify the top 30, that's why we only deal with bottom 70 here
        let bottom_70_amount = (self.networks.len() as f32 * 0.7).round() as usize;
        let bottom_70 = Self::bottom_n_with_indices(&scores, bottom_70_amount);

        for bottom_network_idx in bottom_70 {
            let bottom_net = &mut self.networks[bottom_network_idx];
            bottom_net.mutate();

            println!("genome size {}", bottom_net.get_genes().len());
        }
    }

    /// Get the average score that the networks performed
    /// during the last fitness test
    pub fn previous_aveage_score(&self) -> f32 {
        self.previous_aveage_score
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

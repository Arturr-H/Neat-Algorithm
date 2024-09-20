use std::sync::Arc;

use rand::{rngs::ThreadRng, thread_rng, Rng};
use serde_derive::{Serialize, Deserialize};

use crate::trainer::config::mutation::WeightChangeProbablities;

/// A connection between two `NodeGenes`
#[derive(Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    /// The index / ID of some node
    node_in: usize,
    node_out: usize,

    /// The strength of the bond between the 
    /// two nodes.
    weight: f32,

    /// Used for disabling some connections
    /// and turning them back on later in the evo
    enabled: bool,

    /// The local innovation number
    innovation_number: usize,
}

impl ConnectionGene {
    pub fn new(node_in: usize, node_out: usize, weight: f32, innovation_number: usize) -> Self {
        Self { node_in, node_out, weight, enabled: true, innovation_number }
    }
    
    // Getters
    pub fn node_in(&self) -> usize { self.node_in }
    pub fn node_out(&self) -> usize { self.node_out }
    pub fn weight(&self) -> f32 { self.weight }
    pub fn enabled(&self) -> bool { self.enabled }
    pub fn innovation_number(&self) -> usize { self.innovation_number }

    // Setters
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled
    }

    pub fn mutate_weight(&mut self, weight_change_prob: &WeightChangeProbablities) -> () {
        let mut rng = thread_rng();
        let mut new_weight = self.weight;
        let WeightChangeProbablities {
            addition_small, addition_large, multiplication_small,
            multiplication_large, change_sign } = weight_change_prob;

        let probabilities: Vec<(&usize, fn(&mut f32, ThreadRng))> = vec![
            (addition_small, |i: &mut f32, mut rng: ThreadRng|       { *i += rng.gen_range(-0.2..0.2) }),
            (addition_large, |i: &mut f32, mut rng: ThreadRng|       { *i += rng.gen_range(-1.5..1.5) }),
            (multiplication_small, |i: &mut f32, mut rng: ThreadRng| { *i *= rng.gen_range(0.8..1.2) }),
            (multiplication_large, |i: &mut f32, mut rng: ThreadRng| { *i *= rng.gen_range(0.3..1.7) }),
            (change_sign, |i: &mut f32, mut rng: ThreadRng|          { *i *= -1. }),
        ];

        let total: usize = probabilities.iter().map(|e| e.0).sum();
        let random_number = rng.gen_range(0..total);
        let mut cumulative = 0;
        for (index, &(probability, func)) in probabilities.iter().enumerate() {
            cumulative += probability;
            if random_number < cumulative {
                (func)(&mut self.weight, rng);
                //debug
                break;
            }
        }
    }
}

impl std::fmt::Debug for ConnectionGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.enabled {
            // write!(f, "Conn({}-{} ({:.2} ~ {}))", self.node_in, self.node_out, self.weight, self.innovation_number)
            write!(f, "Conn({}-{})", self.node_in, self.node_out)
        }else {
            write!(f, "----({}-{})", self.node_in, self.node_out)
        }
    }
}

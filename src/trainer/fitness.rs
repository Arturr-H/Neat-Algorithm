/* Imports */
use super::super::neural_network::network::NeatNetwork;

/// Any struct passed as a fitness evaluator into the
/// `Evolution` struct needs to implement this trait.
pub trait FitnessEvaluator: Clone {
    /// This function will evaluate a single network, and return
    /// a fitness score which needs to be >= 0.0.
    fn run(&mut self, network: &mut NeatNetwork) -> f32;
}

/* Imports */
use super::super::neural_network::network::NeatNetwork;

/// Any struct passed as a fitness evaluator into the
/// `Evolution` struct needs to implement this trait.
pub trait FitnessEvaluator: Clone {
    /// This function will evaluate a single network, and return
    /// a fitness score which needs to be >= 0.0.
    fn run(&mut self, network: &mut NeatNetwork) -> f32;

    /// This can be used for debugging. This will run the network
    /// through testing but also display the evaluation process
    /// in some way, e.g to stdout, or displaying a window.
    fn run_visualize(&mut self, _: &mut NeatNetwork) -> () {
        // Default behaviour so users who don't care about visualizing
        // won't need to implement this function.
    }
}

impl<T> FitnessEvaluator for T
    where T: Fn(&mut NeatNetwork) -> f32 + Clone {

    fn run(&mut self, network: &mut NeatNetwork) -> f32 {
        self(network)
    }
}
impl FitnessEvaluator for f32 {
    fn run(&mut self, _: &mut NeatNetwork) -> f32 { *self }
}

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

/* Imports */
use std::{collections::HashMap, os::unix::net, sync::{Arc, Mutex}, time::Duration};
use neat_algorithm::{debug::display::start_debug_display, games::snake::SnakeGameEvaluator, neural_network::activation::Activation, trainer::{config::mutation::{GenomeMutationProbablities, WeightChangeProbablities}, evolution::Evolution}};
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use rand::{thread_rng, Rng};

fn main() -> () {
    let mut evolution = Evolution::new()
        .batch_size(25)
        .with_species_size(6)
        .with_input_nodes(8)
        .with_output_nodes(1)
        .mutation_probabilities(GenomeMutationProbablities {
            split_connection: 2,
            create_connection: 5,
            change_weight: 350,
            toggle_weight: 1,
            nothing: 20,
        })
        .weight_change_probabilities(WeightChangeProbablities {
            addition_small: 25,
            addition_large: 5,
            multiplication_small: 25,
            multiplication_large: 5,
            change_sign: 3
        })
        .replace_worst_every_nth_gen(Some(1000))
        .with_hidden_activation(Activation::LeakyRelu)
        .with_output_activation(Activation::LeakyRelu)
        .set_fitness_evaluator(SnakeGameEvaluator)
        .build();
        
    
    start_debug_display(evolution);
}

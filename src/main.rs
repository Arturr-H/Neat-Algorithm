#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

/* Modules */
mod neural_network;
mod trainer;
mod utils;
mod debug;
mod games;

/* Imports */
use std::{collections::HashMap, sync::{Arc, Mutex}, time::Duration};
use debug::display::start_debug_display;
use games::snake::{Direction, SnakeGame};
use neural_network::{activation::{Activation, NetworkActivations}, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}};
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use trainer::{config::{mutation::{GenomeMutationProbablities, WeightChangeProbablities}, stop_condition::{StopCondition, StopConditionType}}, evolution::{self, Evolution, EvolutionBuilder}};
use trainer::species::Species;
use rand::{thread_rng, Rng};
use utils::{find_max_index, Timer};

fn main() -> () {
    let mut _evolution = Evolution::new()
        .batch_size(25)
        .with_species_size(6)
        .with_input_nodes(8)
        .with_output_nodes(4)
        .mutation_probabilities(GenomeMutationProbablities {
            split_connection: 2,
            create_connection: 5,
            change_weight: 750,
            toggle_weight: 1,
            nothing: 40,
        })
        .weight_change_probabilities(WeightChangeProbablities {
            addition_small: 25,
            addition_large: 5,
            multiplication_small: 25,
            multiplication_large: 5,
            change_sign: 3
        })
        .replace_worst_every_nth_gen(Some(600))
        // .par_chunks_size(3)
        .with_hidden_activation(Activation::LeakyRelu)
        .with_output_activation(Activation::Sigmoid)
        .with_stop_condition(StopCondition::after(StopConditionType::FitnessReached(1000.)))
        .set_fitness_function(|e| SnakeGame::score_game(e, 800, false))
        .build();

    
    // let mut network = NeatNetwork::retrieve("D:\\Programmering\\hej");
    // SnakeGame::score_game(&mut network, 10000, false);
    
    start_debug_display(_evolution);
}


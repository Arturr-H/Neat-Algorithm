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
use utils::find_max_index;

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
        .replace_worst_every_nth_gen(Some(1000))
        // .par_chunks_size(3)
        .with_hidden_activation(Activation::LeakyRelu)
        .with_output_activation(Activation::Sigmoid)
        .with_stop_condition(StopCondition::after(StopConditionType::FitnessReached(1000.)))
        .set_fitness_function(|e| snake_game(e, false))
        .build();
    
    
    // let mut net = NeatNetwork::retrieve("/Users/artur/Desktop/snakemaster3");
    // snake_game(&mut net, true);

    start_debug_display(_evolution);
}

pub fn snake_game(network: &mut NeatNetwork, log: bool) -> f32 {
    let mut moves = 0;
    let mut game = SnakeGame::new();

    // Simulate the game loop (for testing)
    while !game.is_game_over {
        game.update();
        if log {
            game.display();
            std::thread::sleep(Duration::from_millis(10));
        }
        
        let apple_pos = vec![game.apple.x as f32 / 7., game.apple.y as f32 / 7.];
        let snake_head_pos = vec![game.snake.first().unwrap().x as f32 / 7., game.snake.first().unwrap().y as f32 / 7.];
        let proximity = game.get_proximity().to_vec();
        let input = Vec::new().into_iter()
            .chain(snake_head_pos)
            .chain(apple_pos)
            .chain(proximity)
            .collect();

        let decision = find_max_index(&network.calculate_output(input)) as u8;
        /* I am lazy */
        game.set_direction(unsafe {
            std::mem::transmute::<u8, Direction>(decision)
        });

        moves += 1;
        if moves > 10000 {
            break;
        }
    }

    game.score
}

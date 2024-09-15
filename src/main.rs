#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

/* Modules */
mod neural_network;
mod trainer;
mod utils;
mod debug;

/* Imports */
use std::{collections::HashMap, sync::{Arc, Mutex}, time::Duration};
use debug::display::start_debug_display;
use neural_network::{activation::{Activation, NetworkActivations}, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}};
use block_blast::board::{self, board::Board, board_error::PlacementError, cell::Cell};
use trainer::evolution::{self, Evolution, EvolutionBuilder};
use trainer::species::Species;
use rand::Rng;

fn main() -> () {
    // .with_input_nodes(64 /* All cells */ + 36 /* Tiles to choose from */)
    // .with_output_nodes(8/*x*/ + 8/*y - Coordinate for tile placement */ + 3 /*What tile buffer to choose */)
    let mut _evolution = Evolution::new()
        .batch_size(1)
        .with_input_nodes(2)
        .with_output_nodes(1)
        .with_hidden_activation(Activation::Relu)
        .with_output_activation(Activation::Relu)
        .with_species_size(6)
        .set_fitness_function(score_network)
        .build();

    _evolution.run(15);

    // let mut net = NeatNetwork::new(2, 3, Arc::new(Mutex::new(0)), Arc::new(Mutex::new(HashMap::new())), NetworkActivations::new(Activation::Relu, Activation::Relu));
    // println!("{net:?}");
    // net.mutate();
    // println!("{net:?}");
}

fn score_network(network: &mut NeatNetwork) -> f32 {
    let xor = vec![((0.0, 0.0), 0.0),
                    ((1.0, 0.0), 1.0),
                    ((0.0, 1.0), 1.0),
                    ((1.0, 1.0), 0.0)];

    let mut total_score = 0.0;
    for &((input1, input2), expected_output) in xor.iter() {
        let output = network.calculate_output(vec![input1, input2])[0];
        let err = (expected_output - output).abs();

        /* 1. - err becasue sigmoid max is = 1 */
        total_score += (1. - err).max(0.0);
    }

    total_score
}


/*
fn score_network(network: &mut NeatNetwork) -> f32 {
    let mut game = Board::new();
    let mut score = 0.0;

    loop {
        let mut cells: Vec<f32> = game.cells().iter().map(Cell::to_f32).collect();
        let mut tiles = vec![0.0; 36];
        for tile in game.get_tile_buffer() {
            if let Some(tile) = tile {
                let index_of_tile = tile as u8 as usize;
                tiles[index_of_tile] = 1.;
            }
        }
        cells.append(&mut tiles);

        let input = cells;
        let decision = network.calculate_output(input);

        let x = utils::get_max_index_list(&decision[0..8]);
        let y = utils::get_max_index_list(&decision[8..16]);
        let tile_buffer_index = utils::get_max_index_list(&decision[16..19]);
        let mut attempt = 0;

        let mut success = false;
        while !success {
            game.set_cursor(tile_buffer_index[attempt % 3]);
            game.set_cursor_position((x[attempt], y[attempt]).into());

            match game.place() {
                Ok(e) => {
                    score = e as f32;
                    success = true;
                },
                Err(error) => {
                    match error {
                        PlacementError::OutOfBounds | PlacementError::InvalidPlacement => {
                            if attempt >= 7 {
                                return score
                            }
                            attempt += 1;
                            continue;
                        },
                        PlacementError::GameOver => {
                            return score
                        }
                    }
                }
            };
        }

        // game.log();
    }
}
*/

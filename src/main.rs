#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

/* Modules */
mod neural_network;
mod trainer;
mod utils;

/* Imports */
use std::time::Duration;
use neural_network::network::NeatNetwork;
use block_blast::board::{self, board::Board, board_error::PlacementError, cell::Cell};
use trainer::evolution::{Evolution, EvolutionBuilder};

fn main() -> () {
    let mut evolution = Evolution::new()
        .batch_size(100)
        .with_input_nodes(64 /* All cells */ + 36 /* Tiles to choose from */)
        .with_output_nodes(8/*x*/ + 8/*y - Coordinate for tile placement */ + 3 /* What tile buffer to choose */)
        .set_fitness_function(score_network)
        .build()
        .generation();

    let net = NeatNetwork::new(5, 7);
    net.create_python_debug_plotting();

    let xor = vec![((0.0, 0.0), 0.0),
                    ((1.0, 0.0), 1.0),
                    ((0.0, 1.0), 1.0),
                    ((1.0, 1.0), 0.0)];

    // let mut all_values = Vec::new();
    // for net in networks.iter_mut() {
    //     let mut total = 0.0;
    //     for input in xor.iter() {
    //         let err = 1. - net.calculate_output(vec![input.0.0, input.0.1])[0];
    //         total += err;
    //     }

    //     all_values.push(total / 4.);
    // }

    // let top_5 = top_n_with_indices(&all_values, 5);
}

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


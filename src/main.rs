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
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use trainer::{evolution::{self, Evolution, EvolutionBuilder}, evolution_config::{Chain, StopCondition, StopConditionType}};
use trainer::species::Species;
use rand::{thread_rng, Rng};
use utils::{find_max_index, Timer};

fn main() -> () {
    let mut _evolution = Evolution::new()
        .batch_size(50)
        .with_species_size(6)
        .with_input_nodes(6)
        .with_output_nodes(4)
        .with_hidden_activation(Activation::LeakyRelu)
        .with_output_activation(Activation::Sigmoid)
        .with_stop_condition(StopCondition::after_fitness_reached(1000.))
        .set_fitness_function(snake)
        .build();
    
    // let mut net = NeatNetwork::new(64 + 16, 4, Arc::default(), Arc::default(), NetworkActivations::new(Activation::LeakyRelu, Activation::Sigmoid));
    // dbg!(snake(&mut net));
    // score_network(&mut net);

    // let mut net = NeatNetwork::retrieve("/Users/artur/Desktop/yyy");
    // snake(&mut net);

    // dbg!(net.calculate_output(vec![1.0, 0.0, 0.0, 0.0]));
    // dbg!(net.calculate_output(vec![1.0, 0.0]));
    // dbg!(net.calculate_output(vec![0.0, 1.0]));
    // dbg!(net.calculate_output(vec![0.0, 0.0]));
    start_debug_display(_evolution);
}

fn snake(network: &mut NeatNetwork) -> f32 {
    let mut board = vec![vec![0usize; 8]; 8];
    let mut rng = thread_rng();
    let mut score = 0.0f32;
    let mut snake_pos: (usize, usize) = (4, 4);
    let mut apple_pos: (usize, usize) = (4, 4);
    let mut attempts = 0;

    let clamp = |num: usize| -> Option<usize> {
        if num > 7 { None } else { Some(num) }
    };

    let mut spawn_apple = |b: &mut Vec<Vec<usize>>, snake_pos: &(usize, usize), apple_pos: &mut (usize, usize)| {
        /* Spawn apple */
        let mut y = rng.gen_range(0..8);
        let mut x = rng.gen_range(0..8);
        while y == snake_pos.1 && x == snake_pos.0 {
            y = rng.gen_range(0..8);
            x = rng.gen_range(0..8);
        }
        b[y][x] = 1;
        *apple_pos = (x, y);
    };

    spawn_apple(&mut board, &snake_pos, &mut apple_pos);

    loop {
        let mut x = vec![0.0; 3];
        let mut y = vec![0.0; 3];

        /* x */
        match snake_pos {
            pos if pos.0 > apple_pos.0 => { x[2] = 1. },
            pos if pos.0 == apple_pos.0 => { x[1] = 1. },
            pos if pos.0 < apple_pos.0 => { x[0] = 1. },
            _ => ()
        };
        /* y */
        match snake_pos {
            pos if pos.1 > apple_pos.1 => { y[2] = 1. },
            pos if pos.1 == apple_pos.1 => { y[1] = 1. },
            pos if pos.1 < apple_pos.1 => { y[0] = 1. },
            _ => ()
        };

        let input = x.into_iter().chain(y).collect();
        let decision = network.calculate_output(input);
        let snake_dir = find_max_index(&decision);  // 0 = top, 1 = right ...

        let new_pos = match snake_dir {
            0 => (Some(snake_pos.0), snake_pos.1.checked_sub(1)),
            1 => (clamp(snake_pos.0 + 1), Some(snake_pos.1)),
            2 => (Some(snake_pos.0), clamp(snake_pos.1 + 1)),
            3 => (snake_pos.0.checked_sub(1), Some(snake_pos.1)),
            _ => unreachable!()
        };

        // debug_grid(&board, &snake_pos, snake_dir);

        match new_pos {
            (Some(x), Some(y)) => snake_pos = (x, y),
            _ => break
        };

        let cell = &board[snake_pos.1][snake_pos.0].clone();
        if *cell == 1 {
            board[snake_pos.1][snake_pos.0] = 0;
            score += 1.;
            spawn_apple(&mut board, &snake_pos, &mut apple_pos);
        }

        attempts += 1;
        if attempts > 1000 {
            break;
        }
    }

    score.max(0.1)
}

fn debug_grid(grid: &Vec<Vec<usize>>, pos: &(usize, usize), dir: usize) {
    for (y, row) in grid.iter().enumerate() {
        for (x, &square) in row.iter().enumerate() {
            let symbol = match square {
                0 => "⬜️",
                1 => "⬛️",
                _ => "? ",
            };
            if x == pos.0 && y == pos.1 {
                print!("{}", match dir {
                    0 => "🔼 ",
                    1 => "▶️ ",
                    2 => "🔽 ",
                    3 => "◀️ ",
                    _ => unreachable!()
                });
            }else {
                print!("{}", symbol);
            }
        }
        println!();
    }
    println!();
}


// fn fitness(network: &mut NeatNetwork) -> f32 {
//     let a = vec![
//         (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]),
//         (vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]),
//         (vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]),
//         (vec![0.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0, 0.0]),
//     ];

//     let mut error: f32 = 0.0;
//     for (input, expected_output) in a.iter() {
//         let output = network.calculate_output(input.clone());
//         for (o, e) in output.iter().zip(expected_output) {
//             error += (e - o).powi(2);
//         }
//     }
    
//     if error > 0. { 1. / error } else { 0.0 }
// }

// pub fn fitness_xor(network: &mut NeatNetwork) -> f32 {
//     let xor = vec![((0.0, 0.0), 0.0),
//                     ((1.0, 0.0), 1.0),
//                     ((0.0, 1.0), 1.0),
//                     ((1.0, 1.0), 0.0)];

//     let mut error: f32 = 0.0;
//     for &((input1, input2), expected_output) in xor.iter() {
//         let output = network.calculate_output(vec![input1, input2])[0];
//         error += (expected_output - output).powi(2);
//     }
    
//     if error > 0. { 1. / error } else { 0.0 }
// }


// fn score_network(network: &mut NeatNetwork) -> f32 {
//     let mut game = Board::new();
//     let mut score = 0.0;

//     loop {
//         let mut cells: Vec<f32> = game.cells().iter().map(Cell::to_f32).collect();
//         let mut tiles = vec![0.0; 36];
//         for tile in game.get_tile_buffer() {
//             if let Some(tile) = tile {
//                 let index_of_tile = tile as u8 as usize;
//                 tiles[index_of_tile] = 1.;
//             }
//         }
//         cells.append(&mut tiles);

//         let input = cells;
//         let decision = network.calculate_output(input);

//         let x = utils::get_max_index_list(&decision[0..8]);
//         let y = utils::get_max_index_list(&decision[8..16]);
//         let tile_buffer_index = utils::get_max_index_list(&decision[16..19]);
//         let mut attempt = 0;

//         let mut success = false;
//         while !success {
//             game.set_cursor(tile_buffer_index[attempt % 3]);
//             game.set_cursor_position((x[attempt], y[attempt]).into());

//             match game.place() {
//                 Ok(e) => {
//                     score = e as f32;
//                     success = true;
//                 },
//                 Err(error) => {
//                     match error {
//                         PlacementError::OutOfBounds | PlacementError::InvalidPlacement => {
//                             if attempt >= 7 {
//     dbg!(score);

//                                 return score
//                             }
//                             attempt += 1;
//                             continue;
//                         },
//                         PlacementError::GameOver => {
//     dbg!(score);

//                             return score
//                         }
//                     }
//                 }
//             };
//         }

//         game.log();
//     }
// }

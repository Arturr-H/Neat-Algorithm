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
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use trainer::{evolution::{self, Evolution, EvolutionBuilder}, config::stop_condition::{StopCondition, StopConditionType}};
use trainer::species::Species;
use rand::{thread_rng, Rng};
use utils::find_max_index;

fn main() -> () {
    let mut _evolution = Evolution::new()
        .batch_size(50)
        .with_species_size(6)
        .with_input_nodes(4)
        .with_output_nodes(4)
        .with_hidden_activation(Activation::LeakyRelu)
        .with_output_activation(Activation::Sigmoid)
        .with_stop_condition(StopCondition::after(StopConditionType::FitnessReached(1000.)))
        .set_fitness_function(|e| snake(e, false))
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

pub fn snake(network: &mut NeatNetwork, log: bool) -> f32 {
    let mut board = vec![vec![0usize; 8]; 8];
    let mut rng = thread_rng();
    let mut score = 0.0f32;
    let mut snake_pos: (usize, usize) = (rng.gen_range(0..8), rng.gen_range(0..8));
    let mut apple_pos: (usize, usize) = (4, 4);
    let mut attempts = 0;

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
        // let mut x = vec![0.0; 3];
        // let mut y = vec![0.0; 3];
        let mut sp = vec![snake_pos.0 as f32 / 7., snake_pos.1 as f32 / 7.];
        let mut ap = vec![apple_pos.0 as f32 / 7., apple_pos.1 as f32 / 7.];
        
        let input = sp.into_iter().chain(ap).collect();
        let decision = network.calculate_output(input);
        let snake_dir = find_max_index(&decision);  // 0 = top, 1 = right ...

        if log {
            print_board_single(&board, snake_pos, apple_pos);
            std::thread::sleep(Duration::from_millis(200));
        }

        match snake_dir {
            0 => snake_pos.1 = if snake_pos.1 == 0 { break } else { snake_pos.1 - 1 },
            1 => snake_pos.0 = if (snake_pos.0 + 1) > 7 { break } else { snake_pos.0 + 1 },
            2 => snake_pos.1 = if (snake_pos.1 + 1) > 7 { break } else { snake_pos.1 + 1 },
            3 => snake_pos.0 = if snake_pos.0 == 0 { break } else { snake_pos.0 - 1 },
            _ => {}
        };

        let cell = &board[snake_pos.1][snake_pos.0].clone();
        if *cell == 1 {
            board[snake_pos.1][snake_pos.0] = 0;
            score += 1.;
            spawn_apple(&mut board, &snake_pos, &mut apple_pos);
        }

        attempts += 1;
        if attempts > 500 {
            break;
        }

        // Snakes who live longer should be rewarded
        score += 0.001;
    }

    score.max(0.001)
}
fn print_board_single(board: &Vec<Vec<usize>>, snake: (usize, usize), apple_pos: (usize, usize)) {
    let mut display_board = board.clone();
    display_board[snake.1][snake.0] = 1;
    display_board[apple_pos.1][apple_pos.0] = 2;
    for row in display_board {
        for cell in row {
            match cell {
                0 => print!(". "),
                1 => print!("O "),
                2 => print!("A "),
                _ => {}
            }
        }
        println!();
    }
    println!();
    println!();
    println!();
}


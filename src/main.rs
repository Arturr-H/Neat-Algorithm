#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

/* Modules */
mod neural_network;
mod utils;

use std::time::Duration;

/* Imports */
use neural_network::{layer, network::Network};
use block_blast::board::{self, board::Board, board_error::PlacementError, cell::Cell};

fn main() -> () {
    let mut game = Board::new();
    let mut score = 0;
    let mut net = Network::new(&[
        64 /* All cells */ + 36 /* Tiles to choose from */,
        8/*x*/ + 8/*y - Coordinate for tile placement */ + 3 /* What tile buffer to choose */
    ]);

    let restart = |game: &mut Board, score: usize| {
        if score > 50 {
            panic!("SHIT YOU ARe GOOD! GOOD JOB AI");
        }else {
            println!("{}RESTART \r\n", "\r\n".repeat(10));
            game.restart();
        }
    };
    
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
        let decision = net.feed_forward(input);

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
                    score = e;
                    success = true;
                },
                Err(error) => {
                    match error {
                        PlacementError::OutOfBounds | PlacementError::InvalidPlacement => {
                            if attempt >= 7 {
                                restart(&mut game, score);
                                break;
                            }
                            attempt += 1;
                            continue;
                        },
                        PlacementError::GameOver => {
                            restart(&mut game, score);
                        }
                    }
                }
            };
        }
        game.log();
        // std::thread::sleep(Duration::from_secs(1));
    }
}



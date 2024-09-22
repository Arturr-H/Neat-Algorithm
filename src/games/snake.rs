use rand::{seq::SliceRandom, Rng};

const GRID_SIZE: usize = 8;
const INITIAL_SNAKE_LENGTH: usize = 3;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

pub struct SnakeGame {
    pub snake: Vec<Position>,
    pub direction: Direction,
    pub apple: Position,
    pub is_game_over: bool,
    pub score: f32,
    pub apple_worth: f32,
}

impl SnakeGame {
    pub fn new() -> Self {
        let mut snake = Vec::new();
        for i in 0..INITIAL_SNAKE_LENGTH {
            snake.push(Position { x: GRID_SIZE / 2, y: GRID_SIZE / 2 + i });
        }
        
        let apple = SnakeGame::generate_apple(&snake).unwrap();

        SnakeGame {
            snake,
            direction: Direction::North,
            apple,
            is_game_over: false,
            score: 0.,
            apple_worth: 1.
        }
    }

    pub fn generate_apple(snake: &Vec<Position>) -> Option<Position> {
        let mut rng = rand::thread_rng();
        let mut tries = 0;
        let mut available_coords = Vec::new();

        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                if !snake.contains(&Position { x, y }) {
                    available_coords.push(Position { x, y });
                }
            }
        }

        if available_coords.is_empty() { return None }

        loop {
            tries += 1;
            let new_apple = available_coords.choose(&mut rng).unwrap();
            
            if tries > 50 {
                return None;   
            }else if !snake.contains(&new_apple) {
                return Some(*new_apple);
            }
        }
    }

    pub fn update(&mut self) {
        if self.is_game_over {
            return;
        }else {
            self.score += 0.001;
        }

        let head = self.snake.first().unwrap();
        let new_head = match self.direction {
            Direction::North => Position { x: head.x, y: head.y.wrapping_sub(1) },
            Direction::East  => Position { x: head.x + 1, y: head.y },
            Direction::South => Position { x: head.x, y: head.y + 1 },
            Direction::West  => Position { x: head.x.wrapping_sub(1), y: head.y },
        };
        if new_head.x >= GRID_SIZE || new_head.y >= GRID_SIZE {
            self.is_game_over = true;
            return;
        }
        if self.snake.contains(&new_head) {
            self.is_game_over = true;
            return;
        }
        self.snake.insert(0, new_head);
        self.apple_worth = (self.apple_worth - 0.01).max(0.3);
        if new_head == self.apple {
            self.apple = match SnakeGame::generate_apple(&self.snake) {
                Some(e) => e,
                None => {
                    self.score += 10.;
                    self.is_game_over = true;
                    return
                }
            };
            self.score += self.apple_worth;
            self.apple_worth = 1.;
        } else {
            self.snake.remove(self.snake.len() - 1);
        }
    }

    pub fn set_direction(&mut self, new_direction: Direction) {
        if (self.direction == Direction::North && new_direction != Direction::South)
            || (self.direction == Direction::South && new_direction != Direction::North)
            || (self.direction == Direction::East && new_direction != Direction::West)
            || (self.direction == Direction::West && new_direction != Direction::East)
        {
            self.direction = new_direction;
        }
    }

    pub fn display(&self) {
        for y in 0..GRID_SIZE {
            for x in 0..GRID_SIZE {
                let pos = Position { x, y };

                if self.snake.first() == Some(&pos) {
                    print!("H ");
                } else if self.snake.contains(&pos) {
                    print!("S ");
                } else if self.apple == pos {
                    print!("A ");
                } else {
                    print!(". ");
                }
            }
            println!();
        }
        println!();
    }

    pub fn get_proximity(&self) -> [f32; 4] {
        let head = self.snake.first().unwrap();
        let mut proximity = [0.0; 4];

        let directions = [
            Position { x: head.x, y: head.y.wrapping_sub(1) },
            Position { x: head.x + 1, y: head.y },
            Position { x: head.x, y: head.y + 1 },
            Position { x: head.x.wrapping_sub(1), y: head.y },
        ];

        for (i, &pos) in directions.iter().enumerate() {
            if pos.x >= GRID_SIZE || pos.y >= GRID_SIZE || self.snake.contains(&pos) {
                proximity[i] = 1.0;
            }
        }

        proximity
    }
}

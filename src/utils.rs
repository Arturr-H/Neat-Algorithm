use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Returns the index of the maximum value in the input vector
pub fn find_max_index(input: &[f32]) -> usize {
    input
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)|
            a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Less)
        )
        .unwrap_or((0, &0.0))
        .0
}

pub fn get_max_index_list(input: &[f32]) -> Vec<usize> {
    let mut b: Vec<usize> = (0..input.len()).collect();

    // Sort the indices based on the values in `a` in descending order
    b.sort_by(|&i, &j| input[j].partial_cmp(&input[i]).unwrap());

    b
}

pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    pub fn start(name: &str) -> Self {
        let start = Instant::now();
        println!("Starting {}...", name);
        Self {
            name: name.to_string(),
            start,
        }
    }

    pub fn end(self) {
        let duration = self.start.elapsed();
        println!("{} took: {:?}", self.name, duration);
    }
}

pub fn get_unix_time() -> u128 {
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_millis()
}

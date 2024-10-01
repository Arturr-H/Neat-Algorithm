#[derive(Clone, Copy, Debug)]
pub enum FitnessAverage {
    /// α = 0.75
    Exponential75,
    /// α = 0.5
    Exponential50,
    /// α = 0.25
    Exponential25,

    /// sum / count
    Regular,
}

impl FitnessAverage {
    pub fn run(&self, values: &[f32]) -> f32 {
        match self {
            Self::Exponential25 => exponential_average(values, 0.25),
            Self::Exponential50 => exponential_average(values, 0.50),
            Self::Exponential75 => exponential_average(values, 0.75),
            Self::Regular => values.iter().sum::<f32>() / values.len() as f32,
        }
    }
}

pub fn exponential_average(values: &[f32], alpha: f32) -> f32 {
    let mut avg = 0.0;
    let mut weight_sum = 0.0;

    for (i, &value) in values.iter().enumerate() {
        let weight = alpha.powi(i as i32);
        avg += value * weight;
        weight_sum += weight;
    }

    if weight_sum != 0.0 {
        avg / weight_sum
    } else {
        0.0
    }
}

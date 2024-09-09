use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub struct Layer {

    /// The weights are stored like this
    /// [[w; current_layer_size]; prev_layer_size]
    /// 
    /// Therefore to get the fist node from the previous
    /// layer's connection to the second node in the current
    /// layer you need to call `weights[0][1]`
    weights: Vec<Vec<f32>>,

    /// Each index corresponds to the same neuron in the layer
    biases: Vec<f32>,

    /// The amount of neurons in this layer
    size: usize,
    /// The amount of neurons in the previous layer
    prev_layer_size: usize
}

impl Layer {
    pub fn new(prev_layer_size: usize, size: usize) -> Self {
        let std = f32::sqrt(2.0 / prev_layer_size as f32);
        let normal_distribution = Normal::new(0.0, std).unwrap();
        let mut rng = thread_rng();

        let biases = vec![0.1; size];
        let weights: Vec<Vec<f32>> = (0..prev_layer_size)
            .map(|_| {
                (0..size)
                    .map(|_| normal_distribution.sample(&mut rng))
                    .collect::<Vec<f32>>()
            })
            .collect();

        Self {
            size,
            prev_layer_size,
            biases,
            weights,
        }
    }

    /// Input len needs to be same as `self.size`
    pub fn calculate_output(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), self.prev_layer_size);

        // Initialize neurons with their biases in the current layer
        let mut neurons = self.biases.clone();

        for (prev_layer_index, input_neuron) in inputs.iter().enumerate() {
            for curr_layer_index in 0..self.size {
                let weight = self.weights[prev_layer_index][curr_layer_index];
                let weighted_sum = weight * input_neuron;
                neurons[curr_layer_index] += weighted_sum;
            }
        }

        neurons
    }
}
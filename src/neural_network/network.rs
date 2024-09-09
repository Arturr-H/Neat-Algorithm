use super::layer::Layer;

pub struct Network {
    /// Input = index 0
    /// output = index len - 1
    /// hidden = rest
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        
        for sizes in layer_sizes.windows(2) {
            let prev_size = sizes[0];
            let size = sizes[1];

            let layer = Layer::new(prev_size, size);
            layers.push(layer);
        }

        Self {
            layers,
        }
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut inputs = inputs.clone();
        for layer in &self.layers {
            inputs = layer.calculate_output(inputs);
        }

        inputs
    }
}

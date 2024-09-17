use std::default;
use serde_derive::{Serialize, Deserialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct NetworkActivations {
    pub hidden: Activation,
    pub output: Activation
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[repr(u8)]
pub enum Activation {
    Relu,
    LeakyRelu,
    Sigmoid,
    Softmax
}
impl Activation {
    pub fn run(&self, inputs: &Vec<f32>, index: usize) -> f32 {
        match &self {
            Self::LeakyRelu => leaky_relu(inputs, index),
            Self::Relu => relu(inputs, index),
            Self::Sigmoid => sigmoid(inputs, index),
            Self::Softmax => softmax(inputs, index)
        }
    }
}

impl NetworkActivations {
    pub fn new(hidden: Activation, output: Activation) -> Self {
        Self { hidden, output }
    }
}

pub fn relu(inputs: &Vec<f32>, index: usize) -> f32 {
    if inputs[index] > 0.0 { return inputs[index] }
    else { return 0.0 }
}

const LEAKY_RELU_NEGATIVE_SLOPE: f32 = 0.1;
pub fn leaky_relu(inputs: &Vec<f32>, index: usize) -> f32 {
    if inputs[index] > 0.0 { return inputs[index] }
    else { return LEAKY_RELU_NEGATIVE_SLOPE * inputs[index] }
}

pub fn sigmoid(inputs: &Vec<f32>, index: usize) -> f32 {
    1.0 / (1.0 + f32::exp(-inputs[index]))
}

fn softmax(inputs: &Vec<f32>, index: usize) -> f32 {
    let mut exponent_sum = 0.0;
    for i in 0..inputs.len() {
        exponent_sum += inputs[i].exp();
    }

    let res = inputs[index].exp() / exponent_sum;
    res
}

// for #[serde(skip)] macro
impl Default for NetworkActivations {
    fn default() -> Self {
        Self { hidden: Activation::LeakyRelu, output: Activation::Sigmoid }
    }
}

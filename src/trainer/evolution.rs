use crate::neural_network::network::NeatNetwork;

/// Struct to make a set amount of networks
/// compete against eachother.
pub struct EvolutionBuilder {
    /// How many networks that are going to compete
    /// against eachother.
    batch_size: Option<usize>,

    /// The amount of input neurons
    input_nodes: Option<usize>,

    /// The amount of output neurons
    output_nodes: Option<usize>,

    /// This function will run the network trough some test that
    /// the network is trained to do. The function will return an
    /// f32 which evaluates the performance of the network. Higher
    /// return value => better performing network
    fitness_function: Option<fn(&mut NeatNetwork) -> f32>
}

pub struct Evolution {
    /// All the diffrent networks that compete
    networks: Vec<NeatNetwork>,
    batch_size: usize,
    input_nodes: usize,
    output_nodes: usize,
    fitness_function: fn(&mut NeatNetwork) -> f32,
}

impl EvolutionBuilder {
    pub fn new() -> Self {
        Self {
            batch_size: None,
            output_nodes: None,
            input_nodes: None,
            fitness_function: None
        }
    }

    /// The amount of networks that compete against each
    /// other in each generation
    pub fn batch_size(&mut self, size: usize) -> &mut Self { self.batch_size = Some(size); self }

    /// Set amount of input nodes for each network
    pub fn with_input_nodes(&mut self, nodes: usize) -> &mut Self { self.input_nodes = Some(nodes); self }
    /// Set amount of output nodes for each network
    pub fn with_output_nodes(&mut self, nodes: usize) -> &mut Self { self.output_nodes = Some(nodes); self }

    /// This function will run the network trough some test that
    /// the network is trained to do. The function will return an
    /// f32 which evaluates the performance of the network. Higher
    /// return value => better performing network
    /// 
    /// The function could be a game that the network gets to play
    /// and returns the score it managed to get.
    pub fn set_fitness_function(&mut self, func: fn(&mut NeatNetwork) -> f32) -> &mut Self { self.fitness_function = Some(func); self }

    /// Compile all set values and make this
    /// struct ready for evolution
    pub fn build(&mut self) -> Evolution {
        assert!(self.batch_size.is_some(), "Batch is required for evolution");
        assert!(self.input_nodes.is_some(), "Input node amount is required for evolution");
        assert!(self.output_nodes.is_some(), "Output node amount is required for evolution");
        assert!(self.fitness_function.is_some(), "Fitness function is required for evolution");
        let batch_size = self.batch_size.unwrap();
        let input_nodes = self.input_nodes.unwrap();
        let output_nodes = self.output_nodes.unwrap();

        let mut networks = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            networks.push(NeatNetwork::new(input_nodes, output_nodes))
        }

        Evolution {
            networks,
            batch_size,
            input_nodes,
            output_nodes,
            fitness_function: self.fitness_function.unwrap()
        }
    }
}


impl Evolution {
    pub fn new() -> EvolutionBuilder {
        EvolutionBuilder::new()
    }

    /// Runs the networks through a generation of mutation and selection
    pub fn generation(&mut self) -> () {
        // First mutate every network
        for net in self.networks.iter_mut() {
            net.mutate();
        }
    
        // Evaluate the best networks
        let mut performances: Vec<f32> = Vec::new();
        for (index, net) in self.networks.iter_mut().enumerate() {
            performances.push((self.fitness_function)(net));
        }
        let top_5 = top_n_with_indices(&performances, 5);

        dbg!(top_5);
    }
}

fn top_n_with_indices(numbers: &Vec<f32>, n: usize) -> Vec<usize> {
    let mut indexed_numbers: Vec<(usize, &f32)> = numbers.iter().enumerate().collect();
    indexed_numbers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indexed_numbers.into_iter().take(n).map(|(i, _)| i).collect()
}

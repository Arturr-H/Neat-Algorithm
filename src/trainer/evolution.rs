use std::{collections::{HashMap, HashSet}, hash::Hash, sync::{Arc, Mutex}};
use rand::{thread_rng, Rng};

use crate::neural_network::{activation::{Activation, NetworkActivations}, connection_gene::ConnectionGene, network::{self, NeatNetwork}};
use super::{evolution_config::StopCondition, species::Species};

/// How many times we mutate the representative before cloning
/// and creating a distinct species
const SPECIES_REPRESENTATIVE_MUTATIONS: usize = 20;
const AMOUNT_OF_INITIAL_SPECIES: usize = 10;
const DEFAULT_SPECIES_SIZE: usize = 10;

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
    fitness_function: Option<fn(&mut NeatNetwork) -> f32>,

    /// How many networks (including representative) we should have
    /// in each species group (how many networks who compete against
    /// eachother each generation)
    species_size: usize,

    stop_condition: StopCondition,

    hidden_activation: Activation,
    output_activation: Activation,
}

pub struct Evolution {
    /// All the diffrent networks that compete in groups
    species: Vec<Species>,
    batch_size: usize,
    input_nodes: usize,
    output_nodes: usize,
    fitness_function: fn(&mut NeatNetwork) -> f32,
    activations: NetworkActivations,
    global_innovation_number: Arc<Mutex<usize>>,
    stop_condition: StopCondition,
    generation: usize,

    /// To check if we've already got a connection
    /// between two nodes. NEEDS to be (min, max),
    /// and by that I mean the first integer index
    /// needs to be less than the second.
    /// 
    /// The key, (usize, usize) depicts (node_in,
    /// node_out) for some weight, and the value
    /// (usize) depicts the innovation number of
    /// that gene
    global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,
}

impl EvolutionBuilder {
    pub fn new() -> Self {
        Self {
            batch_size: None,
            output_nodes: None,
            input_nodes: None,
            fitness_function: None,
            species_size: DEFAULT_SPECIES_SIZE,
            hidden_activation: Activation::LeakyRelu,
            output_activation: Activation::Sigmoid,
            stop_condition: StopCondition::default()
        }
    }

    /// The amount of networks that compete against each
    /// other in each generation
    pub fn batch_size(&mut self, size: usize) -> &mut Self { self.batch_size = Some(size); self }

    /// Set amount of input nodes for each network
    pub fn with_input_nodes(&mut self, nodes: usize) -> &mut Self { self.input_nodes = Some(nodes); self }
    /// Set amount of output nodes for each network
    pub fn with_output_nodes(&mut self, nodes: usize) -> &mut Self { self.output_nodes = Some(nodes); self }

    /// How many networks (including representative) we should have
    /// in each species group (how many networks who compete against
    /// eachother each generation per group)
    pub fn with_species_size(&mut self, size: usize) -> &mut Self { self.species_size = size; self }

    /// Set the activation function to be applied to all hidden nodes
    pub fn with_hidden_activation(&mut self, activation: Activation) -> &mut Self { self.hidden_activation = activation; self }
    /// Set the activation function to be applied to all output nodes
    pub fn with_output_activation(&mut self, activation: Activation) -> &mut Self { self.output_activation = activation; self }

    /// Set the stop condition for evolution
    pub fn with_stop_condition(&mut self, stop: StopCondition) -> &mut Self { self.stop_condition = stop; self }

    /// This function will run the network trough some test that
    /// the network is trained to do. The function will return an
    /// f32 which evaluates the performance of the network. Higher
    /// return value => better performing network
    /// 
    /// The function could be a game that the network gets to play
    /// and returns the score it managed to get.
    /// 
    /// ## WARNING
    /// Output NEEDS to be bigger than 0
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
        let species_size = self.species_size; // Default is `DEFAULT_SPECIES_SIZE`
        let hidden_activation = self.hidden_activation;
        let output_activation = self.output_activation;
        let activations = NetworkActivations::new(hidden_activation, output_activation);
        
        // Create species
        let mut species: Vec<Species> = Vec::with_capacity(batch_size);
        let mut global_occupied_connections = Arc::new(Mutex::new(HashMap::new()));
        let global_innovation_number = Arc::new(Mutex::new(0));

        for i in 0..batch_size {
            let representative = NeatNetwork::new(
                input_nodes,
                output_nodes,
                global_innovation_number.clone(),
                global_occupied_connections.clone(),
                activations,
            );

            species.push(Species::new(
                global_innovation_number.clone(),
                global_occupied_connections.clone(),
                representative,
                species_size,
            ));
        }

        Evolution {
            species,
            batch_size,
            input_nodes,
            output_nodes,
            fitness_function: self.fitness_function.unwrap(),
            global_innovation_number,
            global_occupied_connections,
            activations,
            stop_condition: self.stop_condition.clone(),
            generation: 0
        }
    }
}


impl Evolution {
    pub fn new() -> EvolutionBuilder {
        EvolutionBuilder::new()
    }

    /// Runs the networks through a generation of mutation and selection
    pub fn generation(&mut self) -> () {
        for species in self.species.iter_mut() {

            // Store fitness in each network
            species.generate_fitness(self.fitness_function);

            let fitness = species.previous_average_score();
            if self.stop_condition.should_stop(fitness, self.generation) {
                println!("STOP STOP STOP!");
                break;
            }

            // Cross-over
            species.crossover(self.fitness_function);

            // Mutate
            species.compute_generation();
        }
    }

    /// Returns a reference to all species
    pub fn species(&self) -> &Vec<Species> {
        &self.species
    }
}

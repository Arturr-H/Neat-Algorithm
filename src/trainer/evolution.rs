use std::{collections::{HashMap, HashSet}, hash::Hash, sync::{Arc, Mutex}};
use rand::{thread_rng, Rng};

use crate::neural_network::{activation::{Activation, NetworkActivations}, connection_gene::ConnectionGene, network::{self, NeatNetwork}};
use super::species::Species;

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
        let species_size = self.species_size; // Default is `DEFAULT_SPECIES_SIZE`
        let hidden_activation = self.hidden_activation;
        let output_activation = self.output_activation;
        let activations = NetworkActivations::new(hidden_activation, output_activation);
        
        // Create species
        let mut species: Vec<Species> = Vec::with_capacity(AMOUNT_OF_INITIAL_SPECIES);
        let mut global_occupied_connections = Arc::new(Mutex::new(HashMap::new()));
        let global_innovation_number = Arc::new(Mutex::new(0));

        for i in 0..AMOUNT_OF_INITIAL_SPECIES {
            let representative = NeatNetwork::new(
                input_nodes,
                output_nodes,
                global_innovation_number.clone(),
                global_occupied_connections.clone(),
                activations
            );

            species.push(Species::new(representative, species_size));
        }

        Evolution {
            species,
            batch_size,
            input_nodes,
            output_nodes,
            fitness_function: self.fitness_function.unwrap(),
            global_innovation_number,
            global_occupied_connections,
            activations
        }
    }
}


impl Evolution {
    pub fn new() -> EvolutionBuilder {
        EvolutionBuilder::new()
    }

    pub fn run(&mut self) -> () {
        for i in 0..1000 {
            let (avg_score, best_score) = self.generation();
            println!("Generation {i} Avg: {:.3} Best: {:.3}", avg_score, best_score);

            // std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    /// Runs the networks through a generation of mutation and selection
    pub fn generation(&mut self) -> (f32, f32) {
        let mut total_average_score = 0.0;
        let mut best_score = 0.0;

        for species in self.species.iter_mut() {
            species.compute_generation(self.fitness_function);
            let score = species.previous_aveage_score();
            total_average_score += score;
            if score > best_score {
                best_score = score;
            }
        }

        (total_average_score / self.species.len() as f32, best_score)
    }

    pub fn crossover(&self, mut network1: &NeatNetwork, mut network2: &NeatNetwork, fitness1: f32, fitness2: f32) -> NeatNetwork {
        let mut rng = thread_rng();
        let mut child_genes: Vec<ConnectionGene> = Vec::new();
    
        let mut i = 0;
        let mut j = 0;
        
        let net1_genes = network1.get_genes();
        let net2_genes = network2.get_genes();

        // Traverse both parent genomes
        while i < net1_genes.len() && j < net2_genes.len() {
            let gene1 = &net1_genes[i];
            let gene2 = &net2_genes[j];

            if gene1.innovation_number() == gene2.innovation_number() {
                // Matching genes: Randomly inherit from either parent
                if rand::random() {
                    println!("----- Inheriting g1");
                    child_genes.push(gene1.clone());
                } else {
                    println!("----- Inheriting g2");
                    child_genes.push(gene2.clone());
                }
                i += 1;
                j += 1;
            } else if gene1.innovation_number() < gene2.innovation_number() {
                // Disjoint gene from net1_genes
                if fitness1 >= fitness2 {
                    println!("----- Inheriting d1");
                    child_genes.push(gene1.clone());
                }
                i += 1;
            } else {
                // Disjoint gene from net2_genes
                if fitness2 >= fitness1 {
                    println!("----- Inheriting d2");
                    child_genes.push(gene2.clone());
                }
                j += 1;
            }
        }
        
        // Handle excess genes from the longer genome
        if fitness1 >= fitness2 {
            while i < net1_genes.len() {
                println!("----- Inheriting ex1");
                child_genes.push(net1_genes[i].clone());
                i += 1;
            }
        } else {
            while j < net2_genes.len() {
                println!("----- Inheriting ex2");
                child_genes.push(net2_genes[j].clone());
                j += 1;
            }
        }
        
        NeatNetwork::new_with_genes(
            self.input_nodes, self.input_nodes,
            self.global_innovation_number.clone(),
            self.global_occupied_connections.clone(),
            self.activations,
            child_genes,
        )
    }


    /// Distance
    pub fn distance(net1: &NeatNetwork, net2: &NeatNetwork) -> f32 {
        let net1_highest = net1.get_highest_local_innovation();
        let net2_highest = net2.get_highest_local_innovation();

        // Excess genes are the difference between the maximum
        // local innovation number of each network. 
        let mut excess = 0.;
        let mut highest_local_innovation;
        if net1_highest > net2_highest {
            highest_local_innovation = net1_highest;
            for gene in net1.get_genes() {
                if gene.innovation_number() > net2_highest { excess += 1.; }
            }
        }else {
            highest_local_innovation = net2_highest;
            for gene in net2.get_genes() {
                if gene.innovation_number() > net1_highest { excess += 1.; }
            }
        }

        // Average weight of matching genes
        let mut total_weight_diff = 0.0;
        let mut matching_weights = 0;

        // Disjoint genes are genes that do not share historical
        // markings with the other network
        let mut disjoint = 0.;

        // (node_in, node_out), weight
        let mut net1_genes = HashMap::new();

        for gene in net1.get_genes() {
            if gene.innovation_number() <= highest_local_innovation {
                net1_genes.insert((gene.node_in(), gene.node_out()), gene.weight());
            }
        }

        for gene in net2.get_genes() {
            if gene.innovation_number() <= highest_local_innovation {
                if let Some(weight) = net1_genes.get(&(gene.node_in(), gene.node_out())) {
                    matching_weights += 1;
                    total_weight_diff += (weight - gene.weight()).abs();
                }else {
                    disjoint += 1.;
                }
            }
        }

        let average_weight_diff = total_weight_diff / matching_weights as f32;

        // TODO: Do constants
        let c1 = 1.0;
        let c2 = 1.0;
        let c3 = 0.4;
        let n = net2.get_genes().len().max(net1.get_genes().len()) as f32;
        let distance = (c1 * excess) / n + (c2 * disjoint) / n + c3 * average_weight_diff;

        distance
    }
}

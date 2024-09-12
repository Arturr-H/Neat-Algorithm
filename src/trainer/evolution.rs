use std::{collections::{HashMap, HashSet}, sync::{Arc, Mutex}};
use crate::{neural_network::network::NeatNetwork, utils::Timer};
use super::species::Species;

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
}

pub struct Evolution {
    /// All the diffrent networks that compete
    networks: Vec<NeatNetwork>,
    batch_size: usize,
    input_nodes: usize,
    output_nodes: usize,
    fitness_function: fn(&mut NeatNetwork) -> f32,

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
        let global_innovation_number = Arc::new(Mutex::new(0));
        let mut global_occupied_connections = Arc::new(Mutex::new(HashMap::new()));
        for _ in 0..batch_size {
            networks.push(NeatNetwork::new(
                input_nodes,
                output_nodes,
                global_innovation_number.clone(),
                global_occupied_connections.clone()
            ));
        }

        Evolution {
            networks,
            batch_size,
            input_nodes,
            output_nodes,
            fitness_function: self.fitness_function.unwrap(),
            global_innovation_number,
            global_occupied_connections
        }
    }
}


impl Evolution {
    pub fn new() -> EvolutionBuilder {
        EvolutionBuilder::new()
    }

    /// Runs the networks through a generation of mutation and selection
    pub fn generation(&mut self) -> () {
        Species::new(NeatNetwork::new(
            self.input_nodes, self.output_nodes,
            self.global_innovation_number.clone(),
            self.global_occupied_connections.clone()
        )).eliminate(self.fitness_function);
    }

    /// Distance
    pub fn distance(net1: &NeatNetwork, net2: &NeatNetwork) -> f32 {
        let net1_highest = net1.get_highest_local_innovation();
        let net2_highest = net2.get_highest_local_innovation();

        // Excess genes are the difference between the maximum
        // local innovation number of each network. 
        let mut excess = 0.;
        let mut highest_local_innovation = 0;
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
        let c1 = 0.5;
        let c2 = 0.5;
        let c3 = 0.5;
        let n = net2.get_genes().len().max(net1.get_genes().len()) as f32;
        let distance = (c1 * excess) / n + (c2 * disjoint) / n + c3 * average_weight_diff;

        distance
    }
}

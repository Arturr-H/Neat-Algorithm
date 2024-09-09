use rand::{thread_rng, Rng};

use super::{connection_gene::ConnectionGene, node_gene::{NodeGene, NodeGeneType}};

#[derive(Debug)]
pub struct NeatNetwork {
    /// The amount of neurons to feed in
    input_size: usize,

    /// The amount of neurons in the output layer
    output_size: usize,

    /// A node gene is a dynamic node which is created 
    /// as the sort of "hidden" layer of the NEAT network
    /// 
    /// When the network mutates, a node gene has a chance
    /// of being created. 
    node_genes: Vec<NodeGene>,

    /// Dynamic weights, which connect two previously
    /// unconnected nodes
    connection_genes: Vec<ConnectionGene>
}

impl NeatNetwork {
    /// Create a new NEAT network with `input` amount
    /// of input neurons and `output` amount of output
    /// neurons.
    pub fn new(input: usize, output: usize) -> Self {
        // Create node genes
        let mut node_genes = Vec::with_capacity(input + output);
        for i in 0..input {
            node_genes.push(NodeGene::new(i, NodeGeneType::Input))
        }
        for i in input..(input+output) {
            node_genes.push(NodeGene::new(i, NodeGeneType::Ouptut))
        }

        // Create connections genes
        let mut connection_genes = Vec::new();
        let mut rng = thread_rng();
        for i in 0..10 {
            let node1 = rng.gen_range(0..input); // input
            let node2 = rng.gen_range(input..(input+output)); // output
            
            connection_genes.push(ConnectionGene::new(node1, node2, rng.gen_range(0.05..0.2)));
        }

        Self {
            input_size: input,
            output_size: output,

            node_genes,
            connection_genes
        }
    }
}

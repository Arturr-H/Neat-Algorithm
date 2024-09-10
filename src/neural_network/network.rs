use std::{collections::HashSet, fmt::Debug, hash::Hash};
use rand::{thread_rng, Rng};
use super::{connection_gene::ConnectionGene, node_gene::{NodeGene, NodeGeneType}};

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
    connection_genes: Vec<ConnectionGene>,

    /// Increases every time we add a new node gene so
    /// we can successfully assign each node gene with
    /// an unique index / ID
    node_gene_index: usize,

    /// To check if we've already got a connection
    /// between two nodes. NEEDS to be (min, max),
    /// and by that I mean the first integer index
    /// needs to be less than the second.
    occupied_connections: HashSet<(usize, usize)>
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
        let mut occupied_connections = HashSet::new();
        let mut rng = thread_rng();
        for i in 0..3 {
            let node1 = rng.gen_range(0..input); // input
            let node2 = rng.gen_range(input..(input+output)); // output

            occupied_connections.insert((node1, node2));
            connection_genes.push(ConnectionGene::new(node1, node2, rng.gen_range(0.05..0.2)));
        }

        Self {
            input_size: input,
            output_size: output,

            node_genes,
            connection_genes,
            node_gene_index: input + output,
            occupied_connections
        }
    }

    /// Will randomly add a new gene
    pub fn mutate(&mut self) -> () {
        let mut rng = thread_rng();
        let will_be_node_gene = thread_rng().gen_bool(0.5);
        
        if will_be_node_gene {
            // self.node_genes.push(NodeGene::new(
            //     self.node_gene_index,
            //     NodeGeneType::Regular
            // ));

            // choose an random existing connection
            // disable the chosen connection
            
            

            // create a new connection between the newly
            // created node and the input with a weight of 1 

            // create a new connection between the newly created
            // node and the ouput with the weight of the chosen connection
            

            self.node_gene_index += 1;
        }
        // Connection gene
        else {
            // TODO: Instead of creating connections between
            // TODO: output and input, try to also create some
            // TODO: between the "dynamic" hidden node genes.
            let mut connection_found = false;
            let mut connection_attempts = 0;
            let mut node1 = 0;
            let mut node2 = 0;

            while !connection_found {
                node1 = rng.gen_range(0..self.input_size); // input
                node2 = rng.gen_range(self.input_size..(self.input_size+self.output_size)); // output
                if !self.occupied_connections.contains(&(node1, node2)) {
                    connection_found = true;
                };

                if connection_attempts > 10 {
                    break;
                }
            }

            if connection_found {
                self.connection_genes.push(ConnectionGene::new(node1, node2, rng.gen_range(0.05..0.2)))
            }
        }
    }
}

impl Debug for NeatNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!(
r#"NeatNetwork:
        ├ input size: {}
        ├ output size: {}
        │
        ├ Gene:nodes = [{:?}],
────────┴ Gene:connections = [{:?}]
            "#,
            self.input_size,
            self.output_size,
            self.node_genes,
            self.connection_genes,
        ))
    }
}

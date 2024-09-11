use std::{collections::{HashMap, HashSet, VecDeque}, fmt::Debug, hash::Hash, process::Output};
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
    occupied_connections: HashSet<(usize, usize)>,

    /// Applied to new genes, which is used later
    /// for crossover mutation
    /// 
    /// Innovation numers in NEAT networks work like this: 
    /// If we have multiple networks, who mutate diffrently
    /// then each added gene (mutation) will increment the
    /// `global_innovation` number, and store it as its own
    /// innovation number.
    /// 
    /// Later when we crossover multiple genomes (networks)
    /// we'll only crossover genes with the same innovation
    /// number as they represent the same "idea", so we don't
    /// ruin the topology.
    global_innovation: usize
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

        // TODO 
        for i in 0..3 {
            let node_in = rng.gen_range(0..input); // input
            let node_out = rng.gen_range(input..(input+output)); // output

            let connection = Self::create_connection(node_in, node_out, rng.gen_range(0.05..0.2), &mut occupied_connections);
            if let Some(conn) = connection {
                connection_genes.push(conn);

                // Register that we've created a new outgoing weight for the new node
                node_genes[node_in].register_new_outgoing(connection_genes.len() - 1);
            }
        }

        Self {
            input_size: input,
            output_size: output,

            node_genes,
            connection_genes,
            node_gene_index: input + output,
            occupied_connections,
            global_innovation: 0
        }
    }

    /// Will randomly add a new gene
    pub fn mutate(&mut self) -> () {
        let mut rng = thread_rng();
        let will_be_node_gene = thread_rng().gen_bool(0.5);
        
        if will_be_node_gene {
            let length = self.connection_genes.len();
            let gene = &mut self.connection_genes[rng.gen_range(0..length)];
            let gene_node_in = gene.node_in();
            let gene_node_out = gene.node_out();

            gene.set_enabled(false);
            self.node_genes.push(NodeGene::new(
                self.node_gene_index,
                NodeGeneType::Regular,
            ));

            let input_connection = Self::create_connection(gene_node_in, self.node_gene_index, 1.0, &mut self.occupied_connections);
            let output_connection = Self::create_connection(self.node_gene_index, gene_node_out, gene.weight(), &mut self.occupied_connections);

            match (input_connection, output_connection) {
                (Some(input), Some(output)) => {
                    self.connection_genes.push(input);
                    self.connection_genes.push(output);

                    // Register that we've created a new incoming weight
                    // for the new node, and the updated node
                    self.node_genes[gene_node_in].register_new_outgoing(self.connection_genes.len() - 2);
                    self.node_genes[self.node_gene_index].register_new_outgoing(self.connection_genes.len() - 1);
                },

                _ => {}
            }

            self.node_gene_index += 1;
        }
        // Connection gene
        else {
            // TODO: Instead of creating connections between
            // TODO: output and input, try to also create some
            // TODO: between the "dynamic" hidden node genes.

            let node_in = rng.gen_range(0..self.input_size); // input
            let node_out = rng.gen_range(self.input_size..(self.input_size+self.output_size)); // output

            let connection = Self::create_connection(node_in, node_out, rng.gen_range(0.05..0.2), &mut self.occupied_connections);
            if let Some(conn) = connection {
                self.connection_genes.push(conn);
            }
        }
    }

    pub fn crossover(&mut self, network1: NeatNetwork, network2: NeatNetwork) -> NeatNetwork {
        //tar in parent 1 och parent 2. 
        
        // Min() av antal innovation numbers i vardera nätverk.
        // Denna siffra representerar antalet innovation numbers i det utvecklade nätverket.
        // Alla connections/nodes med en innovation number *högre* än detta markeras som "excess" och används helt enkelt inte.
    
        // resterade innovation numbers benäms som "disjoint".
        // Dessa disjoints blir pushade till det nya nätverket så att 
        // det finns kontinuitet hela vägen från innovation number 0..kortaste längden innovation numbers


        // börja med att kombinera "commons", nodes och connections som både networks har.
        // Connections / Nodes med lika innovation number kommer inte förändras, då de är samma


        //disjoints är valda från nätverket med högre fitness.

        todo!()
    }

    /// Tries to create a new connection
    fn create_connection(
        node_in: usize,
        node_out: usize,
        weight: f32,
        occupied_connections: &mut HashSet<(usize, usize)>
    ) -> Option<ConnectionGene> {
        let mut connection_found = false;
        let mut connection_attempts = 0;

        while !connection_found {
            if !occupied_connections.contains(&(node_in, node_out)) {
                connection_found = true;
            };

            if connection_attempts > 10 {
                break;
            }

            connection_attempts += 1;
        }

        if connection_found {
            occupied_connections.insert((node_in, node_out));
            Some(ConnectionGene::new(node_in, node_out, weight))
        }else {
            None
        }
    }

    /// Takes the input vector, and propagates it through all
    /// node genes and connections and returns the output layer.
    pub fn calculate_output(&mut self, input: Vec<f32>) -> Vec<f32> {
        assert!(input.len() == self.input_size);

        // TODO: IMPORTANT! After doing the instructions from the todo below, we need to 
        // TODO: IMPORTANT! get all nodes with type input, instead of self.node_genes[index] I THINK but not sure
        for (index, value) in input.iter().enumerate() {
            self.node_genes[index].set_activation(*value);
        }

        // TODO: IMPORTANT!: REPLACE THE BELOW RANGE 0..self.node_genes.len() WITH THE TOPOLOGY ORDER
        // TODO: IMPORTANT!: REPLACE THE BELOW RANGE 0..self.node_genes.len() WITH THE TOPOLOGY ORDER
        // TODO: IMPORTANT!: REPLACE THE BELOW RANGE 0..self.node_genes.len() WITH THE TOPOLOGY ORDER
        // TODO: IMPORTANT!: REPLACE THE BELOW RANGE 0..self.node_genes.len() WITH THE TOPOLOGY ORDER
        // TODO: IMPORTANT!: REPLACE THE BELOW RANGE 0..self.node_genes.len() WITH THE TOPOLOGY ORDER
        // Iterates through all neurons (non input layer) and sums all the incoming nodes * weight
        // and adds a bias. 
        for index in 0..self.node_genes.len() {
            let node = &self.node_genes[index];

            // Skip input nodes
            if node.node_type() == NodeGeneType::Input { continue; };

            // TODO
            // TODO
            // TODO
            // TODO
        }

        self.node_genes[self.input_size..(self.input_size + self.output_size)]
            .iter().map(|e| e.activation()).collect()
    }

    fn search_connection(&self, node_index: usize, nodes: &mut Vec<usize>) -> () {
        for connection in &self.connection_genes {
            /* Found a connection to another node from current */
            if connection.node_in() == node_index {
                self.search_connection(connection.node_out(), nodes)
            }
        }

        nodes.push(node_index);
    }
}

impl Debug for NeatNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!(
r#"NeatNetwork:
    ├ input size: {}
    ├ output size: {}
    │
    ├ Gene:nodes       = {:?},
────┴ Gene:connections = {:?}
            "#,
            self.input_size,
            self.output_size,
            self.node_genes,
            self.connection_genes,
        ))
    }
}

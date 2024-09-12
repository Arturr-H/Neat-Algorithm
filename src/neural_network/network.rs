use std::{collections::{HashMap, HashSet, VecDeque}, fmt::Debug, hash::Hash, process::Output, sync::{Arc, Mutex}};
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

    /// Reference to global innovation number
    global_innovation: Arc<Mutex<usize>>,

    /// An arc pointer pointing to the occupied connections
    /// which are stored in the `Evolution` struct. 
    global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,

    /// Only for checking if we already have a connection.
    /// If we don't have a local connection, but the connection 
    /// was found in occupied GLOBAL connections, then we will
    /// insert a new connection with the innovation number of the
    /// global connection
    local_occupied_connections: HashSet<(usize, usize)>,
}

impl NeatNetwork {
    /// Create a new NEAT network with `input` amount
    /// of input neurons and `output` amount of output
    /// neurons.
    /// 
    /// ## Global innovation
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
    pub fn new(
        input: usize,
        output: usize,
        global_innovation: Arc<Mutex<usize>>,
        global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>
    ) -> Self {
        // Create node genes
        let mut node_genes = Vec::with_capacity(input + output);
        for i in 0..input {
            node_genes.push(NodeGene::new(i, NodeGeneType::Input));
        }
        for i in input..(input+output) {
            node_genes.push(NodeGene::new(i, NodeGeneType::Ouptut));
        }

        // Create connections genes
        let mut local_occupied_connections = HashSet::new();
        let mut connection_genes = Vec::new();
        let mut local_innovation = 0;
        let mut rng = thread_rng();

        // Create a connection between every single input and output node
        for input_idx in 0..input {
            for output_idx in input..(input + output) {
                let (connection, should_increment) = Self::create_connection(
                    input_idx, output_idx,
                    rng.gen_range(0.05..0.2),
                    global_occupied_connections.clone(),
                    &mut local_occupied_connections,
                    local_innovation
                );

                if should_increment { local_innovation += 1 };
                if let Some(conn) = connection { connection_genes.push(conn); };

                // We know for sure that these connections exist
                // and won't be duplicates because of our for-loops
                // local_occupied_connections.insert((input_idx, output_idx));
                
                // Register that we've created a new outgoing weight for the new node
                node_genes[output_idx].register_new_incoming(connection_genes.len() - 1);
            }
        }

        // Set the global innovation because the "starter"
        // connection genes. We do -1 because we didn't set
        // a single connection for the last incremented inno.
        *global_innovation.lock().unwrap() = local_innovation - 1;

        Self {
            input_size: input,
            output_size: output,

            node_genes,
            connection_genes,
            node_gene_index: input + output,
            global_innovation,
            global_occupied_connections,
            local_occupied_connections
        }
    }

    /// Will randomly add a new gene
    pub fn mutate(&mut self) -> () {
        let mut rng = thread_rng();
        let will_be_node_gene = thread_rng().gen_bool(0.5);
        let current_innovation = self.get_innovation();

        // Split weight in half and place node in middle
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

            let (input_connection, should_increment_ingoing) = Self::create_connection(
                gene_node_in, self.node_gene_index,
                1.0,
                self.global_occupied_connections.clone(),
                &mut self.local_occupied_connections,
                current_innovation + 1
            );
            let (output_connection, should_increment_outgoing) = Self::create_connection(
                self.node_gene_index, gene_node_out,
                gene.weight(),
                self.global_occupied_connections.clone(),
                &mut self.local_occupied_connections,
                current_innovation + 2
            );

            // If the genes were actually created we increment the 
            // innovation number accordingly to match the previous
            // current_innovation + 1 and + 2
            if should_increment_ingoing { self.increment_innovation(); };
            if should_increment_outgoing { self.increment_innovation(); };

            // Register that we've created a new incoming weight
            // for the new node, and the updated node and push connection
            if let Some(input) = input_connection {
                self.connection_genes.push(input);
                self.node_genes[self.node_gene_index].register_new_incoming(self.connection_genes.len() - 2);
            };
            if let Some(output) = output_connection {
                self.connection_genes.push(output);
                self.node_genes[gene_node_out].register_new_incoming(self.connection_genes.len() - 1);
            };

            self.node_gene_index += 1;
        }
        // Connection gene
        else {
            // TODO: Instead of creating connections between
            // TODO: output and input, try to also create some
            // TODO: between the "dynamic" hidden node genes.

            let node_in = rng.gen_range(0..self.input_size); // input
            let node_out = rng.gen_range(self.input_size..(self.input_size+self.output_size)); // output

            let (connection, should_increment) = Self::create_connection(
                node_in, node_out,
                rng.gen_range(0.05..0.2),
                self.global_occupied_connections.clone(),
                &mut self.local_occupied_connections,
                current_innovation + 1
            );

            // Increase innovation to match the previous self.get_innovation() + 1
            if should_increment { self.increment_innovation(); };
            if let Some(conn) = connection { self.connection_genes.push(conn); };
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

    /// Tries to create a new connection. If the connection already
    /// exists, we return the connection but it has the innovation
    /// number of the already existing connection gene, so we don't
    /// create same connection genes with diffrent innovation numbers.
    /// 
    /// returns (Option<connection>, a) where a is a boolean indicating wether
    /// we have created a new innovation number. If a is false this
    /// function has instead taken the innovation number from a previous
    /// connection or just not created a connection (we don't need to increase innovation)
    /// 
    /// returns (None, a) if we already have the local connection established
    fn create_connection(
        node_in: usize,
        node_out: usize,
        weight: f32,
        global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,
        local_occupied_connections: &mut HashSet<(usize, usize)>,
        innovation_number: usize
    ) -> (Option<ConnectionGene>, bool) {
        let global_occupied_connections = &mut global_occupied_connections.lock().unwrap();
        match global_occupied_connections.get(&(node_in, node_out)) {
            Some(inherited_innovation) => {
                // Now that we've found that this connection exists for other
                // networks - we need to check if this connection also exists
                // for out current network (local). That's why we now will
                // check if this connection exists in local_occupied_connections
                if local_occupied_connections.contains(&(node_in, node_out)) {
                    // We return None because there's no need to
                    // push any connections because they already exist
                    (None, false)
                }else {
                    local_occupied_connections.insert((node_in, node_out));
                    (Some(ConnectionGene::new(node_in, node_out, weight, *inherited_innovation)), false)
                }
            },
            None => {
                // If it doesn't exist in global connections, it won't exist
                // in our local occupied connections.
                // Returns with true because we've incremented
                global_occupied_connections.insert((node_in, node_out), innovation_number);
                local_occupied_connections.insert((node_in, node_out));
                (Some(ConnectionGene::new(node_in, node_out, weight, innovation_number)), true)
            }
        }
    }

    /// Takes the input vector, and propagates it through all
    /// node genes and connections and returns the output layer.
    pub fn calculate_output(&mut self, input: Vec<f32>) -> Vec<f32> {
        assert!(input.len() == self.input_size);

        // Set activation for input nodes
        for (index, value) in input.iter().enumerate() {
            self.node_genes[index].set_activation(*value);
        }

        // Iterates through all neurons (non input layer) and sums all the incoming nodes * weight
        // and adds a bias. 
        let topology_order = self.topological_sort().unwrap();
        for index in topology_order {
            let node = &self.node_genes[index];

            // Skip input nodes
            if node.node_type() == NodeGeneType::Input { continue; };

            // TODO: Fix bias init (?)
            let mut sum = 0.1;
            for incoming_index in node.incoming_connection_indexes() {
                let connection = &self.connection_genes[*incoming_index];
                if !connection.enabled() { continue; };

                let prev_node = &self.node_genes[connection.node_in()];
                sum += prev_node.activation() * connection.weight();
            }

            self.node_genes[index].set_activation(sum);
        }

        self.node_genes[self.input_size..(self.input_size + self.output_size)]
            .iter().map(|e| e.activation()).collect()
    }

    /// The degree of a node is the amount of weights which are connected to it. And the
    /// indegree is the number of weights coming in, and outdegree are the number going out.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let num_nodes = self.node_genes.len();
        let mut in_degree = vec![0; num_nodes];
        let mut adj_list = vec![vec![]; num_nodes];

        for conn in &self.connection_genes {
            if conn.enabled() {
                adj_list[conn.node_in()].push(conn.node_out());
                in_degree[conn.node_out()] += 1;
            }
        }

        let mut stack: Vec<usize> = (0..num_nodes).filter(|&i| in_degree[i] == 0).collect();
        let mut sorted = vec![];

        while let Some(node) = stack.pop() {
            sorted.push(node);

            for &neighbor in &adj_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    stack.push(neighbor);
                }
            }
        }

        if sorted.len() == num_nodes {
            Some(sorted)
        } else {
            // Graph has a cycle, no valid topological sort
            None
        }
    }

    /// Increment global innovation number
    pub fn increment_innovation(&self) -> usize {
        let inno = &mut *self.global_innovation.lock().unwrap();
        *inno += 1;
        *inno
    }
    pub fn get_innovation(&self) -> usize {
        *self.global_innovation.lock().unwrap()
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

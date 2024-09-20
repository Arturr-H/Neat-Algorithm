/* Imports */
use std::{collections::{HashMap, HashSet}, fmt::Debug, hash::Hash, iter, sync::{Arc, Mutex}};
use rand::{rngs::ThreadRng, seq::SliceRandom, thread_rng, Rng};
use serde_derive::{Serialize, Deserialize};
use crate::{trainer::config::{mutation::GenomeMutationProbablities, network_config::NetworkConfig}, utils::Timer};
use super::{activation::NetworkActivations, connection_gene::ConnectionGene, node_gene::{NodeGene, NodeGeneType}};

/* Constants */
pub const AVERAGE_FITNESS_WINDOW_SIZE: usize = 25;

#[derive(Clone, Serialize, Deserialize)]
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
    #[serde(skip)]
    global_innovation: Arc<Mutex<usize>>,

    /// An arc pointer pointing to the occupied connections
    /// which are stored in the `Evolution` struct. 
    #[serde(skip)]
    global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,

    /// Only for checking if we already have a connection.
    /// If we don't have a local connection, but the connection 
    /// was found in occupied GLOBAL connections, then we will
    /// insert a new connection with the innovation number of the
    /// global connection
    local_occupied_connections: HashSet<(usize, usize)>,

    /// Used for calculating excess nodes in distance function
    highest_local_innovation: usize,

    /// The network activation functons for hidden and output
    /// layers (two diffrent)
    activations: NetworkActivations,

    /// Stores the previous points which were accumulated during
    /// the latest fitness evaluation test.
    previous_fitness: f32,

    /// Stores the fitnesses from the last 5 fitness tests
    average_fitness: [f32; AVERAGE_FITNESS_WINDOW_SIZE],

    #[serde(skip)]
    network_config: Arc<NetworkConfig>,

    topology_sort_cached: Vec<usize>
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
        global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,
        activations: NetworkActivations,
        network_config: Arc<NetworkConfig>
    ) -> Self {
        // Create node genes
        let mut node_genes = Vec::with_capacity(input + output);
        for i in 0..input {
            node_genes.push(NodeGene::new(i, NodeGeneType::Input, 0.0));
        }
        for i in input..(input+output) {
            node_genes.push(NodeGene::new(i, NodeGeneType::Output, 1.0));
        }

        // Create connections genes
        let mut local_occupied_connections = HashSet::new();
        let mut connection_genes = Vec::new();
        let mut highest_local_innovation = 0;
        let mut local_innovation = 0;
        let mut rng = thread_rng();

        // Create a connection between every single input and output node
        for input_idx in 0..input {
            for output_idx in input..(input + output) {
                let (connection, _) = Self::create_connection(
                    input_idx, output_idx,
                    rng.gen_range(0.0..1.0),
                    global_occupied_connections.clone(),
                    &mut local_occupied_connections,
                    &mut highest_local_innovation,
                    local_innovation,
                    true
                );

                // We don't need to know if we should increment
                // because it should always be true for initializing
                // weights
                local_innovation += 1;
                if let Some(conn) = connection { connection_genes.push(conn); };
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
            local_occupied_connections,
            highest_local_innovation,
            activations,
            previous_fitness: 0.,
            average_fitness: [0.; AVERAGE_FITNESS_WINDOW_SIZE],
            network_config: network_config.clone(),

            // TODO: Should we initialize with sorted or not? I think not
            topology_sort_cached: Vec::new()
        }
    }

    /// Create a new network but provide the genes (connections). Used
    /// after crossing over two parents' genes
    pub fn new_with_genes(
        input: usize,
        output: usize,
        global_innovation: Arc<Mutex<usize>>,
        global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,
        activations: NetworkActivations,
        connection_genes: Vec<ConnectionGene>,
        network_config: Arc<NetworkConfig>,
    ) -> Self {
        let mut highest_local_innovation = 0;
        let mut local_occupied_connections = HashSet::new();
        let mut highest_node_index = 0;
        let mut incoming: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut max_incoming = 0;
        
        for connection in &connection_genes {
            let node_in = connection.node_in();
            let node_out = connection.node_out();

            match incoming.get_mut(&node_out) {
                Some(e) => {
                    e.push(node_in);
                    if e.len() > max_incoming {
                        max_incoming = e.len();
                    }
                },
                None => { incoming.insert(node_out, vec![node_in]); }
            };

            local_occupied_connections.insert((node_in, node_out));
            if connection.innovation_number() > highest_local_innovation {
                highest_local_innovation = connection.innovation_number();
            }

            let local_highest_index = node_in.max(node_out);
            if local_highest_index > highest_node_index {
                highest_node_index = local_highest_index;
            }
        }

        let mut node_genes = Vec::new();
        for i in 0..highest_node_index + 1 {
            let node_type = match i {
                _ if i < input => NodeGeneType::Input,
                _ if i < (input + output) => NodeGeneType::Output,
                _ => NodeGeneType::Regular
            };

            // TODO NOT 0.5
            let x = match node_type {
                NodeGeneType::Input => 0.0,
                NodeGeneType::Regular => 0.5,
                NodeGeneType::Output => 1.0,
            };
            let mut node_gene = NodeGene::new(i, node_type, x);
            match incoming.get(&i) {
                Some(indexes) => {
                    node_gene.set_incoming_indexes(indexes.clone());
                    node_gene.set_x(indexes.len() as f32 / max_incoming as f32 * 0.8 + 0.1);
                },
                None => {}
            };

            node_genes.push(node_gene);
        }

        Self {
            input_size: input,
            output_size: output,
            node_genes,
            connection_genes,
            node_gene_index: highest_node_index + 1,
            global_innovation,
            global_occupied_connections,
            local_occupied_connections,
            highest_local_innovation,
            activations,
            previous_fitness: 0.,
            average_fitness: [0.; AVERAGE_FITNESS_WINDOW_SIZE],
            network_config: network_config.clone(),

            // TODO: Should we initialize with sorted or not? I think not
            topology_sort_cached: Vec::new()
        }
    }

    /// Mutates the network in one of many ways
    pub fn mutate(&mut self) -> () {
        let mut rng = thread_rng();
        let GenomeMutationProbablities {
            split_connection,
            create_connection,
            change_weight,
            toggle_weight,
            nothing
        } = self.network_config.mutation_probabilities;

        let probabilities: Vec<(usize, fn(&mut NeatNetwork) -> ())> = vec![
            /* Randomly select one gene for mutation */
            (change_weight, Self::mutate_random_gene_weight),
            
            /* Split connection or create new */
            (split_connection, Self::mutate_split_connection),
            (create_connection, Self::mutate_create_connection),

            /* Toggle random connection */
            (toggle_weight, Self::mutate_toggle_random_gene),

            /* Mutate nothing */
            (nothing, |_| {}),
        ];

        let total: usize = probabilities.iter().map(|e| e.0).sum();
        let random_number = rng.gen_range(0..total);
        let mut cumulative = 0;
        for (index, &(probability, func)) in probabilities.iter().enumerate() {
            cumulative += probability;
            if random_number < cumulative {
                (func)(self);
                //debug
                break;
            }
        }
    }

    fn mutate_random_gene_weight(&mut self) -> () {
        if self.get_genes().len() < 1 { return; };
        let mut rng = thread_rng();
        let length = self.connection_genes.len();
        let gene = &mut self.connection_genes[rng.gen_range(0..length)];
        gene.mutate_weight(&self.network_config.weight_change_probabilities);
    }

    fn mutate_toggle_random_gene(&mut self) -> () {
        if self.get_genes().len() < 1 { return; };
        let mut rng = thread_rng();
        let length = self.connection_genes.len();
        let gene = &mut self.connection_genes[rng.gen_range(0..length)];
        gene.set_enabled(rng.gen_bool(0.5));
    }

    fn mutate_split_connection(&mut self) -> () {
        if self.get_genes().len() < 1 { return; };
        let mut rng = thread_rng();
        let current_innovation = self.get_global_innovation();
        let length = self.connection_genes.len();
            
        let gene = &mut self.connection_genes[rng.gen_range(0..length)];
        let gene_node_in = gene.node_in();
        let gene_node_out = gene.node_out();

        let node_in_x = &self.node_genes[gene_node_in].x();
        let node_out_x = &self.node_genes[gene_node_out].x();
        let mut new_x;
        new_x = (node_in_x + node_out_x) / 2.;
        if node_in_x == node_out_x {
            new_x *= 1.05;
        }

        gene.set_enabled(false);
        self.node_genes.push(NodeGene::new(
            self.node_gene_index,
            NodeGeneType::Regular,
            new_x
        ));

        let (input_connection, should_increment_ingoing) = Self::create_connection(
            gene_node_in, self.node_gene_index,
            1.0,
            self.global_occupied_connections.clone(),
            &mut self.local_occupied_connections,
            &mut self.highest_local_innovation,
            current_innovation + 1,
            false
        );
        let (output_connection, should_increment_outgoing) = Self::create_connection(
            self.node_gene_index, gene_node_out,
            gene.weight(),
            self.global_occupied_connections.clone(),
            &mut self.local_occupied_connections,
            &mut self.highest_local_innovation,
            current_innovation + 2,
            false
        );

        // If the genes were actually created we increment the 
        // innovation number accordingly to match the previous
        // current_innovation + 1 and + 2
        if should_increment_ingoing { self.increment_global_innovation(); };
        if should_increment_outgoing { self.increment_global_innovation(); };

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

    /// Create a random connection
    fn mutate_create_connection(&mut self) -> () {
        let mut rng = thread_rng();
        let current_innovation = self.get_global_innovation();

        let topology_sorted = self.topological_sort().unwrap();
        let mut node_from_idx = rng.gen_range(0..topology_sorted.len() - 1);
        let mut node_to_idx = rng.gen_range(node_from_idx + 1..topology_sorted.len());
        let mut attempts = 0;

        // We don't want to connect a weight FROM an output node.
        while self.is_output(topology_sorted[node_from_idx]) {
            if attempts > 10 { return };
            node_from_idx = rng.gen_range(0..topology_sorted.len() - 1);
            attempts += 1;
        }

        // We don't want to connect a TO an input node.
        while self.is_input(topology_sorted[node_to_idx]) {
            if attempts > 20 { return };
            node_to_idx = rng.gen_range(node_from_idx + 1..topology_sorted.len());
            attempts += 1;
        }

        let node_from = topology_sorted[node_from_idx];
        let node_to = topology_sorted[node_to_idx];

        // We don't want to have connection 5-3 and create 3-5
        if self.local_occupied_connections.get(&(node_to, node_from)).is_some() { return }

        let (connection, should_increment) = Self::create_connection(
            node_from, node_to,
            rng.gen_range(0.0..1.0),
            self.global_occupied_connections.clone(),
            &mut self.local_occupied_connections,
            &mut self.highest_local_innovation,
            current_innovation + 1,
            false
        );

        // Increase innovation to match the previous self.get_global_innovation() + 1
        if should_increment { self.increment_global_innovation(); };
        if let Some(conn) = connection {
            self.connection_genes.push(conn);
            self.node_genes[node_to].register_new_incoming(self.connection_genes.len() - 1);
        };
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
        highest_local_innovation_number: &mut usize,
        innovation_number: usize,
        skip_loop_check: bool
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
                    /* If we don't find a cycle */
                    if skip_loop_check || !Self::has_cycle(local_occupied_connections.iter().chain(iter::once(&(node_in, node_out)))) {
                        let innovation = *inherited_innovation;
                        if innovation > *highest_local_innovation_number {
                            *highest_local_innovation_number = innovation;
                        }
                        local_occupied_connections.insert((node_in, node_out));
                        (Some(ConnectionGene::new(node_in, node_out, weight, innovation)), false)
                    }else {
                        (None, true)
                    }
                }
            },
            None => {
                if skip_loop_check || !Self::has_cycle(local_occupied_connections.iter().chain(iter::once(&(node_in, node_out)))) {
                    // If it doesn't exist in global connections, it won't exist
                    // in our local occupied connections.
                    // Returns with true because we've incremented
                    global_occupied_connections.insert((node_in, node_out), innovation_number);
                    local_occupied_connections.insert((node_in, node_out));
                    if innovation_number > *highest_local_innovation_number {
                        *highest_local_innovation_number = innovation_number;
                    }
                    (Some(ConnectionGene::new(node_in, node_out, weight, innovation_number)), true)
                }else {
                    (None, false)
                }
            }
        }
    }

    // TODO Remove
    pub fn title(&self, text: String) -> String {
        format!("{} ({:.3})", text, self.previous_fitness())
    }

    /// Takes the input vector, and propagates it through all
    /// node genes and connections and returns the output layer.
    pub fn calculate_output(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.calculate_output_cached_topology(input, true)
    }
    pub fn calculate_output_cached_topology(&mut self, input: Vec<f32>, use_topology_cache: bool) -> Vec<f32> {
        assert!(input.len() == self.input_size);
        for node_gene in self.node_genes.iter_mut() {
            node_gene.set_activation(0.0);
        }
        
        // Set activation for input nodes
        for (index, value) in input.iter().enumerate() {
            self.node_genes[index].set_activation(*value);
        }

        // Iterates through all neurons (non input layer) and sums all the incoming nodes * weight
        // and adds a bias. 
        let topology_order =
            if use_topology_cache { &self.topology_sort_cached }
            else {
                self.topology_sort_cached = self.topological_sort().unwrap();
                &self.topology_sort_cached
            };

        for index in topology_order {
            let node = &self.node_genes[*index];

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

            let activated_sum;
            match node.node_type() {
                NodeGeneType::Regular => {
                    activated_sum = self.activations.hidden.run(&vec![sum], 0);
                },
                NodeGeneType::Output => {
                    // We won't activate the output nodes here yet, as we will
                    // do it later on in this function
                    activated_sum = sum;
                },
                _ => unreachable!()
            };
            self.node_genes[*index].set_activation(activated_sum);
        }
        
        let outputs: Vec<f32> = self.node_genes[self.input_size..(self.input_size + self.output_size)]
            .iter().map(|e| e.activation()).collect();
    
        // Apply output activation
        outputs.iter().enumerate().map(|(i, e)| self.activations.output.run(&outputs, i)).collect()
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

    pub fn has_cycle<'a, I>(connections: I) -> bool 
    where I: Iterator<Item = &'a (usize, usize)> {
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(from, to) in connections {
            adj_list.entry(from).or_insert(vec![]).push(to);
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut rec_stack: HashSet<usize> = HashSet::new();

        fn dfs(
            node: usize,
            adj_list: &HashMap<usize, Vec<usize>>,
            visited: &mut HashSet<usize>,
            rec_stack: &mut HashSet<usize>,
        ) -> bool {
            visited.insert(node);
            rec_stack.insert(node);

            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        if dfs(neighbor, adj_list, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(&neighbor) {
                        return true;
                    }
                }
            }

            rec_stack.remove(&node);
            false
        }

        for &node in adj_list.keys() {
            if !visited.contains(&node) {
                if dfs(node, &adj_list, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }

        false
    }

    /// Save network
    pub fn save(&self, path: &str) -> () {
        match bincode::serialize(self) {
            Ok(e) => {
                match std::fs::write(path, e) {
                    Ok(e) => (),
                    Err(e) => {
                        println!("Cant write to {path}");
                        println!("{e:?}");
                    }
                };
            },
            Err(e) => {
                println!("Cant serialize");
                println!("{e:?}");
            }
        }
    }

    /// Retrieve from save file
    pub fn retrieve(path: &str) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => match bincode::deserialize::<Self>(&bytes) {
                Ok(contents) => contents,
                Err(e) => {
                    println!("Error deserializing {e:?}");
                    panic!("NOO");
                }
            },
            Err(e) => {
                println!("Error reading {e:?}");
                panic!("NOO");
            }
        }
    }

    /// Increment global innovation number
    pub fn increment_global_innovation(&self) -> usize {
        let inno = &mut *self.global_innovation.lock().unwrap();
        *inno += 1;
        *inno
    }
    pub fn get_global_innovation(&self) -> usize {
        *self.global_innovation.lock().unwrap()
    }

    pub fn get_highest_local_innovation(&self) -> usize {
        // ? If we count node genes as historical markers change this
        self.highest_local_innovation
    }
    pub fn get_genes(&self) -> &Vec<ConnectionGene> {
        &self.connection_genes
    }

    /// Returns true if the `index` is between
    /// self.input and self.output
    pub fn is_output(&self, index: usize) -> bool {
        index > self.input_size - 1 && index <= (self.input_size + self.output_size - 1)
    }
    /// Returns true if the `index` is between
    /// 0 and self.input
    pub fn is_input(&self, index: usize) -> bool {
        index < self.input_size
    }

    // Getters
    pub fn input_size(&self) -> usize { self.input_size }
    pub fn output_size(&self) -> usize { self.output_size }
    pub fn node_genes(&self) -> &Vec<NodeGene> { &self.node_genes }
    pub fn previous_fitness(&self) -> f32 { self.previous_fitness }
    pub fn activations(&self) -> NetworkActivations { self.activations }
    pub fn local_occupied_connections(&self) -> &HashSet<(usize, usize)> { &self.local_occupied_connections }
    pub fn network_config(&self) -> Arc<NetworkConfig> { self.network_config.clone() }

    /// Returns the average fitness of the previous 
    /// `AVERAGE_FITNESS_WINDOW_SIZE` nr of evaluations
    pub fn average_fitness(&self) -> f32 {
        self.average_fitness.iter().sum::<f32>() / AVERAGE_FITNESS_WINDOW_SIZE as f32
    }

    /// Store the fitness of the current network
    pub fn evaluate_fitness(&mut self, fitness_func: fn(&mut Self) -> f32) -> () {
        self.topology_sort_cached = self.topological_sort().unwrap();
        let score = (fitness_func)(self);
        self.previous_fitness = score;

        // Set new average
        self.average_fitness.rotate_right(1);
        self.average_fitness[0] = score;
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

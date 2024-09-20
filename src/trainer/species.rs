use std::{collections::HashMap, sync::{Arc, Mutex}};
use rand::{thread_rng, Rng};
use crate::neural_network::{connection_gene::ConnectionGene, network::{self, NeatNetwork}};

pub struct Species {
    /// The representative of a species is a network just like the other
    /// networks in the species, but when we compare distance between
    /// nets, we only compare any net to the representative, not to another
    /// net => too much computation. The first element of this list is 
    /// the representative (index 0)
    /// 
    /// All networks who are initially cloned from the representative
    networks: Vec<NeatNetwork>,

    previous_average_score: f32,

    global_innovation_number: Arc<Mutex<usize>>,
    global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,
    name: String,
}

impl Species {
    /// Creates a new species with a representative cloned to
    /// have `size` amount of identical networks that slightly
    /// mutate away from the representative
    pub fn new(
        global_innovation_number: Arc<Mutex<usize>>,
        global_occupied_connections: Arc<Mutex<HashMap<(usize, usize), usize>>>,
        representative: NeatNetwork,
        size: usize
    ) -> Self {
        assert!(size > 0, "Size must be at least 1 to fit representative");
        let mut networks: Vec<NeatNetwork> = Vec::with_capacity(size);
        
        networks.push(representative.clone());
        for i in 0..size - 1 {
            let mut net = representative.clone();
            net.mutate();
            networks.push(net);
        }

        Self {
            networks,
            previous_average_score: 0.,
            global_occupied_connections,
            global_innovation_number,
            name: Self::generate_name(),
        }
    }

    pub fn networks_mut(&mut self) -> &mut Vec<NeatNetwork> {
        &mut self.networks
    }
    
    pub fn networks(&self) -> & Vec<NeatNetwork> {
        & self.networks
    }

    /// Makes every net go trough a fitness function and determines the top 
    /// 30% of all nets. These nets automatically go to the next generation
    /// without changes. The 70% of the rest networks are randomly mutated
    /// and THEN placed in the next generation
    pub fn compute_generation(&mut self) -> () {
        let mut scores: Vec<f32> = self.networks.iter().map(|e| e.fitness()).collect();
        let mut total_score = scores.iter().sum::<f32>();
        self.previous_average_score = total_score / self.networks.len() as f32;

        // We won't modify the top 30, that's why we only deal with bottom 70 here
        let bottom_70_amount = (self.networks.len() as f32 * 0.7).round() as usize;
        let bottom_70 = Self::bottom_n_with_indices(&scores, bottom_70_amount);
        let worst_index = bottom_70[0];
        let top_index = Self::top_n_with_indices(&scores, 1)[0];
        for bottom_network_idx in bottom_70 {
            let bottom_net = &mut self.networks[bottom_network_idx];
            bottom_net.mutate();
        }

        // Chance of cloning one of the best networks to replace one of the worse
        if thread_rng().gen_bool(0.05) {
            // Worst
            self.networks[worst_index] = self.networks[top_index].clone();
        }
    }

    /// Crossover two parents and insert offspring
    pub fn crossover(&mut self, fitness_function: fn(&mut NeatNetwork) -> f32) -> () {
        assert!(self.networks.len() > 1);
        let mut rng = thread_rng();

        // Should happen after calculated fitness, that's why we can
        // get the fitness values here
        let mut summed_fitness = 0.0;
        let networks_with_fitness: Vec<(f32, &NeatNetwork)> = self.networks
            .iter()
            .map(|e| {
                let fitness = e.fitness();
                summed_fitness += fitness;
                (fitness, e)
            })
            .collect();

        let mut networks = Vec::new();
        let mut worst_performing: (usize, f32) = (0, f32::MAX);
        let n1 = &self.networks[rng.gen_range(0..self.networks.len())];
        let n2 = &self.networks[rng.gen_range(0..self.networks.len())];
        if summed_fitness > 0. {
            for i in 0..2 {
                let mut cumulative = 0.0;
                let mut random_fitness = rng.gen_range(0.0..summed_fitness);
                for (network_index, (fitness, net)) in networks_with_fitness.iter().enumerate() {
                    if fitness < &worst_performing.1 {
                        worst_performing.0 = network_index;
                        worst_performing.1 = *fitness;
                    }

                    cumulative += fitness;
                    if random_fitness < cumulative {
                        networks.push((net, fitness));
                    }
                }
            }
        }else {
            networks.push((&n1, &0.0));
            networks.push((&n2, &0.0));
        }

        // TODO I dont know if we should be replacing the worst with offspring but hey
        let mut offspring = self.crossover_networks(networks[0].0, networks[1].0, *networks[0].1, *networks[1].1);
        if !NeatNetwork::has_cycle(offspring.local_occupied_connections().iter()) {
            offspring.evaluate_fitness(fitness_function);
            self.networks[worst_performing.0] = offspring;
        }
    }

    /// Get the offspring of two networks
    pub fn crossover_networks(&self, mut network1: &NeatNetwork, mut network2: &NeatNetwork, fitness1: f32, fitness2: f32) -> NeatNetwork {
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
                    child_genes.push(gene1.clone());
                } else {
                    child_genes.push(gene2.clone());
                }
                i += 1;
                j += 1;
            } else {
                // Disjoint gene from net1_genes
                if fitness1 >= fitness2 {
                    child_genes.push(gene1.clone());
                }else {
                    child_genes.push(gene2.clone());
                }
                i += 1;
                j += 1;
            }
        }
        
        // Handle excess genes from the longer genome
        if fitness1 >= fitness2 {
            while i < net1_genes.len() {
                child_genes.push(net1_genes[i].clone());
                i += 1;
            }
        } else {
            while j < net2_genes.len() {
                child_genes.push(net2_genes[j].clone());
                j += 1;
            }
        }

        NeatNetwork::new_with_genes(
            network1.input_size(), network1.output_size(),
            self.global_innovation_number.clone(),
            self.global_occupied_connections.clone(),
            network1.activations(),
            child_genes,
            network1.network_config()
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

    /// Makes all networks in this species go through fitness
    /// function and store it for later use
    pub fn generate_fitness(&mut self, fitness_function: fn(&mut NeatNetwork) -> f32) -> () {
        for net in self.networks.iter_mut() {
            net.evaluate_fitness(fitness_function);
        }
    }

    /// Get the average score that the networks performed
    /// during the last fitness test
    pub fn previous_average_score(&self) -> f32 {
        self.previous_average_score
    }

    fn bottom_n_with_indices(numbers: &Vec<f32>, n: usize) -> Vec<usize> {
        let mut indexed_numbers: Vec<(usize, &f32)> = numbers.iter().enumerate().collect();
        indexed_numbers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed_numbers.into_iter().take(n).map(|(i, _)| i).collect()
    }
    fn top_n_with_indices(numbers: &Vec<f32>, n: usize) -> Vec<usize> {
        let mut indexed_numbers: Vec<(usize, &f32)> = numbers.iter().enumerate().collect();
        indexed_numbers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_numbers.into_iter().take(n).map(|(i, _)| i).collect()
    }

    fn generate_name() -> String {
        let prefixes = vec![
            "Quantum", "Sigma", "Big", "Fat", "Ugly", "Hawk", "Fentanyl overdosing",
            "Quandale", "Strong", "Obese", "Plus sized", "Drug addicted",
            "Child abusing", "Nerdy", "Holy", "Gay", "Fat ass", "Bitch ass", "Dumb",
            "Cute", "Petite", "Optimum", "Street"
        ];
        let suffixes = vec![
            "machine", "motor", "engine", "brain", "tuah", "ass", "combustion engine", "shitter", "dumpster",
            "iron ingot", "AI", "network", "genome",
        ];

        let mut rng = thread_rng();
        let prefix = prefixes[rng.gen_range(0..prefixes.len())];
        let suffix = suffixes[rng.gen_range(0..suffixes.len())];
        (prefix.to_string() + " " + suffix).to_string()
    }


    // Getters
    pub fn get_name(&self) -> &str {
        &self.name
    }

    
}

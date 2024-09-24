use core::f32;
use std::{collections::HashMap, sync::{Arc, Mutex}};
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};

use crate::neural_network::{activation::{Activation, NetworkActivations}, network::NeatNetwork};
use super::{config::{mutation::{GenomeMutationProbablities, WeightChangeProbablities}, network_config::NetworkConfig, stop_condition::StopCondition}, fitness::FitnessEvaluator, species::{Species, SPECIES_AVERAGE_SCORE_WINDOW_SIZE}};

const DEFAULT_SPECIES_SIZE: usize = 10;

/// Struct to make a set amount of networks
/// compete against eachother.
pub struct EvolutionBuilder<F: FitnessEvaluator> {
    /// How many networks that are going to compete
    /// against eachother.
    batch_size: Option<usize>,

    /// The amount of input neurons
    input_nodes: Option<usize>,

    /// The amount of output neurons
    output_nodes: Option<usize>,

    // TODO: DOC COMMENT?
    fitness_evaluator: Option<F>,

    /// How many networks (including representative) we should have
    /// in each species group (how many networks who compete against
    /// eachother each generation)
    species_size: usize,

    /// How often we replace the worst species with the best. (The
    /// higher number the less often we do that). None = no replacement
    replace_worst_every_nth_gen: Option<usize>,

    network_config: NetworkConfig,
    stop_condition: StopCondition,

    hidden_activation: Activation,
    output_activation: Activation,

    par_chunks_size: usize,
}

pub struct Evolution<F: FitnessEvaluator + Send + Sync> {
    /// All the diffrent networks that compete in groups
    species: Vec<Species>,
    fitness_evaluator: Arc<Mutex<F>>,
    global_innovation_number: Arc<Mutex<usize>>,
    stop_condition: StopCondition,
    generation: usize,
    species_size: usize,
    par_chunks_size: usize,
    replace_worst_every_nth_gen: Option<usize>,

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

impl<F: FitnessEvaluator + Send + Sync> EvolutionBuilder<F> {
    pub fn new() -> Self {
        Self {
            batch_size: None,
            output_nodes: None,
            input_nodes: None,
            fitness_evaluator: None,
            species_size: DEFAULT_SPECIES_SIZE,
            hidden_activation: Activation::LeakyRelu,
            output_activation: Activation::Sigmoid,
            stop_condition: StopCondition::default(),
            network_config: NetworkConfig::default(),
            par_chunks_size: 1,
            replace_worst_every_nth_gen: None
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

    /// Set the diffrent mutation probabilities for evolution
    pub fn mutation_probabilities(&mut self, prob: GenomeMutationProbablities) -> &mut Self { self.network_config.mutation_probabilities = prob; self }
    /// Set the diffrent mutation probabilities for evolution
    pub fn weight_change_probabilities(&mut self, prob: WeightChangeProbablities) -> &mut Self { self.network_config.weight_change_probabilities = prob; self }

    /// Every nth generation we'll replace the worst performing
    /// network with the best berforming so it can mutate in diffrent
    /// ways
    pub fn replace_worst_every_nth_gen(&mut self, nth: Option<usize>) -> &mut Self { self.replace_worst_every_nth_gen = nth; self }

    /// How big each chunk will be when multithreading looping
    /// through all species for running a generation. Default
    /// is 1. The par chunk size is the amount of species one
    /// thread will be handling at once. 
    /// 
    /// Therefore len(species) / par_chunk_size = "amount of threads"
    /// not actually but that's how ***theoretically*** many threads
    /// there will be
    pub fn par_chunks_size(&mut self, size: usize) -> &mut Self { self.par_chunks_size = size; self }

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
    pub fn set_fitness_evaluator(&mut self, eval: F) -> &mut Self { self.fitness_evaluator = Some(eval); self }

    /// Compile all set values and make this
    /// struct ready for evolution
    pub fn build(&mut self) -> Evolution<F> {
        assert!(self.batch_size.is_some(), "Batch is required for evolution");
        assert!(self.input_nodes.is_some(), "Input node amount is required for evolution");
        assert!(self.output_nodes.is_some(), "Output node amount is required for evolution");
        assert!(self.fitness_evaluator.is_some(), "Fitness evaluator is required for evolution");
        let batch_size = self.batch_size.unwrap();
        let input_nodes = self.input_nodes.unwrap();
        let output_nodes = self.output_nodes.unwrap();
        let species_size = self.species_size; // Default is `DEFAULT_SPECIES_SIZE`
        let hidden_activation = self.hidden_activation;
        let output_activation = self.output_activation;
        let activations = NetworkActivations::new(hidden_activation, output_activation);
        let network_config = Arc::new(self.network_config.clone());

        // Create species
        let mut species: Vec<Species> = Vec::with_capacity(batch_size);
        let global_occupied_connections = Arc::new(Mutex::new(HashMap::new()));
        let global_innovation_number = Arc::new(Mutex::new(0));

        for i in 0..batch_size {
            let representative = NeatNetwork::new(
                input_nodes,
                output_nodes,
                global_innovation_number.clone(),
                global_occupied_connections.clone(),
                activations,
                network_config.clone()
            );

            species.push(Species::new(
                global_innovation_number.clone(),
                global_occupied_connections.clone(),
                representative,
                species_size,
                i,
                false
            ));
        }

        Evolution {
            species,
            fitness_evaluator: Arc::new(Mutex::new(self.fitness_evaluator.clone().unwrap())),
            global_innovation_number,
            global_occupied_connections,
            stop_condition: self.stop_condition.clone(),
            generation: 0,
            species_size,
            par_chunks_size: self.par_chunks_size,
            replace_worst_every_nth_gen: self.replace_worst_every_nth_gen
        }
    }
}


impl<F: FitnessEvaluator + Send + Sync> Evolution<F> {
    pub fn new() -> EvolutionBuilder<F> {
        EvolutionBuilder::new()
    }

    /// Runs the networks through a generation of mutation, crossover, selection
    /// and more. Returns true if we should stop generating
    pub fn generation(&mut self) -> bool {
        self.generation += 1;
        
        // (species_fitness, species_index)
        let worst_performing = Arc::new(Mutex::new((f32::MAX, 0)));
        // (network_fitness, species_index, net_index)
        let best_performing = Arc::new(Mutex::new((f32::MIN, 0, 0)));
        let should_stop = Arc::new(Mutex::new(false));
        let should_replace = self.replace_worst_every_nth_gen.is_some();
        
        self.species.par_chunks_mut(self.par_chunks_size).for_each(|species_chunk| {
            for species in species_chunk {
                // Cache fitness in each network 
                // TODO
                species.generate_fitness(self.fitness_evaluator.clone());
                
                // Stop condition
                let previous_average = species.average_fitness();
                if self.stop_condition.should_stop(previous_average, self.generation) {
                    *should_stop.lock().unwrap() = true;
                }

                // Cross-over. The genome will also be running `evaluate_fitness`
                // before inserted into the species, therefore it's guaranteed
                // that all genomes in this species will have correct previous
                // fitnesses.
                //
                // We find do mod by SPECIES_AVERAGE_SCORE_WINDOW_SIZE to wait
                // for the offspring to fully "fill up" its fitness window which
                // leads to better fitness representation
                if self.generation % SPECIES_AVERAGE_SCORE_WINDOW_SIZE == 0 {
                    // TODO
                    // species.crossover(self.fitness_function);
                }

                // Mutate
                species.compute_generation();

                // Find best and worst
                if should_replace && self.generation % self.replace_worst_every_nth_gen.unwrap() == 0 {
                    for (net_index, net) in species.networks().iter().enumerate() {
                        let net_fitness = net.previous_average_fitness();
                        let mut best_perf = best_performing.lock().unwrap();

                        /* Find best network */
                        if net_fitness > best_perf.0 {
                            *best_perf = (net_fitness, species.index(), net_index)
                        }
                    }

                    /* Find worst species index */
                    let mut worst_perf = worst_performing.lock().unwrap();
                    let species_fitness = species.average_fitness();
                    if species_fitness < worst_perf.0 {
                        *worst_perf = (previous_average, species.index());
                    }
                }
            }
        });

        if should_replace {
            self.replace_least_fit(worst_performing, best_performing);
        }
        
        *should_stop.clone().lock().unwrap()
    }

    fn replace_least_fit(&mut self, worst_species: Arc<Mutex<(f32, usize)>>, best_network: Arc<Mutex<(f32, usize, usize)>>) -> () {
        if self.generation % self.replace_worst_every_nth_gen.unwrap() != 0 { return };

        let best_network = best_network.lock().unwrap();
        let worst_species = worst_species.lock().unwrap();

        println!("REPLACING SPECIES {} with fitness {}", worst_species.1, self.species[worst_species.1].average_fitness());
        let best_network = &self.species()[best_network.1].networks()[best_network.2];
        self.species[worst_species.1] = Species::new(
            self.global_innovation_number.clone(),
            self.global_occupied_connections.clone(),
            best_network.clone(),
            self.species_size,
            worst_species.1,
            true
        );
    }

    pub fn average_fitness(&self) -> f32 {
        self.species.iter().map(|e| e.average_fitness()).sum::<f32>() / self.species.len() as f32
    }

    /// Returns a reference to all species
    pub fn species(&self) -> &Vec<Species> {
        &self.species
    }

    pub fn get_generation(&self) -> usize {
        self.generation
    }
}

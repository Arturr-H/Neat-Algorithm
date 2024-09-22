use std::sync::Arc;
use neat_algorithm::neural_network::{activation::{Activation, NetworkActivations}, connection_gene::ConnectionGene, network::NeatNetwork};

#[test]
fn initialize_default() -> () {
    let activations = NetworkActivations::new(Activation::LeakyRelu, Activation::LeakyRelu);
    let net = NeatNetwork::new(1, 2, Arc::default(), Arc::default(), activations, Arc::default());
    assert!(net.input_size() == 1);
    assert!(net.output_size() == 2);
    assert!(net.is_input(0));
    assert!(net.is_output(1) && net.is_output(2));
    assert!(net.get_genes().len() == 2);
}

#[test]
fn initialize_with_genes() -> () {
    let activations = NetworkActivations::new(Activation::LeakyRelu, Activation::LeakyRelu);
    let genes = vec![ConnectionGene::new(0, 1, 1., 0)];
    let net = NeatNetwork::new_with_genes(1, 2, Arc::default(), Arc::default(), activations, genes, Arc::default());
    assert!(net.input_size() == 1);
    assert!(net.output_size() == 2);
    assert!(net.is_input(0));
    assert!(net.is_output(1) && net.is_output(2));
    assert!(net.get_genes().len() == 1);
}

#[test]
fn calculate_output() -> () {
    let activations = NetworkActivations::new(Activation::LeakyRelu, Activation::LeakyRelu);
    let genes = vec![ConnectionGene::new(0, 1, 1., 0), ConnectionGene::new(0, 2, 1., 1)];
    let mut net = NeatNetwork::new_with_genes(1, 2, Arc::default(), Arc::default(), activations, genes, Arc::default());
    assert!(net.calculate_output_cached_topology(vec![1.], false) == vec![1.1, 1.1]);
}

#[test]
fn topology_sort() -> () {
    let activations = NetworkActivations::new(Activation::LeakyRelu, Activation::LeakyRelu);
    let genes = vec![ConnectionGene::new(0, 1, 1., 0), ConnectionGene::new(0, 2, 1., 1)];
    let net = NeatNetwork::new_with_genes(1, 2, Arc::default(), Arc::default(), activations, genes, Arc::default());
    assert!(net.topological_sort() == Some(vec![0, 2, 1]));
}

#[test]
fn fitness() -> () {
    let activations = NetworkActivations::new(Activation::LeakyRelu, Activation::LeakyRelu);
    let mut net = NeatNetwork::new(1, 2, Arc::default(), Arc::default(), activations, Arc::default());
    net.evaluate_fitness(|_| 1.);
    assert!(net.previous_fitness() == 1.);
}

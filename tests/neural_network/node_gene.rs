use neat_algorithm::neural_network::node_gene::{NodeGene, NodeGeneType};

#[test]
fn initialize() -> () {
    let ng = NodeGene::new(NodeGeneType::Input, 0.0);
    assert!(ng.x() == 0.);
    assert!(ng.activation() == 0.);
    assert!(ng.incoming_connection_indexes().is_empty());
}

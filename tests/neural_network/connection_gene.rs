use neat_algorithm::{neural_network::connection_gene::ConnectionGene, trainer::config::mutation::WeightChangeProbablities};

#[test]
fn initialize() -> () {
    let conn = ConnectionGene::new(0, 1, 0.5, 0);

    assert!(conn.enabled());
    assert!(conn.innovation_number() == 0);
    assert!(conn.weight() == 0.5);
    assert!(conn.node_in() == 0);
    assert!(conn.node_out() == 1);
}

#[test]
fn toggle() -> () {
    let mut conn = ConnectionGene::new(0, 1, 0.5, 0);
    conn.set_enabled(false);
    assert!(conn.enabled() == false);
    conn.set_enabled(true);
    assert!(conn.enabled() == true);
}

#[test]
fn mutate() -> () {
    let mut conn = ConnectionGene::new(0, 1, 0.5, 0);
    /* Change sign */
    conn.mutate_weight(&WeightChangeProbablities {
        addition_small: 0,
        addition_large: 0,
        multiplication_small: 0,
        multiplication_large: 0,
        change_sign: 1
    });
    assert!(conn.weight() == -0.5);
    let weight = conn.weight();

    /* Change weight */
    conn.mutate_weight(&WeightChangeProbablities {
        addition_small: 1,
        addition_large: 1,
        multiplication_small: 1,
        multiplication_large: 1,
        change_sign: 0
    });

    /* This has a low risk of failing if
        random happens to to change nothing */
    assert!(conn.weight() != weight);
}

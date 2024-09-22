use neat_algorithm::neural_network::activation::{
    NetworkActivations, Activation,
    sigmoid, relu, leaky_relu
};

#[test]
fn network_activations_init() -> () {
    let net_act = NetworkActivations::new(Activation::Relu, Activation::Linear);
    assert!(matches!(net_act.hidden, Activation::Relu));
    assert!(matches!(net_act.output, Activation::Linear));
}

#[test]
fn negative_activation() -> () {
    assert!(sigmoid(&vec![-1.], 0) < 0.3);
    assert!(relu(&vec![-1.], 0) == 0.);
    assert!(leaky_relu(&vec![-1.], 0) < 0.);
}

#[test]
fn neutral_activation() -> () {
    assert!(sigmoid(&vec![0.0], 0) == 0.5);
    assert!(relu(&vec![0.0], 0) == 0.);
    assert!(leaky_relu(&vec![0.0], 0) == 0.);
}

#[test]
fn positive_activation() -> () {
    assert!(sigmoid(&vec![1.0], 0) >= 0.7);
    assert!(relu(&vec![1.0], 0) == 1.);
    assert!(leaky_relu(&vec![1.0], 0) == 1.);
}

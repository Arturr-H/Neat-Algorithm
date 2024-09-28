/* Imports */
use neat_algorithm::{
    debug::display::start_debug_display,
    games::snake::SnakeGameEvaluator,
    neural_network::activation::Activation,
    trainer::{
        config::mutation::{
            GenomeMutationProbablities,
            WeightChangeProbablities
        },
        evolution::Evolution
    }
};

fn main() -> () {
    let evolution = Evolution::new()
        .batch_size(25)
        .with_species_size(6)
        .with_input_nodes(8)
        .with_output_nodes(4)
        .mutation_probabilities(GenomeMutationProbablities {
            split_connection: 2,
            create_connection: 5,
            change_weight: 850,
            toggle_weight: 1,
            nothing: 20,
        })
        .weight_change_probabilities(WeightChangeProbablities {
            addition_small: 25,
            addition_large: 5,
            multiplication_small: 25,
            multiplication_large: 5,
            change_sign: 3
        })
        .replace_worst_every_nth_gen(Some(100))
        .with_hidden_activation(Activation::LeakyRelu)
        .with_output_activation(Activation::Sigmoid)
        .preestablish_connections(true)
        .set_fitness_evaluator(SnakeGameEvaluator)
        .build();

    // let mut net = NeatNetwork::retrieve("/Users/artur/Desktop/snakeman").unwrap();
    // SnakeGame::score_game(&mut net, 1200, true);
    start_debug_display(evolution);
}

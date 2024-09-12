use rand::{thread_rng, Rng};

/// A connection between two `NodeGenes`
pub struct ConnectionGene {
    /// The index / ID of some node
    node_in: usize,
    node_out: usize,

    /// The strength of the bond between the 
    /// two nodes.
    weight: f32,

    /// Used for disabling some connections
    /// and turning them back on later in the evo
    enabled: bool,

    /// The local innovation number
    innovation_number: usize,
}

impl ConnectionGene {
    pub fn new(node_in: usize, node_out: usize, weight: f32, innovation_number: usize) -> Self {
        Self { node_in, node_out, weight, enabled: true, innovation_number }
    }
    
    // Getters
    pub fn node_in(&self) -> usize { self.node_in }
    pub fn node_out(&self) -> usize { self.node_out }
    pub fn weight(&self) -> f32 { self.weight }
    pub fn enabled(&self) -> bool { self.enabled }
    pub fn innovation_number(&self) -> usize { self.innovation_number }

    // Setters
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled
    }

    pub fn mutate_weight(&mut self) -> () {
        let mut rng = thread_rng();

        match rng.gen_range(0..3) {
            // Add or subtract to weight
            0 => self.weight += rng.gen_range(-1.0..1.0),
            // Multiply weight
            1 => self.weight *= rng.gen_range(0.5..1.5),
            // Change sign
            2 => self.weight *= -1.,
            _ => unreachable!()
        }
    }
}

impl std::fmt::Debug for ConnectionGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.enabled {
            write!(f, "Conn({}-{} ({}))", self.node_in, self.node_out, self.weight)
        }else {
            write!(f, "Conn(...)")
        }
    }
}

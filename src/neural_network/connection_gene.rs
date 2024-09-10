
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
    enabled: bool
}

impl ConnectionGene {
    pub fn new(node_in: usize, node_out: usize, weight: f32) -> Self {
        Self { node_in, node_out, weight, enabled: true }
    }
    
    // Getters
    pub fn node_in(&self) -> usize { self.node_in }
    pub fn node_out(&self) -> usize { self.node_out }
    pub fn weight(&self) -> f32 { self.weight }
    pub fn enabled(&self) -> bool { self.enabled }
}

impl std::fmt::Debug for ConnectionGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Conn({}-{} ({:.2}) {{{}}})", self.node_in, self.node_out, self.weight, self.enabled)
    }
}

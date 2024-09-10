
/// A connection between two `NodeGenes`
pub struct ConnectionGene {
    /// The index / ID of some node
    node1: usize,
    node2: usize,

    /// The strength of the bond between the 
    /// two nodes.
    weight: f32,

    /// Used for disabling some connections
    /// and turning them back on later in the evo
    enabled: bool
}

impl ConnectionGene {
    pub fn new(node1: usize, node2: usize, weight: f32) -> Self {
        Self { node1, node2, weight, enabled: true }
    }
}

impl std::fmt::Debug for ConnectionGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Conn({}-{} ({:.2}) {{{}}})", self.node1, self.node2, self.weight, self.enabled)
    }
}

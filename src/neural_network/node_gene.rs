
/// Input nodes are the ones that recieve data first, 
/// then nodes will propagate data forward until it
/// may reach an output node
pub enum NodeGeneType {
    Input, Ouptut, Regular
}

pub struct NodeGene {
    /// The ID of this node gene, also the index
    /// of the array which this node is located in
    id: usize,

    /// Biases are stored inside of each node gene
    /// instead of per layers, because NEAT networks
    /// don't really have hidden layers
    bias: f32,

    /// Each type of node, as output, input or "dynamic"
    /// hidden nodes are stored in the same place. 
    node_type: NodeGeneType,
}

impl NodeGene {
    pub fn new(index: usize, node_type: NodeGeneType) -> Self {
        Self { id: index, bias: 0.1, node_type }
    }
}

impl std::fmt::Debug for NodeGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_str = match self.node_type {
            NodeGeneType::Input => "I",
            NodeGeneType::Ouptut => "O",
            NodeGeneType::Regular => "R",
        };

        write!(f, "Node({}{})", type_str, self.id)
    }
}

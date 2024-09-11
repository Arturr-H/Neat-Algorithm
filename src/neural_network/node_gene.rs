/// Input nodes are the ones that recieve data first, 
/// then nodes will propagate data forward until it
/// may reach an output node

#[derive(PartialEq, Eq, Clone, Copy)]
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

    /// Used for forward prop so we won't need to re-
    /// calculate every time
    activation: f32,

    /// Indexes of connections where node_in is the 
    /// `id` field in this struct.
    outgoing_connection_indexes: Vec<usize>
}

impl NodeGene {
    pub fn new(index: usize, node_type: NodeGeneType) -> Self {
        Self {
            id: index,
            bias: 0.1,
            node_type,
            activation: 0.,
            outgoing_connection_indexes: Vec::new()
        }
    }

    // Getters
    pub fn id(&self) -> usize { self.id }
    pub fn bias(&self) -> f32 { self.bias }
    pub fn node_type(&self) -> NodeGeneType { self.node_type }
    pub fn activation(&self) -> f32 { self.activation }
    pub fn outgoing_connection_indexes(&self) -> &Vec<usize> { &self.outgoing_connection_indexes }

    // Setters
    pub fn set_activation(&mut self, to: f32) -> () { self.activation = to; }

    /// Appends a new outgoing connection gene to the list
    pub fn register_new_outgoing(&mut self, index: usize) -> () {
        assert!(!self.outgoing_connection_indexes.contains(&index));
        self.outgoing_connection_indexes.push(index);
    }
}

impl std::fmt::Debug for NodeGene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({:?}{} ({:.3}))", self.node_type, self.id, self.activation)
    }
}
impl std::fmt::Debug for NodeGeneType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_str = match self {
            NodeGeneType::Input => "I",
            NodeGeneType::Ouptut => "O",
            NodeGeneType::Regular => "R",
        };

        write!(f, "{type_str}")
    }
}

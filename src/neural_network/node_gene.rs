use serde_derive::{Serialize, Deserialize};

/// Input nodes are the ones that recieve data first, 
/// then nodes will propagate data forward until it
/// may reach an output node
#[derive(PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum NodeGeneType {
    Input, Output, Regular
}

#[derive(Clone, Serialize, Deserialize)]
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

    /// Indexes of connections where node_out is the 
    /// `id` field in this struct.
    incoming_connection_indexes: Vec<usize>,

    /// Debug visualization. X is a value between 0 and 1
    x: f32
}

impl NodeGene {
    pub fn new(index: usize, node_type: NodeGeneType, x: f32) -> Self {
        Self {
            id: index,
            bias: 0.1,
            node_type,
            activation: 0.,
            incoming_connection_indexes: Vec::new(),
            x
        }
    }

    // Getters
    pub fn x(&self) -> f32 { self.x }
    pub fn id(&self) -> usize { self.id }
    pub fn bias(&self) -> f32 { self.bias }
    pub fn node_type(&self) -> NodeGeneType { self.node_type }
    pub fn activation(&self) -> f32 { self.activation }
    pub fn incoming_connection_indexes(&self) -> &Vec<usize> { &self.incoming_connection_indexes }
    pub fn is_indegree_zero(&self) -> bool { self.incoming_connection_indexes.is_empty() }

    // Setters
    pub fn set_activation(&mut self, to: f32) -> () { self.activation = to; }
    pub fn set_x(&mut self, to: f32) -> () { self.x = to; }

    /// Appends a new incoming connection gene to the list
    pub fn register_new_incoming(&mut self, index: usize) -> () {
        assert!(!self.incoming_connection_indexes.contains(&index));
        self.incoming_connection_indexes.push(index);
    }
    pub fn set_incoming_indexes(&mut self, to: Vec<usize>) -> () {
        self.incoming_connection_indexes = to;
    }
    pub fn remove_incoming(&mut self, index: usize) -> () {
        self.incoming_connection_indexes.remove(index);
    }

    /// Display all info about this gene
    pub fn verbose_debug(&self) -> String {
        format!(
            "Incoming: {:?}, Bias: {}, Activation: {}, Id: {}, Type: {:?}",
            self.incoming_connection_indexes, self.bias,
            self.activation, self.id, self.node_type
        )
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
            NodeGeneType::Output => "O",
            NodeGeneType::Regular => "R",
        };

        write!(f, "{type_str}")
    }
}

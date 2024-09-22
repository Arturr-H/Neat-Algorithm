/// When to stop running generations.
#[derive(Clone)]
pub struct StopCondition {
    conditions: Vec<(Chain, StopConditionType)>
}

#[derive(Clone)]
pub enum StopConditionType {
    /// After a set amount of generations.
    GenerationsReached(usize),

    /// After some species has reached this value
    /// as an average fitness score for all nets.
    FitnessReached(f32)
}

#[derive(Clone)]
pub enum Chain {
    Or,
    And,
}

impl StopCondition {
    pub fn after(after: StopConditionType) -> StopCondition {
        StopCondition {
            conditions: vec![(Chain::Or, after)]
        }
    }

    pub fn chain(mut self, chain: Chain, after: StopConditionType) -> Self {
        self.conditions.push((chain, after));
        self
    }

    pub fn should_stop(&self, fitness: f32, generations: usize) -> bool {
        let mut should_stop = false;

        for (chain, condition) in &self.conditions {
            let local_condition = match condition {
                StopConditionType::FitnessReached(req) => &fitness >= req,
                StopConditionType::GenerationsReached(req) => &generations >= req,
            };

            match chain {
                Chain::Or => should_stop = should_stop || local_condition,
                Chain::And => should_stop = should_stop && local_condition,
            }
        }

        should_stop
    }
}

impl Default for StopCondition {
    fn default() -> Self {
        Self { conditions: Vec::new() }        
    }
}

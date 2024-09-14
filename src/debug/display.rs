use std::collections::HashMap;

/* Imports */
use eframe::egui::{self, Painter};
use crate::neural_network::{connection_gene::ConnectionGene, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}};

/* Constants */
const NODE_SIZE: f32 = 10.;

pub fn start_debug_display(network: &NeatNetwork) -> () {
    let net = network.clone();
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Neat Network",
        options,
        Box::new(|_cc| Ok(Box::new(net))),
    ).unwrap();
}

impl eframe::App for NeatNetwork {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            
            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height());
            let padding = 80.;
            let mut positions: Vec<(f32, f32)> = Vec::with_capacity(self.node_genes().len());

            for input_index in 0..self.input_size() {
                let step_size = (h - padding * 2.) / (self.input_size() - 1) as f32;
                let y = step_size * input_index as f32 + padding;
                positions.push((padding, y));
                painter.circle_filled(egui::pos2(padding, y), NODE_SIZE, egui::Color32::YELLOW);
            }

            let topo = self.topological_sort().unwrap();
            let connections = self.get_genes();

            // <Nodes until output, (node indexes)>
            let mut yhash: HashMap<usize, Vec<usize>> = HashMap::new();
            for i in &topo {
                // Skip inputs and outputs
                if i < &(self.input_size() + self.output_size()) { continue; }

                let node = &self.node_genes()[*i];
                let next = get_next_nodes(&connections, self.node_genes(), node);
                let nodes_until_output = search(&connections, &next, 1).unwrap_or(0);

                match yhash.get_mut(&nodes_until_output) {
                    Some(e) => {
                        e.push(*i);
                    },
                    None => { yhash.insert(nodes_until_output, vec![*i]); }
                };
            }

            let padding_x = 180.;
            let padding_y = 80.;
            for (until_output, nodes) in yhash.iter() {
                let step_size_y = (h - padding_y * 2.) / (nodes.len() - 1) as f32;
                let step_size_x = (w - padding_x * 2.) / (yhash.len() - 1) as f32;

                for (index, node) in nodes.iter().enumerate() {
                    let x = *until_output as f32 * step_size_x + padding_x;
                    let y = index as f32 * step_size_y + padding_y;
                    positions.push((x, y));
                    painter.circle_filled(egui::pos2(x, y), NODE_SIZE, egui::Color32::BLUE);
                }
            }

            for output_index in 0..self.output_size() {
                let step_size = (h - padding * 2.) / (self.output_size() - 1) as f32;
                let y = step_size * output_index as f32 + padding;
                positions.push((w - padding, y));
                painter.circle_filled(egui::pos2(w - padding, y), NODE_SIZE, egui::Color32::GREEN);
            }

            for conn in self.get_genes() {
                let input_coord = positions[conn.node_in()];
                let output_coord = positions[conn.node_out()];

                painter.line_segment(
                    [egui::pos2(input_coord.0, input_coord.1), egui::pos2(output_coord.0, output_coord.1)],
                    (1.0, egui::Color32::WHITE),
                );
            }
        });
    }
}

fn search(conns: &Vec<ConnectionGene>, nodes: &Vec<NodeGene>, nodes_until_output: usize) -> Option<usize> {
    for node in nodes {
        match node.node_type() {
            NodeGeneType::Ouptut => return Some(nodes_until_output),
            _ => ()
        };

        let next_nodes = get_next_nodes(&conns, &nodes, node);
        match search(&conns, &next_nodes, nodes_until_output + 1) {
            Some(e) => return Some(e),
            None => ()
        };
    }
    
    None
}

fn get_next_nodes(conns: &Vec<ConnectionGene>, vec: &Vec<NodeGene>, from: &NodeGene) -> Vec<NodeGene> {
    let mut end = Vec::new();

    for node in vec {
        for conn_idx in node.incoming_connection_indexes() {
            let conn = &conns[*conn_idx];
            if conn.node_in() == from.id() {
                end.push(node.clone());
            }
        }
    }

    end
}


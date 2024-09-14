use std::{collections::HashMap, hash::{DefaultHasher, Hash, Hasher}, os::windows::thread};

/* Imports */
use eframe::egui::{self, Key, Painter};
use rand::{thread_rng, Rng};
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

            if ctx.input(|i| i.key_down(Key::Space)) {
                self.mutate();
            }
            
            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height());
            let padding = 80.;
            
            let mut positions: HashMap<usize, egui::Pos2> = HashMap::new();

            for input_index in 0..self.input_size() {
                let step_size = (h - padding * 2.0) / (self.input_size() - 1) as f32;
                let y = step_size * input_index as f32 + padding;
                let pos = egui::pos2(padding, y);
                positions.insert(input_index, pos);
                painter.circle_filled(pos, NODE_SIZE, egui::Color32::YELLOW);
            }

            let topo = self.topological_sort().unwrap();
            let connections = self.get_genes();
            let hidden_size = topo.len() - self.input_size() - self.output_size();


            let mut yhash: HashMap<usize, Vec<usize>> = HashMap::new();
            for &i in &topo {
                if i < self.input_size() || i >= self.input_size() + hidden_size { continue; }
                let node = &self.node_genes()[i];
                let next = get_next_nodes(&connections, self.node_genes(), node);
                let next_nodes: Vec<NodeGene> = next.into_iter().cloned().collect();
                let nodes_until_output = search(&connections, &next_nodes, 1).unwrap_or(0);
                yhash.entry(nodes_until_output).or_insert_with(Vec::new).push(i);
            }

            let padding_x = 180.0;
            let padding_y = 80.0;
            let max_layer = yhash.keys().max().unwrap_or(&0) + 0;

            let output_space_ratio = 0.2;
            let hidden_space = w - padding_x * 2.0;
            let hidden_space_end = w - padding_x - (hidden_space * output_space_ratio);

            //hidden nodes
            for (layer, nodes) in yhash.iter() {
                let step_size_y = (h - padding_y * 2.0) / nodes.len().max(1) as f32;
                let x = (*layer as f32 / max_layer as f32) * (hidden_space_end - padding_x) + padding_x;

                for (index, &node_id) in nodes.iter().enumerate() {
                    let y = index as f32 * step_size_y + padding_y;
                    let pos = egui::pos2(x, y);
                    positions.insert(node_id, pos);
                    painter.circle_filled(pos, NODE_SIZE, egui::Color32::BLUE);
                }
            }

            // Output nodes
            for output_index in 0..self.output_size() {
                let step_size = (h - padding * 2.0) / (self.output_size() - 1) as f32;
                let y = step_size * output_index as f32 + padding;
                let pos = egui::pos2(w - padding, y);
                positions.insert(self.input_size() + hidden_size + output_index, pos);
                painter.circle_filled(pos, NODE_SIZE, egui::Color32::RED);
            }

            for conn in self.get_genes() {
                if let (Some(&input_pos), Some(&output_pos)) = (positions.get(&conn.node_in()), positions.get(&conn.node_out())) {
                    // let color = innovation_to_color(conn.innovation_number());
                    let color = match conn.enabled() {
                        true => egui::Color32::GREEN,
                        false => egui::Color32::from_rgba_unmultiplied(255, 0, 0, 21)
                    };
                    painter.line_segment(
                        [input_pos, output_pos],
                        (1.0, color),
                    );
                }
            }
        });
    }
}

fn innovation_to_color(innovation_number: usize) -> egui::Color32 {
    // Use a hash function to get a consistent "random" number from the innovation number
    let mut hasher = DefaultHasher::new();
    innovation_number.hash(&mut hasher);
    let hash = hasher.finish();

    // Use the hash to generate RGB values
    let r = ((hash & 0xFF0000) >> 16) as u8;
    let g = ((hash & 0x00FF00) >> 8) as u8;
    let b = (hash & 0x0000FF) as u8;

    egui::Color32::from_rgb(r, g, b)
}

fn search(conns: &[ConnectionGene], nodes: &[NodeGene], nodes_until_output: usize) -> Option<usize> {
    for node in nodes {
        if let NodeGeneType::Ouptut = node.node_type() {
            return Some(nodes_until_output);
        }
        let next_nodes = get_next_nodes(conns, nodes, node);
        let next_nodes_slice: Vec<NodeGene> = next_nodes.into_iter().cloned().collect();
        if let Some(result) = search(conns, &next_nodes_slice, nodes_until_output + 1) {
            return Some(result);
        }
    }
    None
}

fn get_next_nodes<'a>(conns: &[ConnectionGene], vec: &'a [NodeGene], from: &NodeGene) -> Vec<&'a NodeGene> {
    vec.iter()
        .filter(|node| node.incoming_connection_indexes()
            .iter()
            .any(|&conn_idx| conns[conn_idx].node_in() == from.id()))
        .collect()
}
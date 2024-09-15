use std::{collections::HashMap, hash::{DefaultHasher, Hash, Hasher}};

/* Imports */
use eframe::egui::{self, Color32, Key, Painter};
use rand::{thread_rng, Rng};
use crate::neural_network::{connection_gene::ConnectionGene, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}};

/* Constants */
const NODE_SIZE: f32 = 5.;

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

            if ctx.input(|i| i.key_released(Key::Space)) {
                self.mutate();
                ctx.request_repaint(); // ! does not fix it
                println!("{:?}", self.node_genes());
            }
            
            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height());
            let padding: f32 = 80.;
            
            let mut positions: HashMap<u32, Vec<usize>> = HashMap::new();
            for (index, node_gene) in self.node_genes().iter().enumerate() {
                // f32 does not implement hash so we transmute
                // it to a u32
                let int = unsafe { std::mem::transmute::<f32, u32>(node_gene.x()) };
                match positions.get(&int) {
                    Some(e) => {
                        positions.get_mut(&int).unwrap().push(index);
                    },
                    None => {
                        positions.insert(int, vec![index]);
                    },
                }
            }

            let mut k: Vec<&u32> = positions.keys().collect();
            k.sort();
            let adjusted_h = h - padding*2.;
            let adjusted_w = w - padding*2.;
            let mut node_positions: HashMap<usize, (f32, f32)> = HashMap::new();

            for i in k {
                let v: &Vec<usize> = &positions[&i];
                let amount_of_nodes = v.len();
                let x_factor = unsafe { std::mem::transmute::<u32, f32>(*i) };

                for (y_fac, node_index) in v.iter().enumerate() {
                    let node = &self.node_genes()[*node_index];
                    let x = padding + adjusted_w * x_factor;
                    let y_steps = adjusted_h / ((amount_of_nodes - 1).max(1)) as f32;
                    let y = padding + (y_fac as f32) * y_steps + if amount_of_nodes == 1 { adjusted_h / 2. } else { 0. };

                    println!("{:?} {} {}", node, x, y);

                    node_positions.insert(*node_index, (x, y));
                    let col = match node.node_type() {
                        NodeGeneType::Input => Color32::YELLOW,
                        NodeGeneType::Regular => Color32::WHITE,
                        NodeGeneType::Ouptut => Color32::BLUE,
                    };

                    painter.circle_filled(egui::Pos2 { x, y }, NODE_SIZE, col);
                }
            }

            for conn in self.get_genes() {
                let n_in = conn.node_in();
                let n_out = conn.node_out();
                let (x1, y1) = node_positions[&n_in];
                let (x2, y2) = node_positions[&n_out];
                let color = match conn.enabled() {
                    true => Color32::WHITE,
                    false => Color32::RED
                };

                painter.line_segment(
                    [(x1, y1).into(), (x2, y2).into()],
                    (0.4, color),
                );
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
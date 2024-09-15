use std::{collections::HashMap, hash::{DefaultHasher, Hash, Hasher}};

/* Imports */
use eframe::egui::{self, pos2, Align2, Color32, FontId, Key, Painter};
use rand::{thread_rng, Rng};
use crate::{neural_network::{connection_gene::ConnectionGene, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}}, trainer::species::Species};

/* Constants */
const NODE_SIZE: f32 = 5.;

struct Networks(Species);
pub fn start_debug_display(networks: Species) -> () {
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Neat Network",
        options,
        Box::new(|_cc| Ok(Box::new(Networks(networks)))),
    ).unwrap();
}

impl eframe::App for Networks {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();

            if ctx.input(|i| i.key_down(Key::Space)) {
                for i in self.0.networks_mut().iter_mut() {
                    i.mutate();
                }
            }
            
            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height());
            let padding: f32 = 80.;

            let w_h_ratio = h / w;
            let amount_of_networks = self.0.networks().len();
            let cols = (amount_of_networks as f32).sqrt() as usize; // Number of columns
            let rows = (amount_of_networks + cols - 1) / cols;    // Number of rows
            
            let cell_width = w / cols as f32;
            let cell_height = h / rows as f32;

            for col in 0..cols + 1 {
                let y = cell_height * col as f32;

                painter.line_segment(
                    [(0., y).into(), (w, y).into()],
                    (0.2, Color32::GRAY),
                );
            }
            for row in 0..rows + 1 {
                let x = cell_width * row as f32;

                painter.line_segment(
                    [(x, 0.).into(), (x, h).into()],
                    (0.2, Color32::GRAY),
                );
            }
            
            let mut coordinates = Vec::new();
            for row in 0..rows {
                for col in 0..cols {
                    let x = col as f32 * cell_width;
                    let y = row as f32 * cell_height;
                    coordinates.push((x, y));
                }
            }
            
            for (index, network) in self.0.networks().iter().enumerate() {
                let coordinate = coordinates[index];
                painter.text(pos2(coordinate.0 + cell_width / 2., coordinate.1 + 40.), Align2::CENTER_CENTER, format!("{} ({})", self.0.get_name(), network.fitness()), FontId::default(), Color32::WHITE);
                let mut positions: HashMap<u32, Vec<usize>> = HashMap::new();
                for (index, node_gene) in network.node_genes().iter().enumerate() {
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
                let adjusted_h = cell_height - padding*2.;
                let adjusted_w = cell_width - padding*2.;
                let mut node_positions: HashMap<usize, (f32, f32)> = HashMap::new();

                for i in k {
                    let v: &Vec<usize> = &positions[&i];
                    let amount_of_nodes = v.len();
                    let x_factor = unsafe { std::mem::transmute::<u32, f32>(*i) };

                    for (y_fac, node_index) in v.iter().enumerate() {
                        let node = &network.node_genes()[*node_index];
                        let x = padding + adjusted_w * x_factor
                            + coordinate.0;
                        let y_steps = adjusted_h / ((amount_of_nodes - 1).max(1)) as f32;
                        let y = padding + (y_fac as f32) * y_steps
                            + if amount_of_nodes == 1 { adjusted_h / 2. } else { 0. }
                            + coordinate.1;

                        node_positions.insert(*node_index, (x, y));
                        let col = match node.node_type() {
                            NodeGeneType::Input => Color32::GREEN,
                            NodeGeneType::Regular => Color32::WHITE,
                            NodeGeneType::Output => Color32::BLUE,
                        };

                        painter.circle_filled(egui::Pos2 { x, y }, NODE_SIZE, col);
                    }
                }

                for conn in network.get_genes() {
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
                        (conn.weight().max(0.2), color),
                    );
                }
            }
        });
    }
}

fn search(conns: &[ConnectionGene], nodes: &[NodeGene], nodes_until_output: usize) -> Option<usize> {
    for node in nodes {
        if let NodeGeneType::Output = node.node_type() {
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
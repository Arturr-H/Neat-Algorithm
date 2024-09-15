use std::{collections::HashMap, hash::{DefaultHasher, Hash, Hasher}};

/* Imports */
use eframe::egui::{self, pos2, Align2, Color32, FontId, Key, Painter, Rect, Ui};
use rand::{thread_rng, Rng};
use crate::{neural_network::{connection_gene::ConnectionGene, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}}, score_network, trainer::{evolution::Evolution, species::Species}};

/* Constants */
const NODE_SIZE: f32 = 5.;

struct DrawContext {
    evolution: Evolution,
    species_index: usize,
    tab_height: f32,
}
pub fn start_debug_display(evolution: Evolution) -> () {
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Neat Network",
        options,
        Box::new(|_cc| Ok(Box::new(DrawContext {
            evolution,
            species_index: 0,
            tab_height: 50.
        }))),
    ).unwrap();
}

impl eframe::App for DrawContext {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();

            draw_top_tab(self, ctx, _frame, painter);
            draw_bottom_tab(self, ctx, _frame, painter);

            if ctx.input(|i| i.key_pressed(Key::Space)) {
                // let best_parent = self.0.networks().iter().max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap()).unwrap().clone();

                self.evolution.generation();
            }
            if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) {
                self.species_index = self.species_index.checked_sub(1).unwrap_or(0);
            }else if ctx.input(|i| i.key_pressed(Key::ArrowRight)) {
                self.species_index = (self.species_index + 1).min(self.evolution.species().len() - 1);
            }
            
            let tab_height = self.tab_height;

            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height() - tab_height);
            let padding: f32 = 80.;

            let w_h_ratio = h / w;
            let networks = self.evolution.species()[self.species_index].networks();
            let species_name = self.evolution.species()[self.species_index].get_name();
            let amount_of_networks = networks.len();
            let cols = (amount_of_networks as f32).sqrt() as usize; // Number of columns
            let rows = (amount_of_networks + cols - 1) / cols;    // Number of rows
            
            let cell_width = w / cols as f32;
            let cell_height = (h - tab_height) / rows as f32;

            for col in 0..cols + 2 {
                let y = cell_height * col as f32 + tab_height;

                painter.line_segment(
                    [(0., y).into(), (w, y).into()],
                    (0.2, Color32::GRAY),
                );
            }
            for row in 0..rows + 2 {
                let x = cell_width * row as f32;

                painter.line_segment(
                    [(x, tab_height).into(), (x, h).into()],
                    (0.2, Color32::GRAY),
                );
            }
            
            let mut coordinates = Vec::new();
            for row in 0..rows {
                for col in 0..cols {
                    let x = col as f32 * cell_width;
                    let y = row as f32 * cell_height + tab_height;
                    coordinates.push((x, y));
                }
            }
            
            for (index, network) in networks.iter().enumerate() {
                let coordinate = coordinates[index];
                painter.text(pos2(coordinate.0 + cell_width / 2., coordinate.1 + 40.), Align2::CENTER_CENTER, format!("{} ({})", species_name, network.fitness()), FontId::default(), Color32::WHITE);
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

fn draw_top_tab(draw_ctx: &mut DrawContext, ctx: &egui::Context, _frame: &mut eframe::Frame, painter: &Painter) -> () {
    let mut x = 0.0;
    let tab_w = 80.;
    let tab_h = draw_ctx.tab_height;
    let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
    let (w, _) = (viewport_rect.width(), viewport_rect.height());

    let (mut min_fitness, mut max_fitness) = (f32::MAX, f32::MIN);
    let average = draw_ctx.evolution
        .species().iter()
        .map(|specie| {
            let average_fitness = specie
                .previous_average_score();

            if average_fitness < min_fitness {
                min_fitness = average_fitness;
            }else if average_fitness > max_fitness {
                max_fitness = average_fitness;
            }

            average_fitness
        })
        .sum::<f32>() / draw_ctx.evolution.species().len() as f32;

    for (index, species) in draw_ctx.evolution.species().iter().enumerate() {
        let x = (index as f32) * tab_w - draw_ctx.species_index as f32 * tab_w + w / 2. - tab_w / 2.;

        if species.previous_average_score() == min_fitness {
            painter.rect_filled(
                Rect { min: pos2(x, 0.), max: pos2(x + tab_w, tab_h) },
                0.0,
                Color32::from_rgba_unmultiplied(255, 10, 10, 15)
            );
        }else if species.previous_average_score() >= max_fitness - (max_fitness - min_fitness) * 0.1 {
            painter.rect_filled(
                Rect { min: pos2(x, 0.), max: pos2(x + tab_w, tab_h) },
                0.0,
                Color32::from_rgba_unmultiplied(10, 255, 10, 15)
            );
        }else if draw_ctx.species_index == index {
            painter.rect_filled(
                Rect { min: pos2(x, 0.), max: pos2(x + tab_w, tab_h) },
                0.0,
                Color32::from_rgba_unmultiplied(255, 255, 255, 5)
            );
        }

        let font = if draw_ctx.species_index == index
            { FontId::new(14., egui::FontFamily::default()) }
            else { FontId::new(12., egui::FontFamily::default()) };
        painter.text(
            pos2(x + tab_w / 2., tab_h / 2. - 10.), Align2::CENTER_CENTER,
            format!("{}", index),
            font.clone(),
            Color32::WHITE
        );
        painter.text(
            pos2(x + tab_w / 2., tab_h / 2. + 4.), Align2::CENTER_CENTER,
            format!("{:.3}", species.previous_average_score()),
            font,
            Color32::WHITE
        );
        painter.text(
            pos2(x + tab_w / 2., tab_h / 2. + 16.), Align2::CENTER_CENTER,
            format!("{}", species.get_name()),
            FontId::new(6., egui::FontFamily::default()),
            Color32::WHITE
        );

        painter.line_segment(
            [(x, 0.).into(), (x, 50.).into()],
            (0.2, Color32::GRAY),
        );
    }
}

fn draw_bottom_tab(draw_ctx: &mut DrawContext, ctx: &egui::Context, _frame: &mut eframe::Frame, painter: &Painter) -> () {
    let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
    let (w, h) = (viewport_rect.width(), viewport_rect.height());
    let tab_h = draw_ctx.tab_height;
    let species = &draw_ctx.evolution.species()[draw_ctx.species_index];

    painter.text(
        pos2(tab_h / 2., h - tab_h/2.), Align2::LEFT_CENTER,
        format!("{:<30} | Fit: {:.3}", species.get_name(), species.previous_average_score()),
        FontId::new(20., egui::FontFamily::Monospace),
        Color32::WHITE
    );
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
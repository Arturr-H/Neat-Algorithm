/* Imports */
use core::f32;
use std::collections::HashMap;
use eframe::{egui::{self, pos2, Align2, Color32, FontId, Key, Painter, Pos2, Rect}, epaint::PathStroke};
use crate::{neural_network::{connection_gene::ConnectionGene, network::NeatNetwork, node_gene::{NodeGene, NodeGeneType}}, trainer::evolution::Evolution};

/* Constants */
const NODE_SIZE: f32 = 5.;

struct DrawContext {
    evolution: Evolution,
    species_index: usize,
    tab_height: f32,

    /// Some(net_idx) if we've double-clicked on a
    /// network and want to enlarge its picture.
    focusing: Option<usize>,
    info: Option<NodeGene>,

    /// What we should save a network as
    save_name: Option<String>,
}
pub fn start_debug_display(evolution: Evolution) -> () {
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Neat Network",
        options,
        Box::new(|_cc| Ok(Box::new(DrawContext {
            evolution,
            species_index: 0,
            tab_height: 50.,
            focusing: None,
            info: None,
            save_name: None
        }))),
    ).unwrap();
}

impl eframe::App for DrawContext {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height() - self.tab_height);
            let mut reset = false;
            draw_top_tab(self, ctx, _frame, painter);
            draw_bottom_tab(self, ctx, _frame, painter);

            ctx.input(|i| {
                if !i.raw.events.is_empty() {
                    for event in &i.raw.events {
                        if let egui::Event::Text(text) = event {
                            if let Some(name) = &mut self.save_name {
                                name.push_str(text);
                            }
                        }
                    }
                }
            });

            if ctx.input(|i| i.key_pressed(Key::Space)) {
                self.evolution.generation();
                reset = true;
            }else if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) {
                self.species_index = self.species_index.checked_sub(1).unwrap_or(0);
                reset = true;
            }else if ctx.input(|i| i.key_pressed(Key::ArrowRight)) {
                self.species_index = (self.species_index + 1).min(self.evolution.species().len() - 1);
                reset = true;
            }else if ctx.input(|i| i.key_pressed(Key::Escape)) {
                reset = true;
            }else if ctx.input(|i| i.key_pressed(Key::S)) {
                if self.save_name.is_none() && self.focusing.is_some() {
                    self.save_name = Some(String::new());
                }
            }
            
            let networks = self.evolution.species()[self.species_index].networks();
            let cols = (networks.len() as f32).sqrt() as usize; // Number of columns
            let rows = (networks.len() + cols - 1) / cols;    // Number of rows
            
            let cell_width = w / cols as f32;
            let cell_height = (h - self.tab_height) / rows as f32;

            let pos = ctx.input(|i| i.pointer.hover_pos()).unwrap_or(pos2(f32::MIN, f32::MIN));
            if self.focusing.is_none() {
                draw_dividers(rows, cols, cell_height, cell_width, w, h, self.tab_height, painter);
            }

            if ctx.input(|i| i.key_pressed(Key::Enter)) {
                if let Some(save_name) = &self.save_name {
                    let focus_index = self.focusing.unwrap();
                    networks[focus_index].save(&save_name);
                    reset = true;
                }
            }
            
            if reset {
                self.focusing = None;
                self.info = None;
                self.save_name = None;
            }

            let mut coordinates = Vec::new();
            for row in 0..rows {
                for col in 0..cols {
                    let x = col as f32 * cell_width;
                    let y = row as f32 * cell_height + self.tab_height;
                    coordinates.push((x, y));
                }
            }
            
            /* Focus and display one network */
            if let Some(focus_index) = self.focusing {
                let network = &networks[focus_index];
                if let Some(info) = draw_network((0., self.tab_height), network, w, h, 60., painter, pos) {
                    self.info = Some(info);
                }
            }

            /* Display all nets */
            else {
                for (index, network) in networks.iter().enumerate() {
                    let coordinate = coordinates[index];
                    let coordinate_end = (coordinate.0 + cell_width, coordinate.1 + cell_height);
                    let hovering = 
                        pos.x < coordinate_end.0 && pos.y < coordinate_end.1
                        && pos.x > coordinate.0 && pos.y > coordinate.1;
                    let clicked = ctx.input(|i| i.pointer.button_released(egui::PointerButton::Primary));
                    let double_clicked = ctx.input(|i| i.pointer.button_double_clicked(egui::PointerButton::Primary));

                    if hovering && double_clicked {
                        self.focusing = Some(index);
                    }else if hovering && clicked {
                        println!("==== {:?} =====", self.evolution.species()[self.species_index].get_name());
                        println!("topology sorted {:?}", network.topological_sort());
                        println!("{:?}", network);
                        println!("Node genes: \n{}\n....", network.node_genes().iter().map(|e| e.verbose_debug() + "\n").collect::<String>());
                    }

                    painter.rect_filled(
                        Rect {
                            min: coordinate.into(),
                            max: pos2(coordinate.0 + cell_width, coordinate.1 + cell_height)
                        },
                        0.0, if hovering {Color32::from_rgba_unmultiplied(255, 255, 255, 1)} else {Color32::TRANSPARENT}
                    );
                    if let Some(info) = draw_network(coordinate, network, cell_width, cell_height, 60., painter, pos) {
                        self.info = Some(info);
                    }
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

    painter.line_segment(
        [(0., draw_ctx.tab_height).into(), (w, draw_ctx.tab_height).into()],
        (0.2, Color32::GRAY),
    );

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

        let line_stroke = if draw_ctx.species_index == index
                            || draw_ctx.species_index == index.checked_sub(1).unwrap_or(0)
                            { (1., Color32::GRAY) } else { (0.2, Color32::GRAY) };
        painter.line_segment(
            [(x, 0.).into(), (x, 50.).into()],
            line_stroke
        );
    }
}

fn draw_bottom_tab(draw_ctx: &mut DrawContext, ctx: &egui::Context, _frame: &mut eframe::Frame, painter: &Painter) -> () {
    let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
    let (w, h) = (viewport_rect.width(), viewport_rect.height());
    let tab_h = draw_ctx.tab_height;
    let species = &draw_ctx.evolution.species()[draw_ctx.species_index];

    if let Some(name) = &draw_ctx.save_name {
        painter.text(
            pos2(tab_h / 2., h - tab_h/2.), Align2::LEFT_CENTER,
            format!("Saving as {name:?}"),
            FontId::new(20., egui::FontFamily::Monospace),
            Color32::WHITE
        );
    }else if let Some(info) = &draw_ctx.info {
        let type_name = match info.node_type() {
            NodeGeneType::Input => "Input",
            NodeGeneType::Output => "Output",
            NodeGeneType::Regular => "Regular",
        };
        painter.text(
            pos2(tab_h / 2., h - tab_h/2.), Align2::LEFT_CENTER,
            format!("::{} w_idx->self{:?} id({}) bias({})", type_name, info.incoming_connection_indexes(), info.id(), info.bias()),
            FontId::new(20., egui::FontFamily::Monospace),
            Color32::WHITE
        );
    }else {
        painter.text(
            pos2(tab_h / 2., h - tab_h/2.), Align2::LEFT_CENTER,
            format!("{:<30} | Fit: {:.3}", species.get_name(), species.previous_average_score()),
            FontId::new(20., egui::FontFamily::Monospace),
            Color32::WHITE
        );
    }
}

fn draw_arrow(painter: &Painter, p1: (f32, f32), p2: (f32, f32), path_stroke: impl Into<PathStroke> + Copy) -> () {
    let rad = f32::consts::PI / 8.;
    let arrow_length = 10.;
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;

    // Midpoint of the line
    let mid_x = (p1.0 + p2.0) / 2.0;
    let mid_y = (p1.1 + p2.1) / 2.0;

    // Normalize direction vector
    let len = (dx * dx + dy * dy).sqrt();
    let unit_dx = dx / len;
    let unit_dy = dy / len;

    // Rotate the vector by +rad and -rad to get the arrowhead directions
    let (arrow_dx1, arrow_dy1) = rotate_vector(unit_dx, unit_dy, rad);
    let (arrow_dx2, arrow_dy2) = rotate_vector(unit_dx, unit_dy, -rad);

    // Scale the rotated vectors by the arrow length
    let arrow1_end = (
        mid_x + arrow_length * arrow_dx1,
        mid_y + arrow_length * arrow_dy1,
    );
    let arrow2_end = (
        mid_x + arrow_length * arrow_dx2,
        mid_y + arrow_length * arrow_dy2,
    );

    painter.line_segment(
        [pos2(mid_x, mid_y), arrow1_end.into()],
        path_stroke
    );
    painter.line_segment(
        [pos2(mid_x, mid_y), arrow2_end.into()],
        path_stroke
    );
}
fn rotate_vector(dx: f32, dy: f32, angle: f32) -> (f32, f32) {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    (
        dx * cos_theta - dy * sin_theta,
        dx * sin_theta + dy * cos_theta,
    )
}

fn draw_dividers(rows: usize, cols: usize, cell_height: f32, cell_width: f32, w: f32, h: f32, tab_height: f32, painter: &Painter) -> () {
    for col in 1..cols + 2 {
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
}

/// Returns a node gene if hovering above it
fn draw_network(
    coordinate: (f32, f32),
    network: &NeatNetwork,
    cell_width: f32, cell_height: f32,
    padding: f32, painter: &Painter,
    mouse: Pos2
) -> Option<NodeGene> {
    let mut return_node = None;
    painter.text(
        pos2(coordinate.0 + cell_width / 2., coordinate.1 + cell_height - 20.),
        Align2::CENTER_CENTER, format!("{:.3}", network.fitness()),
        FontId::default(), Color32::GRAY
    );

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

            /* Hovering over node */
            let hitbox = 5.;
            let hovering = mouse.x > x - hitbox && mouse.x < x + hitbox
                            && mouse.y > y - hitbox && mouse.y < y + hitbox;

            if hovering {
                return_node = Some(node.clone());
            }

            painter.circle_filled(
                egui::Pos2 { x, y },
                if hovering { NODE_SIZE * 1.4 } else { NODE_SIZE },
                col
            );
            painter.text(
                pos2(x + 10., y - 10.),
                Align2::CENTER_CENTER, format!("{}", node_index),
                FontId::default(), Color32::WHITE
            );
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

        let path_stroke = (conn.weight().max(0.2), color);
        draw_arrow(painter, (x1, y1), (x2, y2), path_stroke);
        painter.line_segment(
            [(x1, y1).into(), (x2, y2).into()],
            path_stroke
        );
    }

    return_node
}

/* Imports */
use core::f32;
use std::collections::HashMap;
use eframe::{egui::{self, pos2, Align2, Color32, FontId, Key, Painter, Pos2, Rect}, epaint::PathStroke};
use rand_distr::num_traits::Signed;
use crate::{neural_network::{network::{NeatNetwork, AVERAGE_FITNESS_WINDOW_SIZE}, node_gene::{NodeGene, NodeGeneType}}, trainer::{evolution::Evolution, fitness::FitnessEvaluator}, utils::get_unix_time};

/* Constants */
const NODE_SIZE: f32 = 5.;
const AVERAGE_FITNESS_LEN_GRAPH: usize = 150;
const DRAW_GRAPH_NODE_EACH_NTH_GEN: usize = 25;
struct DrawContext<T: FitnessEvaluator + Send + Sync> {
    evolution: Evolution<T>,
    species_index: usize,
    tab_height: f32,

    /// Some(net_idx) if we've double-clicked on a
    /// network and want to enlarge its picture.
    focusing: Option<usize>,
    info: Option<NodeGene>,

    /// What we should save a network as
    save_name: Option<String>,
    speed_gen: bool,

    /// get_unix_time, last time we generated generation
    previous_gen: u128,
    ms_delay: u128,
    average_fitnesses: [f32; AVERAGE_FITNESS_LEN_GRAPH],

    /// How many times we call `generation` per frame
    /// when in speed gen
    gens_per_frame: usize,
    show_disabled: bool,

    /// For increasing performance, don't draw most things
    hide_all: bool,
    graph_fitnesses: f32
}
pub fn start_debug_display<T: FitnessEvaluator + Send + Sync + 'static>(evolution: Evolution<T>) -> () {
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
            save_name: None,
            speed_gen: false,
            previous_gen: get_unix_time(),
            ms_delay: 0,
            average_fitnesses: [0.0; AVERAGE_FITNESS_LEN_GRAPH],
            gens_per_frame: 1,
            show_disabled: true,
            hide_all: false,
            graph_fitnesses: 0.,
        }))),
    ).unwrap();
}

impl<T: FitnessEvaluator + Send + Sync> eframe::App for DrawContext<T> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height() - self.tab_height);
            let mut reset = false;

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

            if self.speed_gen {
                let new_time = get_unix_time();
                self.ms_delay = new_time - self.previous_gen;
                self.previous_gen = new_time;

                for _ in 0..self.gens_per_frame {
                    if self.evolution.generation() {
                        self.speed_gen = false;
                    };

                    self.graph_fitnesses += self.evolution.species_fitnesses().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                    if self.evolution.get_generation() % DRAW_GRAPH_NODE_EACH_NTH_GEN == 0 {
                        self.average_fitnesses.rotate_left(1);
                        self.average_fitnesses[AVERAGE_FITNESS_LEN_GRAPH - 1] = self.graph_fitnesses / DRAW_GRAPH_NODE_EACH_NTH_GEN as f32;
                        self.graph_fitnesses = 0.;
                    }
                }
                ctx.request_repaint();
            }
            if ctx.input(|i| i.key_pressed(Key::Space)) {
                self.evolution.generation();
                self.graph_fitnesses += self.evolution.species_fitnesses().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                if self.evolution.get_generation() % DRAW_GRAPH_NODE_EACH_NTH_GEN == 0 {
                    self.average_fitnesses.rotate_left(1);
                    self.average_fitnesses[AVERAGE_FITNESS_LEN_GRAPH - 1] = self.graph_fitnesses / DRAW_GRAPH_NODE_EACH_NTH_GEN as f32;
                    self.graph_fitnesses = 0.;
                }
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
            }else if ctx.input(|i| i.key_released(Key::Q)) {
                self.speed_gen = !self.speed_gen;
                self.ms_delay = 0;
                self.previous_gen = get_unix_time();
            }else if ctx.input(|i| i.key_pressed(Key::H)) {
                self.show_disabled = !self.show_disabled;
            }else if ctx.input(|i| i.key_pressed(Key::G)) {
                self.hide_all = !self.hide_all;
            }else if ctx.input(|i| i.key_pressed(Key::Backspace)) {
                self.save_name = self.save_name.as_ref().map(|e| match e.char_indices().next_back() {
                    Some((i, _)) => e[..i].to_string(),
                    None => e.to_string(),
                })
            }
            else if ctx.input(|i| i.key_released(Key::Num1)) { self.gens_per_frame = 1 }
            else if ctx.input(|i| i.key_released(Key::Num2)) { self.gens_per_frame = 2 }
            else if ctx.input(|i| i.key_released(Key::Num3)) { self.gens_per_frame = 3 }
            else if ctx.input(|i| i.key_released(Key::Num4)) { self.gens_per_frame = 4 }
            else if ctx.input(|i| i.key_released(Key::Num5)) { self.gens_per_frame = 5 }
            else if ctx.input(|i| i.key_released(Key::Num6)) { self.gens_per_frame = 6 }
            else if ctx.input(|i| i.key_released(Key::Num7)) { self.gens_per_frame = 7 }
            
            let networks = self.evolution.species()[self.species_index].networks();
            let cols = (networks.len() as f32).sqrt() as usize; // Number of columns
            let rows = (networks.len() + cols - 1) / cols;    // Number of rows
            let (mut _min_fitness, mut max_fitness) = (f32::MAX, f32::MIN);

            if ctx.input(|i| i.key_pressed(Key::F)) {
                let mut best_network = (0, 0); // (species_index, network_index)

                // Find the best network, go through all species and networks
                for (species_index, species) in self.evolution.species().iter().enumerate() {
                    for (network_index, network) in species.networks().iter().enumerate() {
                        let fitness = network.previous_average_fitness();
                        if fitness > max_fitness {
                            max_fitness = fitness;
                            best_network = (species_index, network_index);
                        }
                    }
                }

                self.species_index = best_network.0;
                // self.focusing = Some(best_network.1);
            }

            draw_top_tab(&self, ctx, _frame, painter);
            draw_bottom_tab(&self, ctx, _frame, painter);

            if self.hide_all { return }

            let cell_width = w / cols as f32;
            let cell_height = (h - self.tab_height * 2.) / rows as f32;
            let padding = (cell_height + cell_height) / 2. * 0.2;

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
                if let Some(info) = draw_network(self.focusing, (0., self.tab_height), network, w, h, 60., painter, pos, self.show_disabled) {
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
                        // let mut c = network.clone();
                        // std::thread::spawn(move || {
                        //     // snake_game(&mut c, true);
                        // });
                    }else if hovering && clicked {
                        println!("==== {:?} =====", self.evolution.species()[self.species_index].get_name());
                        // println!("topology sorted {:?}", network.topological_sort());
                        println!("{:?}", network);
                        println!("Node genes: \n{}\n....", network.node_genes().iter().map(|e| e.verbose_debug() + "\n").collect::<String>());
                    }

                    painter.rect_filled(
                        Rect {
                            min: coordinate.into(),
                            max: pos2(coordinate.0 + cell_width, coordinate.1 + cell_height)
                        },
                        0.0, if hovering {
                            Color32::from_rgba_unmultiplied(255, 255, 255, 1)
                        } else {
                            Color32::TRANSPARENT
                        }
                    );
                    if let Some(info) = draw_network(self.focusing, coordinate, network, cell_width, cell_height, padding, painter, pos, self.show_disabled) {
                        self.info = Some(info);
                    }
                }
            }

            draw_fitness_graph(w, h, self.tab_height, &self.average_fitnesses, painter);
        });
    }
}

fn draw_top_tab<T: FitnessEvaluator + Send + Sync>(draw_ctx: &DrawContext<T>, ctx: &egui::Context, _frame: &mut eframe::Frame, painter: &Painter) -> () {
    let tab_w = 80.;
    let tab_h = draw_ctx.tab_height;
    let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
    let (w, _) = (viewport_rect.width(), viewport_rect.height());

    painter.line_segment(
        [(0., draw_ctx.tab_height).into(), (w, draw_ctx.tab_height).into()],
        (0.2, Color32::GRAY),
    );

    let (mut min_fitness, mut max_fitness) = (f32::MAX, f32::MIN);
    let _average = draw_ctx.evolution
        .species().iter()
        .map(|specie| {
            let average_fitness = specie
                .average_fitness();

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

        if species.average_fitness() == min_fitness {
            painter.rect_filled(
                Rect { min: pos2(x, 0.), max: pos2(x + tab_w, tab_h) },
                0.0,
                Color32::from_rgba_unmultiplied(255, 10, 10, 15)
            );
        }else if species.average_fitness() >= max_fitness - (max_fitness - min_fitness) * 0.1 {
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
            format!("{:.3}", species.average_fitness()),
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

fn draw_bottom_tab<T: FitnessEvaluator + Send + Sync>(draw_ctx: &DrawContext<T>, ctx: &egui::Context, _frame: &mut eframe::Frame, painter: &Painter) -> () {
    let viewport_rect = ctx.input(|i: &egui::InputState| i.screen_rect());
    let (_w, h) = (viewport_rect.width(), viewport_rect.height());
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
            format!("::{} w_idx->self{:?} bias({})", type_name, info.incoming_connection_indexes(), info.bias()),
            FontId::new(20., egui::FontFamily::Monospace),
            Color32::WHITE
        );
    }else {
        painter.text(
            pos2(tab_h / 2., h - tab_h/2.), Align2::LEFT_CENTER,
            format!("{}g/f {:<24} | ms: {} | Fit: {:.4} | Generation: {:<8}", draw_ctx.gens_per_frame, species.get_name(), draw_ctx.ms_delay, format!("{:<8}", species.average_fitness()), draw_ctx.evolution.get_generation()),
            FontId::new(20., egui::FontFamily::Monospace),
            Color32::WHITE
        );
    }
}

fn draw_arrow(is_focusing: Option<usize>, weight: f32, painter: &Painter, p1: (f32, f32), p2: (f32, f32), path_stroke: impl Into<PathStroke> + Copy) -> () {
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

    if is_focusing.is_some() {
        painter.text(
            pos2(mid_x, mid_y + 15.),
            Align2::CENTER_CENTER,
            format!("{:.3}", weight),
            FontId::default(),
            Color32::WHITE
        );
    }
    
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
            [(x, tab_height).into(), (x, h - tab_height).into()],
            (0.2, Color32::GRAY),
        );
    }

    painter.line_segment(
        [(0., h).into(), (w, h).into()],
        (0.2, Color32::GRAY),
    );
}

fn draw_fitness_graph(w: f32, h: f32, tab_height: f32, average_fitnesses: &[f32], painter: &Painter) -> () {
    let max_fitness = average_fitnesses.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less)).unwrap_or(&0.0);
    let min_fitness = average_fitnesses.iter().max_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Greater)).unwrap_or(&0.0);
    let value_range = max_fitness - min_fitness;

    /* Holy what a name */
    let amount_of_points = average_fitnesses.len();
    let step_size_x = w / (amount_of_points - 1) as f32;
    
    // println!("{min_fitness} {max_fitness}");
    /* Draw "helper" lines */
    if amount_of_points < 1 { return };

    for i in 0..(amount_of_points - 1) {
        let curr = h - (average_fitnesses[i] - min_fitness) / value_range * (tab_height * 0.9);
        let next = h - (average_fitnesses[i + 1] - min_fitness) / value_range * (tab_height * 0.9);

        let curr_x = i as f32 * step_size_x;
        let next_x = (i + 1) as f32 * step_size_x;

        painter.line_segment(
            [(curr_x, curr).into(), (next_x, next).into()],
            (0.4, Color32::WHITE),
        );
    }

    painter.text(
        pos2(w / 2., h - tab_height / 2.),
        Align2::CENTER_CENTER,
        format!("{:.3}", average_fitnesses[AVERAGE_FITNESS_LEN_GRAPH - 1]),
        FontId::new(28., egui::FontFamily::Monospace),
        Color32::WHITE
    );
}

/// Returns a node gene if hovering above it
fn draw_network(
    is_focusing: Option<usize>,
    coordinate: (f32, f32),
    network: &NeatNetwork,
    cell_width: f32, cell_height: f32,
    padding: f32, painter: &Painter,
    mouse: Pos2,
    show_disabled: bool
) -> Option<NodeGene> {
    let mut return_node = None;
    painter.text(
        pos2(coordinate.0 + cell_width / 2., coordinate.1 + cell_height - cell_height * 0.1),
        Align2::CENTER_CENTER, format!("avg{}={:.3}", AVERAGE_FITNESS_WINDOW_SIZE, network.previous_average_fitness()),
        FontId::default(), Color32::GRAY
    );

    let mut positions: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, node_gene) in network.node_genes().iter().enumerate() {
        // f32 does not implement hash so we transmute
        // it to a u32
        let int = unsafe { std::mem::transmute::<f32, u32>(node_gene.x()) };
        match positions.get(&int) {
            Some(_) => {
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
                NodeGeneType::Input => {
                    /* black = 0.0, white = 1.0 */
                    let clamped_activation = node.activation().clamp(0., 1.);
                    let a = (clamped_activation * 255.) as u8;
                    Color32::from_rgb(a, a, a)
                },
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
        if !show_disabled && !conn.enabled() { continue; }
        let n_in = conn.node_in();
        let n_out = conn.node_out();
        let (x1, y1) = node_positions[&n_in];
        let (x2, y2) = node_positions[&n_out];
        let color = match conn.enabled() {
            true => {
                if conn.weight().is_positive() {
                    Color32::WHITE
                }else {
                    Color32::WHITE
                }
            },
            false => Color32::RED
        };

        let path_stroke = (conn.weight().abs(), color);
        draw_arrow(is_focusing, conn.weight().abs(), painter, (x1, y1), (x2, y2), path_stroke);
        painter.line_segment(
            [(x1, y1).into(), (x2, y2).into()],
            path_stroke
        );
    }

    return_node
}

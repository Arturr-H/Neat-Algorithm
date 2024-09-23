use core::f32;

use eframe::egui::{pos2, CentralPanel, Color32, Context, InputState, Key, Painter};
use rand::{thread_rng, Rng};
use crate::neural_network::network::NeatNetwork;

use crate::utils::find_max_index;

pub struct Pendulum {
    pub origin: Option<vector::Vector>,
    pub position: vector::Vector,
    pub angle: f32,

    pub angular_velocity: f32,
    pub angular_acceleration: f32,

    pub r: f32,
    pub m: f32,
    pub g: f32,
}

impl Pendulum {
    pub fn new(r: f32) -> Pendulum {
        Pendulum {
            origin: None,
            position: vector::Vector::new(0.0, 0.0),

            angle: 0.,//thread_rng().gen_range(-f32::consts::PI..f32::consts::PI),
            angular_velocity: 0.0,
            angular_acceleration: 0.0,

            r,
            m: 1.0,
            g: 0.5,
        }
    }

    pub fn update(&mut self, movement_dir: f32) -> f32 {
        self.angular_acceleration = -1.0 * self.g * self.angle.sin() / self.r;
        self.angular_velocity += self.angular_acceleration + movement_dir * 0.003;
        self.angular_velocity *= 0.99;
        self.angle += self.angular_velocity;
        self.position.set(self.r * self.angle.sin(), self.r * self.angle.cos());
        self.position.add(&self.origin.as_ref().unwrap());
        self.origin.as_mut().unwrap().x -= movement_dir;

        let h = (300. - self.position.y) * 0.0005;
        if h > 0.95 { h*2. } else { h*0.01 }
    }

    pub fn draw(&self, painter: &Painter) {
        painter.line_segment(
            [pos2(self.origin.as_ref().unwrap().x, self.origin.as_ref().unwrap().y), pos2(self.position.x, self.position.y)],
            (3.0, Color32::RED),
        );

        painter.circle_filled(pos2(self.position.x, self.position.y), 20.0, Color32::RED);
    }
    pub fn add_x(&mut self, x: f32) -> () {
        self.origin.as_mut().unwrap().x += x;
    }
    pub fn set_origin(&mut self, origin: vector::Vector) -> () {
        self.origin = Some(origin);
    }

    pub fn display<'a>(self, network: NeatNetwork) -> () {
        let options = eframe::NativeOptions::default();
        let _ = eframe::run_native(
            "Pendulum",
            options,
            Box::new(|_cc| Ok(Box::new(DrawContext {
                pend: self,
                network,
                moves: 0
            }))),
        ).unwrap();
    }
}

impl Default for Pendulum {
    fn default() -> Self {
        Self::new(100.)
    }
}

pub mod vector {
    #[derive(Debug)]
    pub struct Vector {
        pub x: f32,
        pub y: f32,
    }

    impl Vector {
        pub fn new(x: f32, y: f32) -> Vector {
            Vector { x, y }
        }

        pub fn add(&mut self, other: &Vector) -> &Vector {
            self.x += other.x;
            self.y += other.y;
            self
        }

        pub fn set(&mut self, x: f32, y: f32) {
            self.x = x;
            self.y = y;
        }
    }
}

struct DrawContext {
    pend: Pendulum,
    network: NeatNetwork,
    moves: usize,
}
impl eframe::App for DrawContext {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let viewport_rect = ctx.input(|i: &InputState| i.screen_rect());
            let (w, h) = (viewport_rect.width(), viewport_rect.height());

            if self.pend.origin.is_none() {
                self.pend.origin = Some(vector::Vector { x: w / 2., y: h / 2. })
            }

            let input = vec![self.pend.angle, self.pend.angular_acceleration, self.pend.angular_velocity];
            let output = self.network.calculate_output(input);
            let decision = find_max_index(&output);

            if ctx.input(|i| i.key_down(Key::D)) {
                self.pend.update(1.);
            }else if ctx.input(|i| i.key_down(Key::A)) {
                self.pend.update(-1.);
            }else if decision == 0 {
                self.pend.update(-1.);
            }else if decision == 2 {
                self.pend.update(1.);
            }else {
                self.pend.update(0.);
            }

            self.pend.draw(painter);
            // self.pend2.update();
            // self.pend2.draw(graphics);
            ctx.request_repaint();

            

            // let pivot_x = w / 2.0;
            // let pivot_y = h / 2.0;

            // // Calculate pendulum bob position
            // let pendulum_length = 200.0; // Adjust this for visual scale
            // let bob_x = pivot_x + pendulum_length * self.pend.theta.sin();
            // let bob_y = pivot_y + pendulum_length * self.pend.theta.cos();

            // // Draw the pendulum
            // painter.line_segment([pos2(pivot_x, pivot_y), pos2(bob_x, bob_y)], (2.0, Color32::WHITE));
            // painter.circle_filled(pos2(bob_x, bob_y), 20.0, Color32::RED); // Draw the pendulum bob
        });
    }
}

pub fn score_pendulum(network: &mut NeatNetwork) -> f32 {
    let mut score = 0.0;
    let mut pend = Pendulum::default();
    pend.set_origin(vector::Vector { x: 200., y: 200. });
    let mut tries = 0;
    let mut score = 0.0;

    loop {
        let input = vec![pend.angle, pend.angular_acceleration, pend.angular_velocity];
        let output = network.calculate_output(input);
        let decision = find_max_index(&output);
        if decision == 0 {
            score += pend.update(-1.);
        }else if decision == 1 {
            score += pend.update(0.);
        }else if decision == 2 {
            score += pend.update(1.);
        }

        tries += 1;
        if tries > 700 {
            break;
        }
    }

    score
}

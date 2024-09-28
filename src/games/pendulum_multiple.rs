use std::sync::Arc;

/* Imports */
use nannou::prelude::*;
use crate::{neural_network::{activation::NetworkActivations, network::NeatNetwork}, trainer::fitness::FitnessEvaluator};

#[derive(Debug, PartialEq, Clone)]
pub struct XY { x: f32, y: f32 }
impl XY {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn to_nannou(&self) -> nannou::geom::point::Point2 {
        nannou::geom::Vec2::new(
            -self.x as nannou::geom::scalar::Default,
            -self.y as nannou::geom::scalar::Default,
        )
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Polar { pub length: f32, pub angle: f32 }
impl Polar {
    pub fn new(length: f32) -> Self { Self { length, angle: 0. } }
    pub fn to_xy(&self) -> XY {
        let x = self.angle.sin() * self.length;
        let y = self.angle.cos() * self.length;
        XY { x, y }
    }
}

#[derive(Clone)]
struct Model {
    p1: Polar,
    p2: Polar,

    m1: f32,
    m2: f32,

    a1: f32,
    a2: f32,

    v1: f32,
    v2: f32,

    dampening: f32,
    times_above_0p8: usize,
    times_above_thresh: usize,


    gravity: f32,
    network: Option<NeatNetwork>
}
impl Model {
    fn new() -> Model {
        Model {
            p1: Polar::new(150.0),
            p2: Polar::new(150.0),
            m1: 40.0, m2: 40.0, a1: 0.0, a2: 0.0, v1: 0.0, v2: 0.0,
            dampening: 0.01, gravity: 1.0, network: None, times_above_0p8: 1, times_above_thresh: 1,
        }
    }
    fn new_with_model() -> Model {
        Model {
            network: Some(NeatNetwork::new(6, 1, Arc::default(), Arc::default(), NetworkActivations::default(), Arc::default())),
            ..Self::new()
        }
    }

    fn to_input_vec(&self) -> Vec<f32> {
        vec![self.p1.angle.cos(), self.p1.angle.sin(), self.p2.angle.cos(), self.p2.angle.sin(), self.v1, self.v2]
    }
}

#[derive(Clone)]
pub struct DoublePendulumEvaluator {
    model: Model
}
impl DoublePendulumEvaluator {
    pub fn new() -> Self {
        Self { model: Model::new() }
    }
    pub fn run_double_from_file() {
        nannou::app(|_| Model::new_with_model()).simple_window(view).update(update_ai).run();
    }
    pub fn run_double_keyboard() {
        nannou::app(|_| Model::new()).simple_window(view).update(update_keyboard).run();
    }
}
impl FitnessEvaluator for DoublePendulumEvaluator {
    fn run(&mut self, network: &mut NeatNetwork) -> f32 {
        let (score, above) = score_pendulum_multiple(network, &mut self.model);
        if above > 500 {
            self.model.times_above_thresh += 1;
        }
        if self.model.times_above_thresh > 20 {
            self.model.times_above_thresh = 1;
            self.model.gravity += 0.025;
            dbg!(self.model.gravity);
        }

        score
    }
}

fn update_ai(_: &App, m: &mut Model, _update: Update) {
    let inputs = m.to_input_vec();
    let decision = match m.network {
        Some(ref mut e) => e.calculate_output(inputs)[0],
        None => panic!("STupid")
    };
    dbg!(logic_update(m, (decision - 0.5) * 0.005).0);
}
fn update_keyboard(app: &App, m: &mut Model, _update: Update) {
    if app.keys.down.contains(&Key::A) {
        logic_update(m, 0.5*0.005);
    }else if app.keys.down.contains(&Key::D) {
        logic_update(m, -0.5*0.005);
    }else {
        logic_update(m, 0.0);
    }
}

fn logic_update(m: &mut Model, ac: f32) -> (f32, bool) {
    let num = -m.gravity * (2.0 * m.m1 + m.m2) * m.p1.angle.sin()
        - m.m2 * m.gravity * (m.p1.angle - 2.0 * m.p2.angle).sin()
        - 2.0
            * (m.p1.angle - m.p2.angle).sin()
            * m.m2
            * (m.v2 * m.v2 * m.p2.length
                + m.v1 * m.v1 * m.p1.length * (m.p1.angle - m.p2.angle).cos());
    let den =
        m.p1.length * (2.0 * m.m1 + m.m2 - m.m2 * (2.0 * m.p1.angle - 2.0 * m.p2.angle).cos());
    m.a1 = num / den;

    let num = 2.0
        * (m.p1.angle - m.p2.angle).sin()
        * (m.v1 * m.v1 * m.p1.length * (m.m1 + m.m2)
            + m.gravity * (m.m1 + m.m2) * m.p1.angle.cos()
            + m.v2 * m.v2 * m.p2.length * m.m2 * (m.p1.angle - m.p2.angle).cos());
    let den =
        m.p2.length * (2.0 * m.m1 + m.m2 - m.m2 * (2.0 * m.p1.angle - 2.0 * m.p2.angle).cos());
    m.a2 = num / den;

    m.v1 += -ac + m.a1;
    m.v2 += m.a2;
    m.p1.angle += m.v1;
    m.p2.angle += m.v2;
    m.v1 *= 1.0 - m.dampening;
    m.v2 *= 1.0 - m.dampening;

    let s1 = ((f32::cos(m.p1.angle) - 1.) / 2.).abs();
    let s2 = ((f32::cos(m.p2.angle) - 1.) / 2.).abs();
    let s1_over_0p8 = s1 > 0.9;
    let s2_over_0p8 = s2 > 0.9;
    if s1_over_0p8 && s2_over_0p8 {
        m.times_above_0p8 += 1;
    }else {
        m.times_above_0p8 = 1;
    }

    // println!("{} - {}", m.times_above_0p8, (s1 + s2) * m.times_above_0p8.min(50).pow(3) as f32);
    let dist_x = 2. - m.p1.angle.cos().abs() + m.p2.angle.cos().abs();
    let score = (s1 + s2) + (dist_x * 0.5).pow(2);

    (score * m.times_above_0p8 as f32, s1_over_0p8 && s2_over_0p8)
}
fn score_pendulum_multiple(network: &mut NeatNetwork, model: &mut Model) -> (f32, usize) {
    let mut score = 0.0;
    let mut frames = 0;
    let mut above = 0;
    const FRAMES: usize = 750;

    loop {
        let inputs = model.to_input_vec();
        let decision = network.calculate_output(inputs.clone())[0];
        let (score_, above_thresh) = logic_update(model, (decision - 0.5) * 0.005);
        if above_thresh {
            above += 1;
        }
        score += score_;

        if score.is_nan() {
            println!("{:?}", &inputs);
            println!("{}", decision);
            println!("{}", (decision - 0.5) * 0.005);
            println!("{}", model.gravity);
            dbg!("SCORE IS NAN");
            panic!();
            // score = 0.0;
        }

        frames += 1;
        if frames > FRAMES {
            break;
        }
    }

    (score / FRAMES as f32, above)
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);
    let win = app.window_rect().pad_top(300.0);

    let offset1 = model.p1.to_xy().to_nannou();
    let b1 = Rect::from_w_h(model.m1, model.m1)
        .mid_top_of(win)
        .shift(offset1);

    let offset2 = model.p2.to_xy().to_nannou();
    let b2 = Rect::from_w_h(model.m2, model.m2)
        .middle_of(b1)
        .shift(offset2);

    draw.line()
        .start(win.mid_top())
        .end(b1.xy())
        .stroke_weight(3.0)
        .color(WHITE);
    draw.line()
        .start(b1.xy())
        .end(b2.xy())
        .stroke_weight(3.0)
        .color(WHITE);

        // std::thread::sleep(Duration::from_millis(100));
    draw.ellipse().xy(b1.xy()).wh(b1.wh()).color(WHITE);
    draw.ellipse().xy(b2.xy()).wh(b2.wh()).color(WHITE);
    draw.text(&format!("{:.3}", ((f32::cos(model.p1.angle) - 1.) / 2.).abs())).xy(b1.xy()).wh(b1.wh()).color(BLACK);
    draw.text(&format!("{:.3}", ((f32::cos(model.p2.angle) - 1.) / 2.).abs())).xy(b2.xy()).wh(b2.wh()).color(BLACK);

    draw.to_frame(app, &frame).unwrap();
}

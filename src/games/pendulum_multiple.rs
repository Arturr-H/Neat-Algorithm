use nannou::prelude::*;
use rand::{thread_rng, Rng};
use crate::neural_network::network::NeatNetwork;

#[derive(Debug, PartialEq)]
pub struct XY {
    x: f32,
    y: f32,
}

impl XY {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[allow(dead_code)]
    fn round(mut self) -> Self {
        self.x = self.x.round();
        self.y = self.y.round();
        self
    }

    pub fn to_nannou(&self) -> nannou::geom::point::Point2 {
        nannou::geom::Vec2::new(
            -self.x as nannou::geom::scalar::Default,
            -self.y as nannou::geom::scalar::Default,
        )
    }
}

#[derive(Debug, PartialEq)]
pub struct Polar {
    pub length: f32,
    pub angle: f32,
}

impl Polar {
    pub fn new(length: f32) -> Self {
        let angle = thread_rng().gen_range(PI/2.0..PI*2.5);
        Self { length, angle }
    }

    pub fn from_xy(xy: XY) -> Self {
        let length = (xy.x * xy.x + xy.y * xy.y).sqrt();
        let angle = xy.y.atan2(xy.x);

        Self { angle, length }
    }

    pub fn to_xy(&self) -> XY {
        let x = self.angle.sin() * self.length;
        let y = self.angle.cos() * self.length;
        XY { x, y }
    }
}

// PENDULUM
// PENDULUM
// PENDULUM
// PENDULUM

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

    gravity: f32,
    network: NeatNetwork
}
impl Model {
    fn new() -> Model {
        Model {
            p1: Polar::new(150.0),
            p2: Polar::new(150.0),

            m1: 40.0,
            m2: 40.0,

            a1: 0.0,
            a2: 0.0,

            v1: 0.0,
            v2: 0.0,

            dampening: 0.01,
            gravity: 0.08,
            network: NeatNetwork::retrieve("/Users/artur/Desktop/test123"),

            times_above_0p8: 1,
        }
    }

    fn to_input_vec(&self) -> Vec<f32> {
        vec![self.p1.angle.cos(), self.p1.angle.sin(), self.p2.angle.cos(), self.p2.angle.sin(), self.v1, self.v2, self.a1, self.a2]
    }
}

pub struct DoublePendulumVisualizer;
impl DoublePendulumVisualizer {
    pub fn run_double_from_file() {
        nannou::app(|_| Model::new()).simple_window(view).update(update_ai).run();
    }
    pub fn run_double_keyboard() {
        nannou::app(|_| Model::new()).simple_window(view).update(update_keyboard).run();
    }
}


pub fn score_pendulum(value: f32) -> f32 {
    ((f32::cos(value) - 1.) / 2.).abs()
}

fn update_ai(_: &App, m: &mut Model, _update: Update) {
    let inputs = m.to_input_vec();
    let decision = m.network.calculate_output(inputs)[0];
    logic_update(m, (decision - 0.5) * 0.001, m.gravity);
}
fn update_keyboard(app: &App, m: &mut Model, _update: Update) {
    for key in app.keys.down.iter() {
        if *key == Key::Key0 { m.gravity = 0.0 }
        else if *key == Key::Key1 { m.gravity = 0.1 }
        else if *key == Key::Key2 { m.gravity = 0.2 }
        else if *key == Key::Key3 { m.gravity = 0.3 }
        else if *key == Key::Key4 { m.gravity = 0.4 }
        else if *key == Key::Key5 { m.gravity = 0.5 }
        else if *key == Key::Key6 { m.gravity = 0.6 }
        else if *key == Key::Key7 { m.gravity = 0.7 }
        else if *key == Key::Key8 { m.gravity = 0.8 }
        else if *key == Key::Key9 { m.gravity = 0.9 }
    };

    if app.keys.down.contains(&Key::A) {
        logic_update(m, 0.5*0.01, m.gravity);
    }else if app.keys.down.contains(&Key::D) {
        logic_update(m, -0.5*0.01, m.gravity);
    }else {
        logic_update(m, 0.0, m.gravity);
    }
}

fn logic_update(m: &mut Model, ac: f32, gravity: f32) -> (f32, bool) {
    let num = -gravity * (2.0 * m.m1 + m.m2) * m.p1.angle.sin()
        - m.m2 * gravity * (m.p1.angle - 2.0 * m.p2.angle).sin()
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
            + gravity * (m.m1 + m.m2) * m.p1.angle.cos()
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

    let s1 = score_pendulum(m.p1.angle);
    let s2 = score_pendulum(m.p2.angle);
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

    (score * m.times_above_0p8.min(50).pow(2) as f32, s1_over_0p8 && s2_over_0p8)
}
pub fn score_pendulum_multiple(network: &mut NeatNetwork, gravity: f32) -> (f32, usize) {
    let mut model = Model::new();
    let mut score = 0.0;
    let mut frames = 0;
    let mut above = 0;
    const FRAMES: usize = 750;

    loop {
        let inputs = model.to_input_vec();
        let decision = network.calculate_output(inputs)[0];
        let (score_, above_thresh) = logic_update(&mut model, (decision - 0.5) * 0.001, gravity);
        if above_thresh {
            above += 1;
        }
        score += score_;

        if score.is_nan() {
            score = 0.0;
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
    draw.background().color(WHITE);
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
        .color(GRAY);
    draw.line()
        .start(b1.xy())
        .end(b2.xy())
        .stroke_weight(3.0)
        .color(GRAY);

        // std::thread::sleep(Duration::from_millis(100));
    draw.ellipse().xy(b1.xy()).wh(b1.wh()).color(BLACK);
    draw.ellipse().xy(b2.xy()).wh(b2.wh()).color(BLACK);
    draw.text(&format!("{:.3}", score_pendulum(model.p1.angle))).xy(b1.xy()).wh(b1.wh()).color(WHITE);
    draw.text(&format!("{:.3}", score_pendulum(model.p2.angle))).xy(b2.xy()).wh(b2.wh()).color(WHITE);

    draw.to_frame(app, &frame).unwrap();
}

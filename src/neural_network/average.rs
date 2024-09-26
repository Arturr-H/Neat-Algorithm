pub fn exponential_average(values: &[f32], alpha: f32) -> f32 {
    let mut avg = 0.0;
    let mut weight_sum = 0.0;

    for (i, &value) in values.iter().enumerate() {
        let weight = alpha.powi(i as i32);
        avg += value * weight;
        weight_sum += weight;
    }

    if weight_sum != 0.0 {
        avg / weight_sum
    } else {
        0.0
    }
}

extern crate mlkit;

use mlkit::sgd;

fn main() {
    let inputs = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let outputs = vec![7.0, 9.0, 11.0, 13.0, 15.0];
    let coeff = sgd::sgd_optimizer(&inputs, &outputs, 0.055, 0.9, 15);
    println!("coefficients : {:?}", coeff);
}

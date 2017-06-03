extern crate mlkit;
extern crate rand;

use mlkit::sgd;
use rand::Rng;

fn main() {
    let inputs = gen_rand_2d(10, 1);
    let outputs = gen_rand_1d(10);

    let coeff = sgd::sgd_optimizer(&inputs, &outputs, 0.05, 0.6, 10);
    println!("p: {:?}", sgd::curve(&inputs, &coeff));
    println!("v: {:?}", outputs);
}

// generate a 2d matrix of m*n with random values
fn gen_rand_2d (m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut _rng = rand::thread_rng();
    let mut _2d = vec![ vec![]; m ]; // initialize 2d vector with m rows
    for i in 0..m {
        for _ in 0..n {
            (_2d[i]).push(_rng.gen::<f64>());
        }
    }
    return _2d;
}

// generate randomize vector of size m
fn gen_rand_1d (n: usize) -> Vec<f64> {
    let mut _v = vec![];
    let mut _rng = rand::thread_rng();
    for _ in 0..n {
        _v.push(_rng.gen::<f64>());
    }
    return _v;
}


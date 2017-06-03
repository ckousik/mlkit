// Task : Stochastic Gradient Descent Optimizer
// Author : Chinmay Kousik
// Date : 2-JUN-2017
// Version : 0.0.1
//
// TODO: Improve performance for larger data sets
// TODO: parallelization
// TODO: momentum
//
// Parallelization:
// Draw m random samples from the set. Run optimizer on each subset. Average results of each
// parallel iteration. Results should converge as all subsets are drawn from the same data set.
extern crate rand;
use self::rand::Rng;

use std::{f32, u32};

// add 1 to start of each vector
fn add_bias_term(input : &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut _result = Vec::new();
    for i in 0..input.len() {
        let mut _c = input[i].clone();
        _c.insert(0,1.0);
        _result.push(_c);
    }
    return _result;
}

// sgd_diff calculates the difference of the hypothesis and observation at a given point
// x: Vec<f32> : input vector
// w: Vec<f32> : weights
// y: f32 : expected output
fn sgd_diff(x: &Vec<f32>, w: &Vec<f32>, y: f32) -> f32 {
    let hypothesis = x.iter().zip(w.iter()).fold(0.0, |sum, (xj, wj)| sum + (*xj)*(*wj));
    return hypothesis - y;
}

// sgd_optimizer_run takes x and y and returns linear regression coefficients
// x_batch: Vec<Vec<f32>> : inputs
// y_batch: Vec<f32> : outputs
// coeff : Vec<f32> : current coefficients
// lr : f32 : learning rate
// mom: f32 : momentum
fn sgd_optimizer_run(x_batch: &Vec<Vec<f32>>, y_batch: &Vec<f32>, coeff: &Vec<f32>, lr: f32, mom: f32) -> Vec<f32> {
    // shuffle data set
    let x_b = add_bias_term(x_batch);
    let (x_b, y_b) = shuffle_data_set(&x_b, y_batch);
    let mut _w = coeff.clone();

    // initialize momentum vector
    let mut _momentum = vec![0.0; _w.len()];

    for (i,x) in x_b.iter().enumerate(){
        // clone for computing cost function
        for (j,_) in coeff.iter().enumerate() {
            _momentum[j] = mom * _momentum[j] + lr*sgd_diff(x, &_w, y_b[i]) * (&x[j]);
            _w[j] = _w[j] - _momentum[j];
        }
    }
    return _w;
}

// sgd_optimizer takes x,y, learning rate, and number of epochs and uses stochastic gradient
// descent to calculate regression coefficients
// x : Vec<Vec<f32>> : input vector
// y : Vec<f32> : observed outputs
// lr: f32 : learning rate of the optimizer
// mom : f32: momentum
// epochs : u32 : number of times sgd optimization is run
pub fn sgd_optimizer(x: &Vec<Vec<f32>>, y: &Vec<f32>, lr: f32, mom: f32, epochs: u32) -> Vec<f32>{ 
    // number of features + bias
    let mut _guess = vec![1.0; x[0].len() + 1];

    let mut _e = epochs;
    while _e > 0 {
        _guess = sgd_optimizer_run(x,y,&_guess, lr,mom);
        _e = _e - 1;
    }

    return _guess;
}

// Fisher-Yates shuffle algorithm
fn shuffle_data_set (x_data: &Vec<Vec<f32>>, y_data: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut _x = x_data.clone();
    let mut _y = y_data.clone();

    let mut _rng = rand::thread_rng();

    let mut i = x_data.len() - 1;

    while i > 0 {
        let j = _rng.gen::<usize>() % (i+1);
        // swap without borrowing
        let tmp = _x[i].clone();
        _x[i] = _x[j].clone();
        _x[j] = tmp;

        let tmp = _y[i];
        _y[i] = _y[j];
        _y[j] = tmp;

        i = i-1;
    }
    return (_x,_y);
}


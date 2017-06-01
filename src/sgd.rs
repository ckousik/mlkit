// Task : Stochastic Gradient Descent Optimizer
// Author : Chinmay Kousik
// Date : 2-JUN-2017
// Version : 0.0.1
//
// TODO: Improve performance for larger data sets
// TODO: reduce cloning if possible, although only smaller vectors are cloned

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
pub fn sgd_diff(x: &Vec<f32>, w: &Vec<f32>, y: f32) -> f32 {
    let hypothesis = x.iter().zip(w.iter()).fold(0.0, |sum, (xj, wj)| sum + (*xj)*(*wj));
    return hypothesis - y;
}

// sgd_optimizer_run takes x and y and returns linear regression coefficients
// x_batch: Vec<Vec<f32>> : inputs
// y_batch: Vec<f32> : outputs
// coeff : Vec<f32> : current coefficients
// lr : f32 : learning rate
pub fn sgd_optimizer_run(x_batch: &Vec<Vec<f32>>, y_batch: &Vec<f32>, coeff: &Vec<f32>, lr: f32) -> Vec<f32> {
    let x_b = add_bias_term(x_batch);
    let mut _w = coeff.clone();

    for (i,x) in x_b.iter().enumerate(){
        // clone for computing cost function
        let ww = _w.clone();
        for (j,_) in ww.iter().enumerate() {
            _w[j] = _w[j] - lr * sgd_diff(x, &ww, y_batch[i])*(&x[j]);
        }
    }
    return _w;
}

// sgd_optimizer takes x,y, learning rate, and number of epochs and uses stochastic gradient
// descent to calculate regression coefficients
pub fn sgd_optimizer(x: &Vec<Vec<f32>>, y: &Vec<f32>, lr: f32, epochs: u32) -> Vec<f32>{ 
    // number of features + bias
    let mut _guess = vec![1.0; x[0].len() + 1];
    let mut _e = epochs;
    while _e > 0 {
        _guess = sgd_optimizer_run(x,y,&_guess, lr);
        _e = _e - 1;
    }
    return _guess;
}

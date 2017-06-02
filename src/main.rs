extern crate linregress;

use linregress::sgd;

fn main() {
    let inputs = vec![vec![2.0,3.0,4.0], 
        vec![5.0,6.0,7.0],
        vec![5.5,6.2,7.1]];
    let outputs = vec![1.0,2.0,2.1];
    let coeff = sgd::sgd_optimizer(&inputs, &outputs, 0.005, 10);
    println!("coefficients : {:?}", coeff);
}

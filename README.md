# Python Implementation of the Fourier Basis 

This software contains:
1. A Python implementation of the Fourier Basis, in `fourier_fa.py`.
2. An example Sarsa(λ) with linear function approximation implementation, in `sarsa_lambda_linear.py`.
3. Two example RLGym agents that use Sarsa(λ) with the Fourier basis, in `mc_sarsa_lambda.py` (Mountain Car) and `acro_sarsa_lambda.py` (Acrobot).

The first two modules require NumPy, and the two example agents require both Matplotlib and RLGym (tested on RLGym 0.18.0).

Author: [George Konidaris](http://cs.brown.edu/people/gdk/), Brown University

License: GPL 3

## Overview

The Fourier basis is a linear value function approximation scheme for reinforcement learning tasks with continuous state spaces, where the value function _V(s)_ is approximated as a weighted sum of Fourier basis functions φ: 

  > _V(s)_ = Σ<sub>i</sub> w<sub>i</sub> φ<sub>i</sub>(s).

A similar scheme is used to approximate the Q-function _Q(s, a)_. 

The above equation is linear in the weights _w<sub>i</sub>_, which are set via learning. The Fourier basis uses Fourier terms for the basis functions φ<sub>i</sub>:

 > φ<sub>i</sub>(s) = cos(π c<sub>i</sub> . s),

where each _c<sub>i</sub>_ is a coefficient vector (of the same size as the state vector _s_), each element of which is an integer between 0 and some fixed upper bound, called the _order_. The basis functions are constructed by generating all such coefficient vectors. For state dimension _d_ and order _n_, there are _(n+1)<sup>d</sup>_ such coefficient vectors, which makes the Fourier basis (like all linear bases) suitable only for small problems (typically _d <= 7_).

Note that _s_ must be scaled so that **each of its elements lies in [0, 1].**

## Paper

This code implements the Fourier basis as originally described in:

> G.D. Konidaris, S. Osentoski, and P.S. Thomas. Value Function 
Approximation in Reinforcement Learning using the Fourier Basis. 
In *Proceedings of the Twenty-Fifth AAAI Conference on Artificial 
Intelligence*, pages 380-385, August 2011.


If you use the Fourier basis in a research article, please cite the original paper:

    @inproceedings{FourierAAAI2011,
        author = "G.D. Konidaris and S. Osentoski and P.S. Thomas",
        title = "Value Function Approximation in Reinforcement Learning 
                 using the {F}ourier Basis",
        booktitle = "Proceedings of the Twenty-Fifth Conference on  
                     Artificial Intelligence",
        year = 2011,
        pages = "380-385"
    }

## FourierBasis class

The FourierBasis class provides methods for constructing and using a Fourier basis.

### Construction

To construct a Fourier basis, you can supply an order and a dimension. For example:

    fb = FourierBasis(order=3, d=2)
 ... creates a basis of order 3, for a 2-dimensional state space.
 Therefore:

    print(fb.get_coefficients())
outputs:

    [[0 0]
     [0 1]
     [0 2]
     [0 3]
     [1 0]
     [1 1]
     [1 2]
     [1 3]
     [2 0]
     [2 1]
     [2 2]
     [2 3]
     [3 0]
     [3 1]
     [3 2]
     [3 3]]

You can also ask for a specific coefficient:

    print(fb.get_coefficient(3))
returns
    
    [0 3]

Alternatively, for finer grained control over the basis you can supply a list of orders, and the dimension will be set to the length of the list:

    fb = fourier_fa.FourierBasis(order=[3, 1])
 
 ... creates a Fourier basis up to order 3 on the first state variable, and up to order 1 on the second state variable. Now

    print(fb.get_coefficients())
returns

    [[0 0]
     [0 1]  
     [1 0]
     [1 1]
     [2 0]
     [2 1]
     [3 0]
     [3 1]]

Additionally, in either case you can use the `independent` boolean parameter (which defaults to `False`) to construct a basis with independent Fourier coefficients (i.e., where only one coefficient is non-zero at a time):

    fb = FourierBasis(order=3, d=2, independent=True)
    print(fb.coefficients)
... outputs:

    [[0. 0.]
     [1. 0.]
     [2. 0.]
     [3. 0.]
     [0. 1.]
     [0. 2.]
     [0. 3.]]

An independent basis is much smaller that a dependent basis, but is a much more restricted function approximation class.

### Evaluation

A Fourier basis instance can be evaluated by passing a state into the 'evaluate' function as follows:

    fb = FourierBasis(order=1, d=2)
    fb.evaluate([0.2, 0.7])
 ... returns a numpy array with a real value for each basis function (i.e., coefficient). In this case, the array contains 4 values:

    array([ 1.        , -0.58778525,  0.80901699, -0.95105652])

These outputs are intended to serve as the feature values for a linear function approximator (i.e., multiplied by weights and then summed). 

### Gradient Factors

When using the Fourier basis in a gradient descent framework, the original paper suggests scaling the gradients by one over the Euclidean norm of the coefficient vector (setting the all-zero coefficient factor to 1). These factors can be accessed using `get_gradient_factors` and `get_gradient_factor`.

## SarsaLambdaLinearFA class and Examples

The `SarsaLambdaLinearFA` class is a simple linear online Sarsa(λ) implementation; example domain implementations for Acrobot (`acro_sarsa_lambda.py`) and Mountain Car (`mc_sarsa_lambda.py`) for reference.
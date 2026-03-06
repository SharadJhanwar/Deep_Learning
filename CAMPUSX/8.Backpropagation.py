"""
Backpropagation (Introduction)

Backpropagation is the core learning algorithm used in training neural networks.
It is a method to compute how much each weight in the network contributed to the
overall error, and then update those weights to reduce the error.

-----------------------------------------------------------
1. WHY DO WE NEED BACKPROPAGATION?
-----------------------------------------------------------
A neural network makes predictions by passing data forward (forward pass).
But after knowing the output is wrong, how do we adjust the weights?

We need:
1. A way to measure error (Loss Function)
2. A method to compute how each weight caused the error
3. A rule to update weights in the correct direction (Gradient Descent)

Backpropagation + Gradient Descent = Learning

-----------------------------------------------------------
2. CORE IDEA
-----------------------------------------------------------
Backpropagation uses calculus (chain rule) to compute the gradient of the loss
with respect to every weight in the network.

Steps:
1. Forward pass → compute predicted output
2. Compute loss (difference between prediction and actual value)
3. Backward pass → compute gradients (partial derivatives)
4. Update weights:
       weight_new = weight_old - learning_rate * gradient

-----------------------------------------------------------
3. CHAIN RULE (Very Important!)
-----------------------------------------------------------
If output depends on intermediate values, we apply the chain rule.

Example:
    y = f(g(x))
    dy/dx = (dy/dg) * (dg/dx)

Neural networks have many layers stacked like f(g(h(x))), so chain rule is used
repeatedly to compute gradients back from output to input.

-----------------------------------------------------------
4. SIMPLE NUMERICAL EXAMPLE
-----------------------------------------------------------
Suppose a network has one neuron:

    input: x = 2
    weight: w = 3
    bias: b = 1
    activation: y = w*x + b

Actual target: t = 15

Forward pass:
    y = 3*2 + 1 = 7

Loss (L):
    L = (y - t)^2 = (7 - 15)^2 = 64

Backward pass:
We compute derivative dL/dw.

L = (wx + b - t)^2

dL/dw = 2*(wx + b - t)*x
       = 2*(7 - 15)*2
       = 2*(-8)*2
       = -32

Update weight:
    w_new = w - lr * dL/dw
If lr (learning rate) = 0.01:
    w_new = 3 - 0.01*(-32)
           = 3 + 0.32
           = 3.32

The model adjusts w upward because it needs a larger output.

-----------------------------------------------------------
5. BACKPROPAGATION IN A MULTI-LAYER NETWORK
-----------------------------------------------------------
Each layer receives gradients from the layer ahead and passes them backward.
For layer L:
    dL/dW_L = delta_L * activation_(L-1)

Where delta_L is the error term for that layer.

The idea:
• error at output layer → push backward
• each layer computes its own contribution
• gradients flow backwards until input layer

This process repeats for many epochs until the loss becomes small.

-----------------------------------------------------------
6. KEY POINTS TO REMEMBER
-----------------------------------------------------------
✓ Backpropagation computes gradients using the chain rule
✓ It works layer-by-layer from output → input
✓ It works with any differentiable activation function
✓ Uses Gradient Descent (or variants like Adam) to update weights
✓ It is the backbone of neural network training

-----------------------------------------------------------
7. MINI IMPLEMENTATION (MANUAL BACKPROP FOR 1 NEURON)
-----------------------------------------------------------

import numpy as np

# Inputs and target
x = 2
t = 15

# Initial weight and bias
w = 3.0
b = 1.0
lr = 0.01   # learning rate

# Forward pass
y = w*x + b
loss = (y - t)**2

# Backward pass
dL_dy = 2*(y - t)
dL_dw = dL_dy * x
dL_db = dL_dy

# Update
w -= lr * dL_dw
b -= lr * dL_db

print("Updated weight:", w)
print("Updated bias:", b)
print("Loss:", loss)

-----------------------------------------------------------
8. SUMMARY
-----------------------------------------------------------
Backpropagation = Fast algorithm to compute gradients.
It uses calculus (chain rule) and gradient descent to update weights.
It is the backbone of training all neural networks including CNNs, RNNs, and Transformers.

"""
